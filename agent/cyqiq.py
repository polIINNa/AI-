from __future__ import annotations

import os
import json
from pathlib import Path
from logging import getLogger
from typing import Annotated, Sequence

import pandas as pd

from langchain import hub
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import FunctionMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent._typing import AgentState
from .utils import get_last_chains, get_action

DATA_DIR = Path(__file__).parent.parent / 'data'
CYQIQ_DATA_DIR = Path(__file__).parent.parent / 'cyqiq_data'

MODULE_LOGGER = getLogger('cyqiq_logger')

with open(DATA_DIR / 'deposits_2.json', 'r') as f:
    data = json.load(f)
output = []
for deposit_name in data.keys():
    d = {'Название': deposit_name}
    for key in data[deposit_name].keys():
        d[key] = data[deposit_name][key]
    output.append(d)
df = pd.json_normalize(output)
df_list = [df]
df_dic = {'df_deposit': df}


# tool для получения информации о таблице
@tool
def view_pandas_dataframes(
        df_names_list: Annotated[
            Sequence[str], "List of maximum 3 pandas dataframes you want to look at, e.g. [df1, df2, df3]"]):
    """Use this to view the head(10) of dataframes to answer your question"""

    markdown_str = "Here are .head(10) of the dataframes you requested to see:\n"
    for df in df_names_list:
        df_head = df_dic[df].head(10).to_markdown()
        markdown_str += f"{df}:\n{df_head}\n"

    markdown_str = markdown_str.strip()
    return markdown_str


# tool для исполнения действия из цепочки
@tool
def evaluate_pandas_chain(
        chain: Annotated[str, "Цепочка действий, которые применяются к pandas датафрейму, "
                              "например df1.groupby('age').mean() -> df1.sort_values() -> <END>"],
        inter):
    """
    Evaluate a sequence of actions applied to a pandas dataframe.

    Arguments:
    chain -- Цепочка действий, которые применяются к pandas датафрейму, например df1.groupby('age').mean() -> df1.sort_values() -> <END>
    inter_df -- Промежуточный pandas DataFrame

    Returns:
    Результаты выполнения действий из цепочки, текущая операция, обновленный промежуточный DataFrame.
    """
    action = get_action(actions=chain)
    MODULE_LOGGER.debug(f'РАБОТА TOOL evaluate_pandas_chain. Операция: {action}')
    try:
        upd_inter = eval(action, {"inter": inter, "df_dic": df_dic})
        if upd_inter is None:
            return 'Empty dataframe', action, inter
        else:
            return 'Success', action, upd_inter
    except Exception as e:
        return f"An exception occured: {e}", action, inter


def get_answer(user_input: str, logger=MODULE_LOGGER) -> str | None:
    # questions_str = """
    # df_deposit: This dataset presents various types of investment products offered by different financial institutions, with details on interest rates, minimum deposit amounts, and interest payment frequency, highlighting the diverse options available for potential investors.
    # """
    questions_str = """
    df_deposit: Этот набор данных представляет различные виды инвестиционных продуктов, предлагаемых различными финансовыми учреждениями, с указанием процентных ставок, минимальных сумм депозитов и частоты выплаты процентов, подчеркивая разнообразие доступных потенциальным инвесторам вариантов.
    """

    tools = [evaluate_pandas_chain, view_pandas_dataframes]
    tool_executor = ToolExecutor(tools)
    functions = [convert_to_openai_function(t) for t in tools]  # из тулов langchain получаем функции для gpt openai

    SYSTEM_PROMPT = hub.pull("hrubyonrails/multi-cot").messages[0].prompt.template
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), MessagesPlaceholder(variable_name="messages")])
    prompt = prompt.partial(num_dfs=len(df_list))
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    prompt = prompt.partial(questions_str=questions_str)
    # passing in past successful queries
    chain_examples = ""
    if isinstance(get_last_chains(), pd.DataFrame):
        for index, row in get_last_chains()[["query", "chain"]].iterrows():
            chain_examples += f'Question: {row["query"]}\nChain: {row["chain"]}\n\n'
    prompt = prompt.partial(chain_examples=chain_examples)

    llm = ChatOpenAI(model_name="gpt-4o", openai_proxy=os.getenv('OPENAI_PROXY'),
                     openai_api_key=os.getenv('OPENAI_API_KEY'))
    llm_chain = prompt | llm.bind_functions(functions)

    def call_model(state):
        return {"messages": [llm_chain.invoke(state)]}  # в llm отправляется весь чат (ChatPromptTemplate)

    def should_continue(state):
        last_message = state['messages'][-1]
        logger.debug(f'should_continue: {last_message}')
        if "function_call" not in last_message.additional_kwargs:
            logger.debug('нет вызова тула')
            return "end"
        else:
            logger.debug('есть вызов тула')
            return "continue"

    def call_tool(state):
        last_message = state['messages'][-1]  # последнее сообщение содержит вызов функции
        tool_input = last_message.additional_kwargs["function_call"]["arguments"]
        tool_input_dict = json.loads(tool_input)
        tool_input_dict['inter'] = state['inter']
        if last_message.additional_kwargs['function_call']['name'] == 'view_pandas_dataframes':
            logger.debug('Вызов tool view_pandas_dataframes')
            action = ToolInvocation(
                tool='view_pandas_dataframes',
                tool_input=tool_input_dict
            )
            result = tool_executor.invoke(action)
            function_message = FunctionMessage(content=str(result), name=action.tool)
            return {"messages": [function_message]}
        elif last_message.additional_kwargs['function_call']['name'] == 'evaluate_pandas_chain':
            logger.debug('Вызов tool evaluate_pandas_chain')
            action = ToolInvocation(
                tool='evaluate_pandas_chain',
                tool_input=tool_input_dict
            )
            result = tool_executor.invoke(action)
            response, attempted_action, inter = result[0], result[1], result[2]
            logger.debug(f'RESULT {inter}')
            if inter is None:
                logger.debug('None DF')
                non_df_info = f"""
                You have previously performed the actions: 
                {state['actions']}

                Attempted action: 
                {attempted_action}

                Response after attempted_action: 
                {response}
                
                Dataframe after attempted_action: None

                You must correct your approach and continue until you can answer the question:
                {state['question']}

                Continue the chain with the following format: action_i -> action_i+1 ... -> <END>
                """
                logger.debug(response)
                function_message = FunctionMessage(content=str(non_df_info), name=action.tool)
                return {"messages": [function_message]}
            else:
                if 'Success' in response:
                    logger.debug('Success')
                    success_info = f"""
                    You have previously performed the actions: 
                    {state['actions']}

                    Attempted action: 
                    {attempted_action}

                    Dataframe after attempted_action:
                    inter.head(10).to_markdown()

                    You must continue until you can answer the question:
                    {state['question']}

                    Continue the  chain with the following format: action_i -> action_i+1 ... -> <END>
                    """
                    function_message = FunctionMessage(content=str(success_info), name=action.tool)
                    return {"messages": [function_message], "actions": [attempted_action], "inter": inter}
                else:
                    logger.debug('Error')
                    error_info = f"""
                    You have previously performed the actions: 
                    {state['actions']}

                    Attempted action: 
                    {attempted_action}

                    Response after attempted_action:
                    {response}

                    Dataframe before attempted_action:
                    {inter.head(10).to_markdown()}

                    You must correct your approach and continue until you can answer the question:
                    {state['question']}

                    Continue the chain with the following format: action_i -> action_i+1 ... -> <END>
                    """
                    function_message = FunctionMessage(content=str(error_info), name=action.tool)
                    return {"messages": [function_message]}

    # Инициализируем граф
    workflow = StateGraph(AgentState)
    # Определяем ноды графа, между которыи будет происходить работа
    workflow.add_node('agent', call_model)
    workflow.add_node('tool', call_tool)
    # Устанавливаем начальное состояние
    workflow.set_entry_point('agent')
    # Устанавливаем ребро, которая будет проверять надо ли продролажть вызывать инструменты или нет
    workflow.add_conditional_edges(
        'agent',  # вершина, после которой будет вызвано условное ребро
        should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "tool",
            # Otherwise we finish.
            "end": END
        }
    )
    # Добавляем обычное ребро, которое связывает ноду agent и tool. Agent вызывается после tool
    workflow.add_edge('tool', 'agent')
    app = workflow.compile()

    inputs = {"messages": [HumanMessage(content=user_input)], "actions": ["<BEGIN>"], "question": user_input,
              "memory": ""}
    for output in app.stream(inputs, {"recursion_limit": 40}):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            if key == "agent":
                logger.info("🤖 Agent working...")
            elif key == "tool":
                if value["messages"][0].name == "view_pandas_dataframes":
                    logger.info("🛠️ Current action: viewing dataframes")
                else:
                    if "actions" in value.keys():
                        logger.info(f"🛠️ Current action: {value['actions']}")
                    else:
                        logger.info("⚠️ An error occured or empty dataframe, retrying...")
            else:
                logger.info("🏁 Finishing up...")
            logger.info("---")
            pass

    return output['agent']['messages'][0].content.replace('<END>', '')
