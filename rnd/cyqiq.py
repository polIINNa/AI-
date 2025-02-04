# basics
from __future__ import annotations

import json
import os
import httpx
from typing import TypedDict, Annotated, Sequence, Optional, Dict
import operator

from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END

from data.prompt import SYSTEM_PROMPT


load_dotenv()

if __name__ == '__main__':

    with open('/Users/21109090/Downloads/deposits_2.json', 'r') as f:
        data = json.load(f)
    df_dict = []
    for deposit_name in data.keys():
        d = {'Название': deposit_name}
        for key in data[deposit_name].keys():
            d[key] = data[deposit_name][key]
        df_dict.append(d)
    df = pd.json_normalize(df_dict)
    df = df.fillna("")
    dfs_dict = {'df_deposit': df}

    # Парсим цепочку действий для получения действия, которое надо выполнять в текущий момент
    def get_action(chain: str, actions_history: Sequence[str]):
        actions = chain.split('->')
        if "<BEGIN>" in chain:
            action = actions[1].strip()
        else:
            prev_action = actions_history[-1]
            action = actions[0].strip()
            if prev_action == action:
                if len(actions) > 2:
                    action = actions[1].strip()
                else:
                    action = prev_action
        return action


    @tool
    def view_pandas_dataframes(
            df_name: Annotated[str, "Название таблицы, которую надо посмотреть"]):
        """
        Инструмент для просмотра данных в таблице
        """
        df = dfs_dict[df_name]
        return f"{df.head(20).to_markdown()}".strip()


    @tool
    def evaluate_pandas_chain(
            chain: Annotated[str, "Цепочка действий, которые применяются к pandas датафрейму, например df1.groupby('Минимальная ставка').mean() -> df1.sort_values() -> <END>"],
            inter: Annotated[Optional[Dict], "Промежуточная таблица"],
            actions_history: Annotated[Sequence[str], "История успешных действий по преобразованию pandas датафрейму"]):
        """
        Инструмент для выполнения цепочки
        """
        df = dfs_dict['df_deposit']
        action = get_action(chain=chain, actions_history=actions_history)
        if inter is not None:
            inter_df = pd.DataFrame(inter)
        else:
            inter_df = None
        prev_inter_df = inter_df
        print(f'Операция для выполнения: {action}')
        try:
            inter = eval(action, {'df_deposit': df, 'inter': inter_df})
            if inter is None or inter.isna().all().all():
                return 'Пустой датафрейм', action, prev_inter_df
            else:
                return 'Success', action, inter
        except Exception as e:
            if str(e) == "invalid syntax (<string>, line 1)":
                return 'Ошибка: нельзя использовать операцию присваивания "=" ', action, prev_inter_df
            else:
                return f"В процессе работы возникла ошибка: {str(e)}", action, prev_inter_df

    tools = [evaluate_pandas_chain, view_pandas_dataframes]
    tool_executor = ToolExecutor(tools)
    functions = [convert_to_openai_function(t) for t in tools]  # из тулов langchain получаем функции для gpt openai

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), MessagesPlaceholder(variable_name="messages")])
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    llm = ChatOpenAI(model_name="gpt-4o", http_client=httpx.Client(proxies=os.getenv('OPENAI_PROXY')),
                     openai_api_key=os.getenv('OPENAI_API_KEY'))
    llm_chain = prompt | llm.bind_functions(functions)


    def call_model(state):
        return {"messages": [llm_chain.invoke(state)]}  # в llm отправляется весь чат (ChatPromptTemplate)


    def should_continue(state):
        last_message = state['messages'][-1]
        print('should_continue: ', last_message)
        if "function_call" not in last_message.additional_kwargs:
            print('нет вызова тула')
            return "end"
        else:
            print('есть вызов тула')
            return "continue"


    def call_tool(state):
        last_message = state['messages'][-1]  # последнее сообщение содержит вызов функции
        tool_input = last_message.additional_kwargs["function_call"]["arguments"]
        tool_input_dict = json.loads(tool_input)
        if last_message.additional_kwargs['function_call']['name'] == 'view_pandas_dataframes':
            print('Вызов tool view_pandas_dataframes')
            action = ToolInvocation(
                tool='view_pandas_dataframes',
                tool_input=tool_input_dict
            )
            result = tool_executor.invoke(action)
            function_message = FunctionMessage(content=str(result), name=action.tool)
            return {"messages": [function_message]}
        elif last_message.additional_kwargs['function_call']['name'] == 'evaluate_pandas_chain':
            print('Вызов tool evaluate_pandas_chain')
            if state['inter'] is not None:
                tool_input_dict['inter'] = state['inter'].to_dict()
            else:
                tool_input_dict['inter'] = None
            tool_input_dict['actions_history'] = state['actions']
            action = ToolInvocation(
                tool='evaluate_pandas_chain',
                tool_input=tool_input_dict
            )
            result = tool_executor.invoke(action)
            response, attempted_action, inter = result[0], result[1], result[2]
            print(response)
            if inter is None:
                print('None DF')
                non_df_info = f"""
                Последнее примененное действие: 
                {attempted_action}

                Таблица после выполнения действия: None

                Ты должен скорректировать свои действия и продолжить цепочку для получения таблицы, из которой можно ответить на вопрос:
                {state['question']}

                Список успешно примененных действий: 
                {state['actions']}
                
                """
                function_message = FunctionMessage(content=str(non_df_info), name=action.tool)
                return {"messages": [function_message]}
            else:
                if 'Success' in response:
                    if isinstance(inter, pd.DataFrame) is True:
                        print('SUCCESS DF')
                        print(inter)
                        success_info_df = f"""
                        Последнее примененное действие: 
                        {attempted_action}

                        Таблица после выполнения действия: 
                        {inter.head(10).to_markdown()}

                        Ты должен дальше продолжить цепочку для получения таблицы, из которой можно ответить на вопрос:
                        {state['question']}

                        Список успешно примененных действий: 
                        {state['actions']}    
                        
                        """
                        function_message = FunctionMessage(content=str(success_info_df), name=action.tool)
                        return {"messages": [function_message], "actions": [attempted_action], "inter": inter}
                    else:
                        print('SUCCESS NO DF')
                        print(inter)
                        success_info_no_df = f"""                        
                        Результат выполнения действия: 
                        {inter}

                        Ты должен ответить на вопрос:
                        {state['question']} 

                        """
                        function_message = FunctionMessage(content=str(success_info_no_df), name=action.tool)
                        return {"messages": [function_message], "actions": [attempted_action]}

                else:
                    print('Error')
                    error_info = f"""
                    Последнее примененное действие: 
                    {attempted_action}

                    Сообщение об ошибке:
                    {response}

                    Ты должен скорректировать свои действия и продолжить цепочку для получения таблицы, из которой можно ответить на вопрос:
                    {state['question']}

                    Таблица ДО выполнения последнего действия:
                    {inter.head(10).to_markdown()}

                    Список успешно примененных действий: 
                    {state['actions']}
                    
                    """
                    function_message = FunctionMessage(content=str(error_info), name=action.tool)
                    return {"messages": [function_message]}

    # Определение состояний графа
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        actions: Annotated[Sequence[str], operator.add]
        inter: pd.DataFrame
        question: str
        memory: str


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

    user_query = "Какие вклады можно открыть сроком более 1 года?"
    inputs = {"messages": [HumanMessage(content=user_query)], "actions": ["<BEGIN>"], "question": user_query,
              "memory": ""}
    for output in app.stream(inputs, {"recursion_limit": 30}):
        for key, value in output.items():
            if key == "agent":
                print("🤖 Agent working...")
            elif key == "tool":
                if value["messages"][0].name == "view_pandas_dataframes":
                    print("🛠️ Current action: viewing dataframes")
                else:
                    if "actions" in value.keys():
                        print('action')
                    else:
                        print(f"⚠️ An error occured or empty dataframe, retrying...")
            else:
                print("🏁 Finishing up...")
            print("---")
            pass
