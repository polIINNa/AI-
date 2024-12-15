
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

from data.prompt import OTR_SYSTEM_PROMPT

if __name__ == '__main__':

    load_dotenv()
    df = pd.read_excel('/Users/21109090/Desktop/ai_ консультант_таблицы/ОТР Транспорт.xlsx')
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    otr_transport = df.loc[:, ~df.columns.str.contains("СрЗНАЧ|Динамика")].iloc[:, :43]
    new_columns = [col for col in otr_transport.columns]
    otr_transport.columns = new_columns
    dfs_dict = {'otr_transport': otr_transport}


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
            chain: Annotated[
                str, "Цепочка действий, которые применяются к pandas датафрейму, например df1.groupby('Минимальная ставка').mean() -> df1.sort_values() -> <END>"]):
        """
        Инструмент для выполнения цепочки
        """
        actions = chain.split('->')
        if '<END>' in chain:
            actions.pop()
        # пытаемся выполнить всю цепочку
        success_actions = []
        inter = None
        prev_inter = inter
        action = actions[0]
        try:
            print('Старт исполнения цепочки')
            for action in actions:
                print('Текущая операция', action)
                inter = eval(action, {'otr_transport': otr_transport, 'inter': prev_inter})
                if inter is None:
                    print(f'В процессе операции {action} получился пустой датафрейм')
                    return 'None DF', success_actions, prev_inter, action
                success_actions.append(action)
                prev_inter = inter
            print('Ура, цепочка исполнена успешно')
            return 'Success', success_actions, inter
        except Exception as e:
            if str(e) == "invalid syntax (<string>, line 1)":
                return 'Ошибка: нельзя использовать операцию присваивания "=" ', success_actions, prev_inter, action
            return f'Error: {str(e)}', success_actions, prev_inter, action


    tools = [evaluate_pandas_chain, view_pandas_dataframes]
    tool_executor = ToolExecutor(tools)
    functions = [convert_to_openai_function(t) for t in tools]  # из тулов langchain получаем функции формата openai

    prompt = ChatPromptTemplate.from_messages(
        [("system", OTR_SYSTEM_PROMPT), MessagesPlaceholder(variable_name="messages")])
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    gpt = ChatOpenAI(model_name="gpt-4o-mini", http_client=httpx.Client(proxies=os.getenv('OPENAI_PROXY')),
                     openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0)

    llm_chain = prompt | gpt.bind_functions(functions)


    def call_model(state):
        return {"messages": [llm_chain.invoke(state)]}  # в llm отправляется весь чат (ChatPromptTemplate)


    def should_continue(state):
        last_message = state['messages'][-1]
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        else:
            return "continue"


    def call_tool(state):
        last_message = state['messages'][-1]  # последнее сообщение содержит вызов функции
        print('---LAST MESSAGE: ', last_message)
        tool_input = last_message.additional_kwargs["function_call"]["arguments"]
        print('--- АРГУМЕНТЫ ФУНКЦИИ: ', tool_input)
        tool_input_dict = json.loads(tool_input)

        if last_message.additional_kwargs['function_call']['name'] == 'view_pandas_dataframes':
            print('Вызов tool view_pandas_dataframes')
            tool_calling = ToolInvocation(
                tool='view_pandas_dataframes',
                tool_input=tool_input_dict
            )
            result = tool_executor.invoke(tool_calling)
            function_message = FunctionMessage(content=str(result), name=tool_calling.tool)
            return {"messages": [function_message]}
        elif last_message.additional_kwargs['function_call']['name'] == 'evaluate_pandas_chain':
            print('Вызов tool evaluate_pandas_chain')
            tool_calling = ToolInvocation(
                tool='evaluate_pandas_chain',
                tool_input=tool_input_dict
            )
            result = tool_executor.invoke(tool_calling)
            response_message, res_df, success_actions = result[0], result[2], result[1]
            if response_message == 'Success':
                print('!!! SUCCESS')
                success_info = f"""                        
                Результат выполнения цепочки: 
                {res_df}

                Ты должен ответить на вопрос:
                {state['question']} 
                
                Список успешно примененных действий: 
                {success_actions}
                
                """
                print(res_df)
                function_message = FunctionMessage(content=str(success_info), name=tool_calling.tool)
                return {"messages": [function_message]}
            elif response_message == 'None DF':
                print('!!! NONE DF')
                non_df_info = f"""
                Последнее примененное действие: 
                {result[3]}

                Таблица после выполнения действия: None

                Ты должен скорректировать свои действия и продолжить цепочку для получения таблицы, из которой можно ответить на вопрос:
                {state['question']}

                Список успешно примененных действий: 
                {success_actions}

                """
                function_message = FunctionMessage(content=str(non_df_info), name=tool_calling.tool)
                return {"messages": [function_message]}
            else:
                if isinstance(res_df, pd.DataFrame) is False:
                    print('!!! ERROR NO DF')
                    error_info = f"""
                    Последнее примененное действие: 
                    {result[3]}

                    Результат после выполнения действия: 
                    {res_df}
                    
                    Сообщение об ошибке:
                    {response_message}

                    Ты должен скорректировать свои действия и продолжить цепочку для получения таблицы, из которой можно ответить на вопрос:
                    {state['question']}

                    Список успешно примененных действий: 
                    {success_actions}

                    """
                else:
                    print('!!! ERROR')
                    error_info = f"""
                    Последнее примененное действие: 
                    {result[3]}
    
                    Сообщение об ошибке:
                    {response_message}
    
                    Ты должен скорректировать свои действия и продолжить цепочку для получения таблицы, из которой можно ответить на вопрос:
                    {state['question']}
    
                    Таблица ДО выполнения последнего действия:
                    {res_df.head(10).to_markdown()}
    
                    Список успешно примененных действий: 
                    {success_actions}
    
                    """
                function_message = FunctionMessage(content=str(error_info), name=tool_calling.tool)
                return {"messages": [function_message]}

    # Определение состояний графа
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        question: str


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
            "continue": "tool",
            "end": END
        }
    )
    # Добавляем обычное ребро, которое связывает ноду agent и tool. Agent вызывается после tool
    workflow.add_edge('tool', 'agent')
    app = workflow.compile()

    user_query = "Какой объем выручки у компании с inn= 360400269820 в марте?"
    inputs = {"messages": [HumanMessage(content=user_query)], "question": user_query}
    answer = ''
    for output in app.stream(inputs, {"recursion_limit": 30}):
        for key, value in output.items():
            if key == 'agent':
                print('🤖 Получен ответ агента AIMessage 🤖')
                aim = value['messages'][0]
                if aim.additional_kwargs != {}:
                    print(f'Агент вызвал функцию: {aim.additional_kwargs}')
                else:
                    answer = aim.content
                    print(f'Агент сформировал сообщение без вызова функции: {aim.content}')
            else:
                fm = value["messages"][0]
                print(f'🛠️ Отработала функция {fm.name}')
                if 'Сообщение об ошибке' in fm.content:
                    print('⚠️ Функция завершила работу с ошибкой')
                else:
                    print('✅ Функция завершила работу успешно')
            print('------')
    print('ФИНАЛЬНЫЙ ОТВЕТ: ', answer)
