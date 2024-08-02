import pandas as pd


def save_new_chain(user_query, output_chain, answer):
    try:
        df = pd.read_csv('/Users/21109090/PycharmProjects/ai_table_assistant/cyqiq_data/chains.csv', index_col=0)
        df.loc[len(df)] = {'query': user_query, 'chain': output_chain, 'answer': answer}
        df.to_csv('/Users/21109090/PycharmProjects/ai_table_assistant/cyqiq_data/chains.csv')

    except FileNotFoundError:
        df = pd.DataFrame([[user_query, output_chain, answer]])
        df.columns = ['query', 'chain', 'answer']
        df.to_csv('/Users/21109090/PycharmProjects/ai_table_assistant/cyqiq_data/chains.csv')


def get_last_chains(how_many=5):
    try:
        df = pd.read_csv('/Users/21109090/PycharmProjects/ai_table_assistant/cyqiq_data/chains.csv', index_col=0)
        return df.tail(how_many)
    except:
        return ''


def get_action(actions: str):
    """Парсим цепочку действий для получения действия, которое надо выполнять в текущий момент"""
    if "<BEGIN>" in actions:
        action = actions.split('->')[1].strip()
    else:
        action = actions.split('->')[0].strip()
    return action
