import operator
from typing import TypedDict, Annotated, Sequence

import pandas as pd
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Определение состояний графа"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[str], operator.add]
    inter: pd.DataFrame
    question: str
    memory: str
