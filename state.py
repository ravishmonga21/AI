from typing import Any
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    next: str                            # controls routing between nodes
    user_query: str                      # original raw query from the user
    chart_config: dict[str, Any] | None  # ECharts options object for the UI
    final_answer: str                    # clean text response to return to the user