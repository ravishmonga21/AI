from langgraph.graph import StateGraph, START, END
from state import AgentState
from supervisor.supervisor_agent import supervisor_agent
from sql.sql_agent import sql_agent
from visualisation.visualisation_agent import viz_agent

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("sql_agent", sql_agent)
    graph.add_node("viz_agent", viz_agent)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", "__end__"),
        {
            "sql_agent": "sql_agent",
            "viz_agent": "viz_agent",
            "__end__":   END,
        }
    )

    graph.add_conditional_edges(
        "sql_agent",
        lambda state: state.get("next", "supervisor"),
        {
            "supervisor": "supervisor",
            "__end__":    END,
        }
    )

    graph.add_edge("viz_agent", END)

    return graph.compile()

app = build_graph()