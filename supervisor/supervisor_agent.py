import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END

from llm_factory import get_llm

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)

llm = get_llm("gpt-4o-mini")
system_prompt = Path(__file__).with_name("supervisor.yaml").read_text()

def supervisor_agent(state: dict[str, Any]) -> Command:
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    last_name = getattr(last_message, "name", "") if last_message else ""

    # If last message is from viz_agent — always end
    if last_name == "viz_agent":
        logger.info("Supervisor: viz_agent results received, ending.")
        return Command(
            update={"final_answer": last_message.content, "next": "__end__"},
            goto=END
        )

    # Call LLM to decide next step
    response = llm.invoke([SystemMessage(content=system_prompt), *messages])
    content = response.content.lower()

    if "viz_agent" in content:
        next_node = "viz_agent"
    elif "sql_agent" in content:
        next_node = "sql_agent"
    else:
        next_node = "__end__"

    logger.info("Supervisor routing to: %s", next_node)

    return Command(
        update={
            "messages": [AIMessage(content=response.content, name="supervisor")],
            "final_answer": last_message.content if next_node == "__end__" else "",
            "next": next_node
        },
        goto=next_node if next_node != "__end__" else END
    )