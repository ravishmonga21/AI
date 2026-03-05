import json
import logging
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import END

from llm_factory import get_llm

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)

llm = get_llm("gpt-4o-mini")
system_prompt = Path(__file__).with_name("visualisation.yaml").read_text()

def viz_agent(state: dict[str, Any]) -> Command:
    messages = state.get("messages", [])
    user_query = state.get("user_query", "")

    # SQL data is always in the last sql_agent message
    sql_data = next(
        (msg.content for msg in reversed(messages)
         if getattr(msg, "name", "") == "sql_agent"),
        None
    )

    if not sql_data:
        logger.error("Viz agent: no SQL data found in state.")
        return Command(
            update={
                "chart_config": {"error": "No SQL data available to visualize."},
                "next": "__end__"
            },
            goto=END
        )

    logger.info("Viz agent: using SQL data: %s", sql_data)

    # Generate ECharts config using real SQL data
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"User question: {user_query}\n\n"
            f"Data from database (use EXACT values, do not make up data):\n{sql_data}\n\n"
            f"Return ONLY valid JSON. No comments, no markdown, no extra text."
        ))
    ])
    logger.info("Visualization agent raw response: %s", response.content)

    # Strip comments and code fences then parse
    try:
        clean = re.sub(r'//.*', '', response.content)
        clean = re.sub(r'```json|```', '', clean).strip()
        chart_config = json.loads(clean)
    except json.JSONDecodeError:
        logger.error("Viz agent returned invalid JSON: %s", response.content)
        chart_config = {
            "chart_type": None,
            "options": None,
            "error": "Visualization agent returned malformed JSON.",
            "raw": response.content
        }

    return Command(
        update={
            "messages": [AIMessage(content=json.dumps(chart_config), name="viz_agent")],
            "chart_config": chart_config,
            "next": "__end__"
        },
        goto=END
    )
