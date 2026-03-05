import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command
from langgraph.graph import END

from llm_factory import get_llm

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)

# Setup
df = pd.read_csv("data/imdb.csv")
conn = sqlite3.connect("data/imdb.db")
df.to_sql("imdb", conn, if_exists="replace", index=False)
conn.close()

db = SQLDatabase.from_uri("sqlite:///data/imdb.db",include_tables=["imdb"])
llm = get_llm("gpt-4o-mini")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
system_prompt = Path(__file__).with_name("sql.yaml").read_text()

_sql_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

# ← Reusable helper for viz_agent to call directly
def run_sql_query(query: str) -> str:
    response = _sql_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return response["messages"][-1].content

# LangGraph node
def sql_agent(state: dict[str, Any]) -> Command:
    response = _sql_agent.invoke({"messages": state["messages"]})
    last_message = response["messages"][-1]
    logger.info("SQL agent response: %s", last_message.content)
    return Command(
        update={
            "messages": [AIMessage(content=last_message.content, name="sql_agent")],
            "next": "supervisor"
        },
        goto="supervisor"
    )