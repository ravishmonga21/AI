from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or export it in your shell.")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

class Calculator(BaseModel):
    result: float

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers"""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    return a / b

tools = [add, subtract, multiply, divide]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful calculator assistant.",
    response_format=Calculator,
)

response = agent.invoke({"messages": [{"role": "user", "content": "What is 10 + 5 and divide with 10 - 5?"}]})
print(response["structured_response"])