import os 
import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or export it in your shell.")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to .env or export it in your shell.")


def get_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    if model == "gpt-4o-mini":
        return ChatOpenAI(model=model, api_key=OPENAI_API_KEY)
    elif model == "claude-4-6-sonnet":
        return ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY)
    else:
        raise ValueError(f"Model {model} not supported")

