import sys
from dotenv import load_dotenv
from langchain.tools import tool
from pydantic import BaseModel
from typing import Any
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=10,length_function=len)
chunks = text_splitter.split_text("""TechFlow Solutions is a software company founded in 2018 in San Francisco. 
        The company specializes in building AI-powered workflow automation tools for enterprises. 
        TechFlow has over 500 employees and serves more than 2000 business customers globally. 
        The company's flagship product is FlowEngine, an intelligent process automation platform. 
        TechFlow was founded by Sarah Chen and Michael Rodriguez, both former Google engineers.""",)
# print(len(chunks))

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder
import uuid
import os

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or export it in your shell.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env or export it in your shell.")

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chunk_vectors = embeddings_model.embed_documents(chunks)

chunk_records = [
    {
        "id": str(uuid.uuid4()),
        "values": vector,
        "metadata": {
            "chunk_text": chunk,
            "source": "techflow_demo",
            "chunk_index": idx,
            "tags": [
                "techflow", "software", "company", "workflow",
                "automation", "ai", "process", "platform", "google", "engineers"
            ],
        },
    }
    for idx, (chunk, vector) in enumerate(zip(chunks, chunk_vectors))
]
pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index("test-index"):
    pc.create_index(
        name="test-index",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index("test-index")
index.upsert(chunk_records)
# print(index.describe_index_stats())

#Retrieval
def dense_retrieval(query:str, top_k:int=5) -> list[dict]:
    query_vector = embeddings_model.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results

# print(dense_retrieval("What is TechFlow Solutions?"))

#Reranking
def rerank(query: str, results: Any, top_k: int = 5) -> list[dict]:
    matches = results.matches if hasattr(results, "matches") else results.get("matches", [])
    if not matches:
        return []

    pairs = [(query, match.metadata.get("chunk_text", "")) for match in matches]
    rerank_scores = cross_encoder.predict(pairs)

    reranked_results = []
    for match, rerank_score in zip(matches, rerank_scores):
        reranked_results.append(
            {
                "id": match.id,
                "rerank_score": float(rerank_score),
                "retrieval_score": float(match.score),
                "metadata": match.metadata,
            }
        )

    reranked_results.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked_results[:top_k]

# print(rerank("What is TechFlow Solutions?", dense_retrieval("What is TechFlow Solutions?")))

#LLM Response
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

system_message = SystemMessage(content="You are a helpful assistant that can answer questions about the given text.")
human_message = HumanMessage(content="What is TechFlow Solutions?")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

def llm_response(query:str, results:list[dict]) -> str:
    human_message = HumanMessage(content=query)
    return llm.invoke([system_message, human_message] + [AIMessage(content=result["metadata"]["chunk_text"]) for result in results])

print(llm_response("What is TechFlow Solutions?", rerank("What is TechFlow Solutions?", dense_retrieval("What is TechFlow Solutions?"))))
