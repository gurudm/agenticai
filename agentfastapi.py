from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Banking Analyser")

os.environ["ANTHROPIC_API_KEY"] = os.environ.get("CLAUDEKEY")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)



def vector_store() -> Chroma:

    if os.path.exists("./quaterly.db"):
        return Chroma(persist_directory="./quaterly.db", embedding_function=embeddings)
    
    loader = PyPDFLoader("./2026-q1-td-transcript.pdf")

    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_documents(pages)

    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./quaterly.db")

vectordb = vector_store()

@tool
def search_document(query:str):
    """This tool is used to search for the any financial questions related to quterly performance
    conversation between stakeholders, management, transcripts"""

    result = vectordb.similarity_search_with_score(query=query, k=5)

    selected = [doc for doc, score in result if score < 1.85]

    context_portion = []
    for item in selected:
        page_number = item.metadata.get("page", "unknown")
        content = item.page_content
        context_portion.append(f"Page number: {page_number} content: {content}")

    return "\n".join(context_portion)

@tool
def percent_calculation(numerator: float, denominator: float):
    """This tool is to calculate ratio or percentage when numerator and denominator is provided
    use this tool to calculate ratio or percentage for financialy performace queries"""

    percent_value = (numerator/denominator) * 100

    return percent_value

@tool
def summarize_report(text: str):
    """This tool provides a summary of earnings, risks, performance and forecast
    Use this tool to summarize quaterly performance, earnings, risks and forecast"""

    risk_keywords = [
        "risk", "concern", "decline", "increase", "pressure",
        "uncertainty", "provision", "loss", "exposure", "volatile"
    ]

    sentences = text.split(".")

    risk_sentences = [
        sentence for sentence in sentences
        if any([risk in sentence for risk in risk_keywords])
    ]

    if not risk_sentences:
        return "No risks identified"
    
    return f"Risk factors identified\n\n {["- s\n" for s in risk_sentences]}"

tools = [search_document, percent_calculation, summarize_report]

memory = MemorySaver()

agent = create_agent(
    llm,
    tools,
    checkpointer=memory
)




class RequestModel(BaseModel):

    query: str
    session_id: str

class ResponseModel(BaseModel):

    response: str
    session_id: str
    tool_calls: list


@app.get("/")
def root():
    return {"status": "This is an intelligent banking analyst interface"}

@app.post("/ask")
def ask_analyst(question: RequestModel, response_model=ResponseModel):

    session_id = question.session_id
    query = question.query
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config
    )

    tool_call = []
    for message in result["messages"]:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_cal in message.tool_calls:
                tool_call.append(f"Tool used: {tool_cal["name"]}\n\t{tool_cal["args"]}")

    return {
        "response" : result["messages"][-1].content,
        "session_id": session_id,
        "tool_calls": tool_call
    }


@app.get("/health")
def health_check():
    return {
        "status": "Green",
        "document_count": vectordb._collection.count(),
        "model": "claude-sonnet-4-20250514"
    }