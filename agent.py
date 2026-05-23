import os
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


os.environ['ANTHROPIC_API_KEY'] = os.environ.get('CLAUDEKEY', '')

llm = ChatAnthropic(model='claude-sonnet-4-20250514', temperature=0)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


loader = PyPDFLoader("2026-q1-td-transcript.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(pages)

if os.path.exists("banking_db"):
    vector_store = Chroma(persist_directory="./banking_db", embedding_function=embeddings)
else:
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)


# tools

@tool
def search_documents(query: str) -> str:
    """Search the TD bank Q1 earnings transcripts for information.
    user this tool when asked about TD bank financials, earnings, executive comments, ROE targets,
    efficiency ratio and anything related to quaterly financial report"""

    result = vector_store.similarity_search_with_score(query, k=5)

    doc = [doc for doc, score in result if score < 1.8]

    context_parts = []
    for item in doc:
        pagenum = item.metadata.get("page", "unknown")
        context = item.page_content
        context_parts.append(f"Page: {pagenum}: {context}")

    return "\n".join(context_parts)


@tool
def calculate_ratio(numerator:float, denominator:float) -> str:
    """Calculate a financial ratio given a numerator and denominator
    use this when you need to compute percentages or ratios from financial figures"""

    if denominator == 0:
        return "Error cannoth divide by 0"
    
    ratio = (numerator/denominator) * 100
    return f"Ratio = {ratio:.2f}%"

@tool
def summarize_risk_factors(text: str) -> str:
    """Analyse and summarize risk factors from given financial text
    Use this tool when asked to identify or summarize risks in financial data"""

    risk_keywords = [
        "risk", "concern", "decline", "increase", "pressure",
        "uncertainty", "provision", "loss", "exposure", "volatile"
    ]

    sentences = text.split(".")

    risk_sentences = [
        s.strip() for s in sentences
        if any(key in s.lower() for key in risk_keywords)
    ]

    if not risk_sentences:
        return "No specific risk factors analyzed in the given sentences"
    
    return "Risk factors identified:\n {[f'- s' for s in risk_sentences]}"


# Create agent

tools = [search_documents, calculate_ratio, summarize_risk_factors]

memory = MemorySaver()

agent = create_agent(
    llm,
    tools,
    checkpointer=memory
)

config = {"configurable": {"thread_id": "tdanalysis_1"}}


while True:
    input_msg = input("You: ")
    if input_msg == 'quit':
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": input_msg}]},
        config
    )


    final_response = response["messages"][-1].content
    print(f"Agent: {final_response}")

    for msg in response["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"Tool unsed {tool_call['name']}")
                print(f"Tool input {tool_call['args']}")
        elif msg.type == tool:
            print(f"Tool result preview {msg.content[:100]}...")
