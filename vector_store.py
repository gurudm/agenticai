from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI
from pydantic import BaseModel
import os
import pypdf

app = FastAPI(title="BankingAI")


os.environ["ANTHROPIC_API_KEY"] = os.environ.get("CLAUDEKEY")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)



# Read documnet
loader = PyPDFLoader("/Users/gurudm/development/agenticai/2026-q1-td-transcript.pdf")
pages = loader.load()

# Split into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=['\n\n', '\n', '.', ' ']
)

chunks = text_splitter.split_documents(pages)

print(f"Split pdf into {len(chunks)} chunks")

print(f"first chunk is {chunks[0].page_content}")
print(f"first chunk metadata is {chunks[0].metadata}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")




if not os.path.exists("./quaterly_db"):
    vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./quaterly_db")
else:
    vector_store = Chroma(persist_directory="./quaterly_db", embedding_function=embeddings)

print(f"Stored chunks in vector store {vector_store._collection.count()}")

def get_documents(query):
    docs = vector_store.similarity_search_with_score(query, k=5)
    doc = [doc for doc, score in docs if score < 1.8]

    if not doc:
        return None, []
    
    context_parts = []
    for ele in doc:
        context_parts.append(f"Page {ele.metadata.get('page', 'unknown')}\n{ele.page_content}")

    return "\n\n".join(context_parts)

prompt = ChatPromptTemplate([
    ("system", """You are a senior banking analyst.
Answer questions using ONLY the context provided below.
Always mention which page the information came from if available.
Respond in this JSON format:
{{
  "response": "your detailed answer",
  "source_pages": [1, 2],
  "confidence": "high/medium/low"
}}
If the context does not contain enough information say:
"I don't have sufficient information in the documents to answer that."
Never make up information.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


chain = prompt | llm

store = {}

def get_session_history(session_id) -> InMemoryChatMessageHistory:
    if not session_id in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)


class AppRequestModel(BaseModel):
    question: str
    session_id: str

class AppResponseModel(BaseModel):
    session_id: str
    response: str

@app.get("/")
def home():
    return {"status": "A Banking Analyst chatbot"}


@app.post("/ask", response_model=AppResponseModel)
def ask(request: AppRequestModel):

    session_id = request.session_id
    question = request.question

    config = {"configurable": {"session_id": session_id}}
    context = get_documents(question)
    print(f"\n=== RETRIEVED CONTEXT ===\n{context}\n=========================\n")


    chat_response = chain_with_history.invoke(
        {
            "question": question,
            "context": context
        },
        config
    )

    return AppResponseModel(
        session_id=request.session_id,
        response=chat_response.content
    )