from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
import os
from fastapi import FastAPI
from pydantic import BaseModel


os.environ['ANTHROPIC_API_KEY'] = os.environ.get('CLAUDEKEY')


app = FastAPI(title="Aibanking")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

prompt = ChatPromptTemplate([
    ("system", "You are a top ayurveda pandith (doctor) who is very popular in providing traditional ayurvedic treatement"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

chain = prompt | llm

store = {}


def get_session_history(session_id) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chatwithmemory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


# request and response model

class RequestModel(BaseModel):
    session_id: str
    message: str

class ResponseModel(BaseModel):
    session_id: str
    response: str


@app.get('/')
def homepage():
    return {'status': "Banking application ai bot"}

@app.post('/request', response_model=ResponseModel)
def chatbot(request: RequestModel):
    config = {"configurable": {"session_id": request.session_id}}

    response = chatwithmemory.invoke(
        {"input": request.message},
        config=config
    )

    return ResponseModel(
        session_id= request.session_id,
        response=response.content
    )

@app.delete("/delete/{session_id}")
def remove_session(session_id: str):
    if session_id in store:
        del store[session_id]
        return f"{session_id} deleted"
    return f"{session_id} not found"

