import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


os.environ['ANTHROPIC_API_KEY'] = os.environ.get('CLAUDEKEY')

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

prompt = ChatPromptTemplate([
    ("system", """"You are a senior banking analyst with memory of our full conversation.
You analyze financial data and remember everything discussed previously.
Always refer back to previous analyses when relevant."""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{input}")
])

chain = prompt |llm

# Memory store - Holds memory per session

store =  {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Wrap chain with memory

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# config ties messages to a session

config = {"configurable": {"session_id": "banking_session1"}}

print("Banking analyst with memory. Type quit to exit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chain_with_memory.invoke(
        {"input": user_input},
        config=config
    )
    print(f"\nAnalyst: {response}\n")