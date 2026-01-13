from langgraph.graph  import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-safeguard-20b")

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    
#node function
def chat_bot(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
#memorycheckpoint
checkpointer = SqliteSaver(conn=conn)

#graph
graph = StateGraph(ChatState)

graph.add_node('chat_bot', chat_bot)

graph.add_edge(START, 'chat_bot')
graph.add_edge('chat_bot', END)

chatbot = graph.compile(checkpointer=checkpointer)


def thread_id_database():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) # type: ignore
    return list(all_threads)