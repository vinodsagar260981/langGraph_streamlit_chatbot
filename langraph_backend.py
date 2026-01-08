from langgraph.graph  import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-safeguard-20b")

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    
#node function
def chat_bot(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

#memorycheckpoint
checkpointer = InMemorySaver()

#graph
graph = StateGraph(ChatState)

graph.add_node('chat_bot', chat_bot)

graph.add_edge(START, 'chat_bot')
graph.add_edge('chat_bot', END)

chatbot = graph.compile(checkpointer=checkpointer)

# response = chatbot.invoke({'messages': [HumanMessage(content="Hi name is vinod")]}, config = {"configurable": {"thread_id": "thread_1"}})

# CONFIG = {"configurable": {"thread_id": "thread_1"}}

# print(chatbot.get_state(config=CONFIG).values['messages']) # type: ignore