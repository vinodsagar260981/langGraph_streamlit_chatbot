from langgraph.graph  import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-safeguard-20b")

# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
        "calculator": {
            "transport": "stdio",
            "command": r"C:\Users\vinod\OneDrive\Desktop\Coding\AIWorld\Langgraph_streamlit_chatbot\.venv\Scripts\python.exe",          
            "args": [r"C:\Users\vinod\OneDrive\Desktop\Coding\AIWorld\Langgraph_streamlit_chatbot\main.py"],
        }
    }
)


#state 
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]


async def build_graph():
    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)
    
    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    tool_node = ToolNode(tools)
    
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)

    graph.add_edge("tools", "chat_node")    
    chatbot = graph.compile()
    return chatbot

async def main():
    chatbot = await build_graph()
    
    #running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content = "Find the modulus of 132456 and 23 and give the answer like cricket commentator")]}) # type: ignore
    print(result['messages'][-1].content)


if __name__ == "__main__":
    asyncio.run(main())