import streamlit as st
from  langgraph_tool_backend import chatbot, thread_id_database
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


# **************************************************** Utility function ********************************************
#generate thread_id
def generate_thread_id():
    return str(uuid.uuid4())

def reset_id():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])# adding thread_id to chat_thread
    st.session_state["message_history"] = []
    
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get("messages", [])


# ***************************************************** Session setup **********************************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
    
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = thread_id_database()
    
add_thread(st.session_state["thread_id"]) #add thread id to chat_ threads
    
# ***************************************************** Sidebar UI *************************************************
st.sidebar.title("🤖 kṛtrim 🤖")

if st.sidebar.button('New Chat'):
    reset_id()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(str(thread_id))

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
            

# ***************************************************** Main UI ***************************************************

for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.text(message['content'])
        
user_input = st.chat_input("Type Here")

if user_input:
    
    #add messages to message_histroy
    st.session_state["message_history"].append({'role':'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    # CONFIG = {'configurable': {'thread_id':st.session_state["thread_id"]}}
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG, # type: ignore
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True) # type: ignore
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )