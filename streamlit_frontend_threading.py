import streamlit as st
from langraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid


# **************************************************** Utility function ********************************************
def generate_thread_id():
    return str(uuid.uuid4())

def reset_id():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
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
    st.session_state["chat_threads"] = []
    
add_thread(st.session_state["thread_id"])
    
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
    
    CONFIG = {'configurable': {'thread_id':st.session_state["thread_id"]}}

    # add the message to message_histroy
    with st.chat_message("assistant"):
        ai_message =st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream( # type: ignore
                        {'messages': [HumanMessage(content=user_input)]},
                        config = CONFIG, # type: ignore
                        stream_mode='messages')
        )
        
        st.session_state["message_history"].append({'role':'assistant', 'content': ai_message})