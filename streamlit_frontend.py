import streamlit as st
from langraph_backend import chatbot
from langchain_core.messages import HumanMessage

st.title("JEEVAI CHATBOT")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

CONFIG = {'configurable': {'thread_id':"thread_1"}}

for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.text(message['content'])
        
user_input = st.chat_input("Type Here")

if user_input:
    
    #add messages to histor
    st.session_state["message_history"].append({'role':'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG) # type: ignore
    ai_message = response['messages'][-1].content
    
    st.session_state["message_history"].append({'role':'assistant', 'content': ai_message})    
    with st.chat_message("assistant"):
        st.markdown(ai_message)