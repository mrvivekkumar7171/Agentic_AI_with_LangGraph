import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}
# To identify the user, whose conversation stored in the memory.

# Session State is a type of dictionary helps to preserve the data until the whole page is reload.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history
for message in st.session_state['message_history']:
    # Text box
    with st.chat_message(message['role']):
        # text in text box
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

# Take the input from the user
user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # calling the LLM
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    
    ai_message = response['messages'][-1].content
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)

# to get the chat history
# Note the history will be stored till the program is running, 
# once the program end or restated the program the data is lost.
chatbot.get_state(config=CONFIG)