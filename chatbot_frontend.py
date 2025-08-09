from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot, retrieve_all_threads
import streamlit as st
import uuid


# **************************************** utility functions *************************

# function to generate a unique thread id
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

# function to reset the chat for new chat session
def reset_chat():

    # generating a new thread id
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id

    # saving the thread id in session state
    add_thread(st.session_state['thread_id'])

    # resetting the message history
    st.session_state['message_history'] = []

# function to add a thread id to the chat threads list
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

# function to load the conversation/chat history for a given thread id
def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']


# **************************************** Session Setup ******************************

# Session State is a type of dictionary helps to preserve the data until the whole page is reload.
# Initialize message history in session state
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Initialize thread_id in session state
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# Initialize chat threads that already exist in session state
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads() # list of thread ids that already exist in the database

# Add the current thread id to the chat threads list on page load
add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('TheSoftMax Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('Conversations History')

for thread_id in st.session_state['chat_threads'][::-1]:
    # Display the thread id as a button in the sidebar
    # When the button is clicked, load the conversation history for that thread id
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        # changing the format of messages received from get_state method to be compatible with the chat UI
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        # passing the new formatted messages to the session state
        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    # Text box
    with st.chat_message(message['role']):
        # text in text box
        st.text(message['content'])

# Take the input from the user
user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # To identify the user, whose conversation is stored in the memory.
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # first add the message to message_history
    with st.chat_message('assistant'):

        # write_stream of streamlit is used to stream the messages from the python to UI.
        # other is status container to show the status of response.
        ai_message = st.write_stream(
            # calling the LLM using stream method
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= CONFIG,
                stream_mode= 'messages'
            )
        )

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})