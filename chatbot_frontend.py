# run : streamlit run .\chatbot_frontend.py
# visit for langsmith : https://smith.langchain.com/
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chatbot_backend import chatbot, retrieve_all_threads
import streamlit as st
import uuid

# =========================== Utilities ===========================
# function to generate a unique thread id
def generate_thread_id():
    return uuid.uuid4()

# function to reset the chat for new chat session
def reset_chat():
    # generating a new thread id
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    # saving the thread id in session state
    add_thread(thread_id)
    # resetting the message history
    st.session_state["message_history"] = []

# function to add a thread id to the chat threads list
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# function to load the conversation/chat history for a given thread id
def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

# ======================= Session Initialization ===================
# Session State is a type of dictionary helps to preserve the data until the whole page is reload.
# Initialize message history in session state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Initialize thread_id in session state
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Initialize chat threads that already exist in session state
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads() # list of thread ids that already exist in the database

# Add the current thread id to the chat threads list on page load
add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("TheSoftMax Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversations History")

for thread_id in st.session_state["chat_threads"][::-1]:
    # Display the thread id as a button in the sidebar
    # When the button is clicked, load the conversation history for that thread id
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        # changing the format of messages received from get_state method to be compatible with the chat UI
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        # passing the new formatted messages to the session state
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# loading the conversation history
for message in st.session_state["message_history"]:
    # Text box
    with st.chat_message(message["role"]):
        # text in text box
        st.text(message["content"])

# Take the input from the user
user_input = st.chat_input("Type here")

if user_input:
    # first add the message to message_history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # To identify the user, whose conversation is stored in the memory.
    # CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    CONFIG = {
        "run_name": "chat_turn",
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"],
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "parser": "StrOutputParser"
        },
        "tags": ["llm app", "report_generation", "summarization"]
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        # write_stream of streamlit is used to stream the messages from the python to UI.
        # other is status container to show the status of response.
        # calling the LLM using stream method
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

# question to ask the llm/agent.
# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.