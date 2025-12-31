# https://smith.langchain.com/              To run : streamlit run .\frontend.py
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend import chatbot, retrieve_all_threads, ingest_pdf, thread_document_metadata
import streamlit as st
import uuid



# =========================== Utilities ===========================
def generate_thread_id():
    """
    Generate a unique thread id
    """

    return uuid.uuid4()

def add_thread(thread_id):
    """
    Add thread id to the chat threads list in session state
    """

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    """
    Reset the chat for new chat session
    """

    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = [] # resetting the message_history

def load_conversation(thread_id):
    """
    Load the conversation/chat history for a given thread id
    """

    # To fetch the current state or final state if the workflow is finished with metadata for the thread ID.
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) # return messages if in state else empty list



# ======================= Session Initialization ===================
# Session State is a type of dictionary helps to preserve the data until the whole page is reload.
if "message_history" not in st.session_state: # Initializing message_history in session_state
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state: # Initializing thread_id in session_state
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state: # Loading chat_threads if don't exist in session_state
    st.session_state["chat_threads"] = retrieve_all_threads() # list of thread ids that already exist in the database

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"]) # Add the current thread id to the chat_threads list on page load

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# ============================ Sidebar ============================
st.sidebar.title("TheSoftMax")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Conversations History")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# for thread_id in threads:

#     # Display the thread id as a button in the sidebar. When the button is clicked, load the conversation history for that thread id.
#     if st.sidebar.button(str(thread_id)):
#         st.session_state["thread_id"] = thread_id
#         messages = load_conversation(thread_id)

#         # load the conversation history for that thread id
#         temp_messages = []
#         for msg in messages:
#             role = "user" if isinstance(msg, HumanMessage) else "assistant"
#             temp_messages.append({"role": role, "content": msg.content})
        
#         # updating the conversation history for that thread id on the UI
#         st.session_state["message_history"] = temp_messages



# ============================ Main UI ============================
for message in st.session_state["message_history"]:
    
    # For each role in message_history display the message in the chat window
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    
    # first update the message to message_history
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    # Display the user message in the chat window
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "run_name": "chat_turn",
        "configurable": {"thread_id": thread_key}, # To identify the user and his conversations.
        "metadata": {
            "thread_id": thread_key,
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "parser": "StrOutputParser"
        },
        "tags": ["llm app", "report_generation", "summarization"]
    }

    with st.chat_message("assistant"): # Assistant/Chatbot streaming response
        status_holder = {"box": None} # Hold the status of tool usage by Assistant

        def ai_only_stream():
            """
            Stream only the assistant messages from the chatbot response to the UI.
            """

            # Calling the chatbot with stream method to get the response in stream
            for message_chunk, _ in chatbot.stream( # calling the LLM using stream method
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # status_holder show tool is running if message_chunk is ToolMessage
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool") # getting tool_name from the ToolMessage
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)

                # Stream if message_chunk is AIMessage
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content # Yield the content of AIMessage instead of Return for streaming

        ai_message = st.write_stream(ai_only_stream()) # to stream the messages from the python to UI

        # update status_holder if tool was used
        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    # Update message history with assistant message
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()