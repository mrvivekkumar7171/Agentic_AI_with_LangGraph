# https://smith.langchain.com/              To run : streamlit run .\frontend.py
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend import chatbot, retrieve_all_threads, ingest_pdf, get_thread_metadata
from langgraph.types import Command
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

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

add_thread(st.session_state["thread_id"]) # Add the current thread id to the chat_threads list on page load

thread_key = str(st.session_state["thread_id"])
# Fetch persistent metadata for the current thread
current_metadata = get_thread_metadata(thread_key)
files_info = current_metadata.get("files", {})

threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# Configuration for the current thread
CONFIG = {
    "run_name": "chat_turn",
    "configurable": {"thread_id": thread_key}, 
    "metadata": {
        "thread_id": thread_key,
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "parser": "StrOutputParser"
    },
    "tags": ["llm app", "report_generation", "summarization"]
}


# ============================ Sidebar ============================
st.sidebar.title("TheSoftMax")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# --- Files List Section ---
st.sidebar.subheader("Uploaded Documents")

if files_info:
    for fname, stats in files_info.items():
        st.sidebar.text(f"📄 {fname}")
        st.sidebar.caption(f"{stats['chunks']} chunks | {stats['pages']} pages")
else:
    st.sidebar.info("No documents uploaded yet.")

# --- Upload Section ---
uploaded_pdf = st.sidebar.file_uploader(" ", type=["pdf"],key=f"uploader_{st.session_state['file_uploader_key']}")

if uploaded_pdf:
    # Check against persistent metadata
    if uploaded_pdf.name in files_info:
        st.sidebar.info(f"`{uploaded_pdf.name}` is already indexed.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )

            if summary.get("error") or summary.get("chunks", 0) == 0:
                status_box.update(label="❌ Failed", state="error")
                st.sidebar.error(summary.get("error", "Processing failed."))
            else:
                status_box.update(label="✅ Indexed", state="complete")
                st.session_state["file_uploader_key"] += 1
                st.rerun() # Rerun to update the file list immediately

st.sidebar.divider()
st.sidebar.subheader("History")

if not threads:
    st.sidebar.write("No past conversations.")
else:
    for thread_id in threads:
        # Highlight current thread
        label = f"➤ {thread_id}" if str(thread_id) == thread_key else str(thread_id)
        if st.sidebar.button(label, key=f"side-thread-{thread_id}"):
            selected_thread = thread_id


# ============================ Main UI ============================

# 1. Display Chat History
for message in st.session_state["message_history"]:

    # For each role in message_history display the message in the chat window
    with st.chat_message(message["role"]):
        st.text(message["content"])


# 2. Check for HITL Interrupts (Paused State)
snapshot = chatbot.get_state(CONFIG)
pending_interrupt_value = None

if snapshot.next:
    # Check tasks for interrupts. In LangGraph, tasks contain the interrupt info.
    for task in snapshot.tasks:
        if task.interrupts:
            # The value passed to `interrupt(...)` in backend.py
            pending_interrupt_value = task.interrupts[0].value
            break


# 3. Helper Function to Stream Response
def stream_graph_response(input_payload):
    with st.chat_message("assistant"): # Assistant/Chatbot streaming response
        status_holder = {"box": None} # Hold the status of tool usage by Assistant

        def ai_only_stream():
            """
            Stream only the assistant messages from the chatbot response to the UI.
            """

            # Calling the chatbot with stream method to get the response in stream
            for message_chunk, _ in chatbot.stream( # calling the LLM using stream method
                input_payload,
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

    return ai_message


# 4. Handle Input (Either User Text or Approval Buttons)
if pending_interrupt_value:
    # --- HITL MODE ---
    st.info(f"⚠️ Action Required: **{pending_interrupt_value}**")
    
    col1, col2 = st.columns([1, 1])
    decision = None
    
    with col1:
        if st.button("✅ Approve"): decision = "yes"
    with col2:
        if st.button("❌ Deny"): decision = "no"

    if decision:
        # Resume the graph with the decision using Command
        st.session_state["message_history"].append({"role": "user", "content": f"[Decision: {decision}]"})
        
        response_content = stream_graph_response(Command(resume=decision))
        
        st.session_state["message_history"].append({"role": "assistant", "content": response_content})
        st.rerun()

else:
    # --- STANDARD CHAT MODE ---
    user_input = st.chat_input("Type here ...")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        response = stream_graph_response({"messages": [HumanMessage(content=user_input)]})
        
        st.session_state["message_history"].append({"role": "assistant", "content": response})
        
        # Check if we hit an interrupt immediately after streaming
        snapshot = chatbot.get_state(CONFIG)
        if snapshot.next and any(task.interrupts for task in snapshot.tasks):
            st.rerun() # Rerun to show the approval buttons


# 5. Thread Switching Logic
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    # Load messages
    messages = load_conversation(selected_thread)

    # Format messages for UI
    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages

    # Trigger reload to fetch new metadata for this thread
    st.rerun()