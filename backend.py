from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Optional, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
import requests, sqlite3, os, tempfile, json
from langgraph.types import interrupt
from langchain_core.tools import tool
from langsmith import traceable
from dotenv import load_dotenv

# =========================== LLMs and Keys ===========================
load_dotenv()
WEATHER_KEY = os.getenv("WEATHERSTACK_KEY")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# =========================== PDF retriever store (per thread) ===========================
STORAGE_DIR = "storage"

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

def _get_thread_storage_path(thread_id: str):
    """Create and return the path for a specific thread's storage."""
    path = os.path.join(STORAGE_DIR, str(thread_id))
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def _load_thread_metadata(thread_id: str) -> dict:
    """Load metadata (list of files) from disk for a thread."""
    path = os.path.join(_get_thread_storage_path(thread_id), "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"files": {}}

def _save_thread_metadata(thread_id: str, metadata: dict):
    """Save metadata to disk."""
    path = os.path.join(_get_thread_storage_path(thread_id), "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def _get_retriever(thread_id: str):
    """
    Load the FAISS index from disk for the given thread.
    Returns None if no index exists.
    """
    if not thread_id:
        return None
        
    thread_path = _get_thread_storage_path(thread_id)
    index_path = os.path.join(thread_path, "index")
    
    # Check if index file exists (FAISS saves as index.faiss)
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        return None

    try:
        # allow_dangerous_deserialization is required for local pickle loading
        vector_store = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    """
    Process a PDF, add it to the thread's persistent FAISS index, and update metadata.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    
    thread_path = _get_thread_storage_path(thread_id)
    
    # 1. Save temp file to process
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # 2. Extract Text
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            return {
                "error": "No text found in PDF. It might be a scanned image or empty.",
                "filename": filename,
                "chunks": 0
            }

        # 3. Load existing index or create new one
        index_path = os.path.join(thread_path, "index")
        
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            vector_store = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)

        # 4. Save index back to disk
        vector_store.save_local(index_path)

        # 5. Update Metadata
        metadata = _load_thread_metadata(thread_id)
        metadata["files"][filename] = {
            "chunks": len(chunks),
            "pages": len(docs)
        }
        _save_thread_metadata(thread_id, metadata)

        return {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks)
        }

    except Exception as e:
        return {"error": str(e), "filename": filename, "chunks": 0}
        
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

def get_thread_file_names(thread_id: str) -> List[str]:
    """Helper to get list of filenames for system prompt."""
    if not thread_id:
        return []
    meta = _load_thread_metadata(thread_id)
    return list(meta.get("files", {}).keys())


# =========================== Tools ===========================
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    # For API key : https://www.alphavantage.co/support/#api-key
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_KEY}"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    # The string passed to interrupt() will be available in the frontend state
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}?")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'http://api.weatherstack.com/current?access_key={WEATHER_KEY}&query={city}'
    response = requests.get(url)
    return response.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """

    t_id = str(thread_id) if thread_id else None
    retriever = _get_retriever(t_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    # Extract clean text and unique sources
    context = []
    sources = set()
    
    for doc in result:
        context.append(doc.page_content)
        # Assuming PyPDFLoader puts source path in metadata, we strip to filename
        if "source" in doc.metadata:
            sources.add(os.path.basename(doc.metadata["source"]))

    return {
        "query": query,
        "context": context,
        "sources": list(sources)
    }

# Update tools list to include purchase_stock
tools = [search_tool, get_stock_price, purchase_stock, calculator, get_weather_data, rag_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# =========================== State ===========================
class ChatState(TypedDict):
    """
    The ChatState is state that is being transfered between each nodes in the Graph.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# =========================== Nodes ===========================
@traceable(tags=["chat"])
def chat_node(state: ChatState, config=None):
    """
    LLM node that may answer or request a tool call based on user's input.
    """

    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # Dynamic System Prompt: Injects list of available files
    uploaded_files = get_thread_file_names(str(thread_id)) if thread_id else []

    files_context = "No PDF documents uploaded yet."
    if uploaded_files:
        files_context = f"Available documents for RAG: {', '.join(uploaded_files)}"
    
    system_message = SystemMessage(
        content=(
            f"""
            You are a helpful assistant. {files_context} \nFor questions about the uploaded document(s), call the `rag_tool` and include the thread_id
            `{thread_id}`. You can also use the `search_tool`, `get_stock_price`, `purchase_stock`, `get_weather_data`
            and `calculator` tools when helpful. If no document is available, ask the user to upload a PDF.
            """
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]} # updating the response to the ChatState's messages


# =========================== Checkpointer ===========================
# for production or multi-instance setups. Requires connection details like host, port, database name, user, and password.
# postgres_saver = PostgresSaver.from_conn_string("postgresql://user:password@localhost:5432/mydatabase")

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False) # check_same_thread is set to False to allow multiple threads to use the same connection
checkpointer = SqliteSaver(conn=conn)


# =========================== Graph ===========================
# Create the ChatState instance
graph = StateGraph(ChatState)

# Nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition) # tools_condition decides whether to call a tool or END.
graph.add_edge('tools', 'chat_node')

# Compile Graph
chatbot = graph.compile(checkpointer=checkpointer) # pass the checkpointer object at the compilation to add it at each steps.


# =========================== Helper ===========================
def retrieve_all_threads():
    """
    return the list of all unique thread_ids from the database
    """
    all_threads = set()
    for checkpoint in checkpointer.list(None): # all checkpoints of all threads ids from the database
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

def get_thread_metadata(thread_id: str) -> dict:
    """Expose metadata loader to frontend."""
    return _load_thread_metadata(str(thread_id))