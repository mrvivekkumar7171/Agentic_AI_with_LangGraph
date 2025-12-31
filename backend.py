from __future__ import annotations
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
import requests, sqlite3, os, tempfile
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
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """
    Fetch the retriever for a thread if available.
    """
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            return {
                "error": "No text found in PDF. It might be a scanned image.",
                "filename": filename or os.path.basename(temp_path),
                "documents": len(docs),
                "chunks": 0,
            }

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass



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
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator, get_weather_data, rag_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# =========================== State ===========================
class ChatState(TypedDict):
    """
    The ChatState is state that is being transfered between each nodes in the Graph where
    BaseMessage is the base class of all messages incluing HumanMessage, AIMessage, SystemMessage etc.
    add_messages is reducer function to add new BaseMessage, without removing the past one.
    
    :var messages: Description
    :vartype messages: Any
    :var response: Description
    :vartype response: AIMessage
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

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
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

def thread_has_document(thread_id: str) -> bool:
    """
    Docstring for thread_has_document
    
    :param thread_id: Description
    :type thread_id: str
    :return: Description
    :rtype: bool
    """

    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """
    Docstring for thread_document_metadata
    
    :param thread_id: Description
    :type thread_id: str
    :return: Description
    :rtype: dict
    """

    return _THREAD_METADATA.get(str(thread_id), {})