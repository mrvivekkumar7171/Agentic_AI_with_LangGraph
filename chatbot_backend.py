from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver # pip install langgraph-checkpoint-sqlite
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
# from langgraph.checkpoint.memory import InMemorySaver # used for in memory storage
from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.postgres import PostgresSaver # used for postgresql storage
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import sqlite3
import os

load_dotenv()
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI()

# -------------------
# 2. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en") # pre-build tools.

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
    # DockString is must for tools as LLM model read this to use it.
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    # to get the api key visit : https://www.alphavantage.co/support/#api-key
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_KEY}"
    r = requests.get(url)
    return r.json()



tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # BaseMessage is the base message capable to storing all message incluing HumanMessage, AIMessage, SystemMessage etc.
    # add_messages is a Reducer function to prevent from preventing erasing when adding the new message
# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"] # take user query from state
    response = llm_with_tools.invoke(messages) # send to llm
    return {"messages": [response]} # response store state

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
# If we will ask 10 + 2 then it will tell 12. If we will aks again 5 * previous result then it won't give the right answer. Because we are calling the llm with current HumanMessage only.
# To solve this problem we have to use the concept of Persistent, where we store the past conversation data to a databases (like sqlite and postgresql) if data is largescale or RAM/memory if data is small.
# checkpointer = InMemorySaver() # to store the previous conversation to the memory
# postgres_saver = PostgresSaver.from_conn_string(
#     "postgresql://user:password@localhost:5432/mydatabase" # for production or multi-instance setups. Requires connection details like host, port, database name, user, and password.
# )
# sqlite_saver = SqliteSaver.from_conn_string("sqlite:///checkpoints.db") # Good for small to medium projects or local development. 
# If you want custom SQLite options (like WAL mode, PRAGMA tuning, shared cache). If you already have an existing SQLite connection being used elsewhere in your app. If you need thread-safe or async connection handling. to visualize the database you can use SQLite studio.
                    # or
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False) # check_same_thread is set to False to allow multiple threads to use the same connection
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set() # store only unique thread ids
    for checkpoint in checkpointer.list(None):# to get checkpoints of all threads ids
        all_threads.add(checkpoint.config['configurable']['thread_id']) # saving the thread id to the set

    return list(all_threads) # returning the list of unique thread ids from the database