from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langsmith import traceable
from dotenv import load_dotenv
import requests, sqlite3, os



# =========================== LLM and Keys ===========================
load_dotenv()
WEATHER_KEY = os.getenv("WEATHERSTACK_KEY")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



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

tools = [search_tool, get_stock_price, calculator, get_weather_data]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# =========================== State ===========================
class ChatState(TypedDict):
    # BaseMessage is the base class of all messages incluing HumanMessage, AIMessage, SystemMessage etc.
    messages: Annotated[list[BaseMessage], add_messages]


# =========================== Nodes ===========================
@traceable(name="chat_node_fn", tags=["chat"])
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call based on user's input."""
    
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
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
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge("chat_node", END)

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