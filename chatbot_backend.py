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
import requests, sqlite3, os


# -------------------
# 1. LLM and Keys
# -------------------
load_dotenv()
WEATHER_KEY = os.getenv("WEATHERSTACK_KEY")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
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
import sqlite3
import re
import heapq
from typing import Dict, List, Tuple, Any, Optional

# ------------------------
# 1) Robust retrieval of thread ids from sqlite checkpointer
# ------------------------
def retrieve_all_threads_sqlite(conn: sqlite3.Connection, thread_col_name: str = "thread_id") -> List[str]:
    """
    Query the sqlite checkpointer DB directly to get unique thread ids.
    This assumes the default schema where checkpoints store a JSON config
    that contains a 'configurable.thread_id' or similar.
    - conn: sqlite3 connection
    - thread_col_name: if your DB stores thread id in a dedicated column, use that name.
    Returns list of unique thread ids (strings).
    """
    cur = conn.cursor()
    threads = set()

    # Attempt 1: If there's a column that directly stores thread_id
    try:
        # you may need to adjust table name; common name is 'checkpoints' or 'checkpoint'
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        # find candidate table names
        candidates = [t for t in tables if "checkpoint" in t.lower() or "check" in t.lower()]
        for tbl in candidates:
            # try to see if thread_id exists as column
            cur.execute(f"PRAGMA table_info({tbl})")
            cols = [r[1] for r in cur.fetchall()]
            if thread_col_name in cols:
                cur.execute(f"SELECT DISTINCT {thread_col_name} FROM {tbl} WHERE {thread_col_name} IS NOT NULL")
                for row in cur.fetchall():
                    threads.add(str(row[0]))
                if threads:
                    return list(threads)
    except Exception:
        # ignore and fallback to JSON parsing below
        pass

    # Attempt 2: Parse JSON blobs in 'config' or 'data' columns to extract thread_id
    # This handles the typical shape SqliteSaver uses (storing JSON in a column)
    json_cols = []
    for tbl in candidates:
        cur.execute(f"PRAGMA table_info({tbl})")
        for r in cur.fetchall():
            colname = r[1]
            # heuristic: config, data, payload, checkpoint, metadata
            if colname.lower() in ("config", "data", "payload", "checkpoint", "metadata"):
                json_cols.append((tbl, colname))

    # For each candidate table + column, attempt to parse JSON rows
    for tbl, col in json_cols:
        try:
            cur.execute(f"SELECT {col} FROM {tbl} WHERE {col} IS NOT NULL")
            rows = cur.fetchall()
            for (blob,) in rows:
                if not blob:
                    continue
                # blob may be text or bytes; try parse
                try:
                    import json
                    obj = json.loads(blob) if isinstance(blob, (str, bytes)) else blob
                    # try common nesting: obj['configurable']['thread_id'] or obj['config']['thread_id']
                    thread_id = None
                    if isinstance(obj, dict):
                        # walk possible keys
                        candidates_keys = [
                            ("configurable", "thread_id"),
                            ("config", "thread_id"),
                            ("meta", "thread_id"),
                            ("thread_id",),
                        ]
                        for keys in candidates_keys:
                            cur_obj = obj
                            found = True
                            for k in keys:
                                if isinstance(cur_obj, dict) and k in cur_obj:
                                    cur_obj = cur_obj[k]
                                else:
                                    found = False
                                    break
                            if found:
                                thread_id = cur_obj
                                break
                    if thread_id:
                        threads.add(str(thread_id))
                except Exception:
                    # sometimes blob isn't JSON, fallback to regex search
                    s = str(blob)
                    m = re.search(r'"thread_id"\s*:\s*"([^"]+)"', s)
                    if m:
                        threads.add(m.group(1))
            if threads:
                return list(threads)
        except Exception:
            continue

    # Final fallback: use checkpointer.list if available (slower)
    try:
        for ck in checkpointer.list(None):
            try:
                tid = ck.config.get("configurable", {}).get("thread_id")
                if tid:
                    threads.add(str(tid))
            except Exception:
                continue
    except Exception:
        pass

    return list(threads)


# ------------------------
# 2) search threads by keyword (sqlite)
# ------------------------
def search_threads_by_keyword(conn: sqlite3.Connection, keyword: str, snippet_len: int = 80) -> List[Tuple[str, str]]:
    """
    Scan JSON blobs in known checkpoint tables for the keyword.
    Returns list of tuples (thread_id, snippet).
    """
    cur = conn.cursor()
    results = {}
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    # heuristics for columns that may store text
    candidate_cols = []
    for tbl in tables:
        cur.execute(f"PRAGMA table_info({tbl})")
        for r in cur.fetchall():
            colname = r[1]
            if colname.lower() in ("config", "data", "payload", "checkpoint", "metadata", "content"):
                candidate_cols.append((tbl, colname))

    import json
    for tbl, col in candidate_cols:
        try:
            cur.execute(f"SELECT rowid, {col} FROM {tbl} WHERE {col} IS NOT NULL")
            for rowid, blob in cur.fetchall():
                if not blob:
                    continue
                text = ""
                if isinstance(blob, (bytes, bytearray)):
                    try:
                        text = blob.decode("utf-8", errors="ignore")
                    except:
                        text = str(blob)
                elif isinstance(blob, str):
                    text = blob
                else:
                    try:
                        text = json.dumps(blob)
                    except:
                        text = str(blob)

                idx = text.lower().find(keyword.lower())
                if idx != -1:
                    # try to extract thread id if present
                    m = re.search(r'"thread_id"\s*:\s*"([^"]+)"', text)
                    thread_id = m.group(1) if m else f"row_{tbl}_{rowid}"
                    snippet = text[max(0, idx - 20): idx + snippet_len]
                    results[thread_id] = snippet
        except Exception:
            continue

    return list(results.items())


# ------------------------
# 3) Generic Dijkstra (adjacency dict)
# ------------------------
def dijkstra_shortest_path(adj: Dict[Any, List[Tuple[Any, float]]], start: Any, goal: Any) -> Optional[List[Any]]:
    """
    adj: adjacency dict of form {node: [(neighbor, weight), ...], ...}
    returns shortest path as list [start, ..., goal] or None if unreachable
    """
    if start == goal:
        return [start]
    dist = {}
    prev = {}
    pq = []
    heapq.heappush(pq, (0, start))
    dist[start] = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        if u == goal:
            # reconstruct path
            path = []
            cur = goal
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            path.append(start)
            return list(reversed(path))
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None


# ------------------------
# 4) Helper to convert StateGraph to adjacency dict
# ------------------------
def graph_to_adj(state_graph) -> Dict[Any, List[Tuple[Any, float]]]:
    """
    Try to extract adjacency information from a StateGraph instance.
    This is heuristic — different versions may store edges differently.
    """
    adj = {}
    # try attributes commonly used
    if hasattr(state_graph, "edges"):
        # edges might be list of (u, v) or dict
        try:
            for e in state_graph.edges:
                u, v = e[0], e[1]
                adj.setdefault(u, []).append((v, 1.0))
        except Exception:
            pass

    # Some implementations store adjacency in _graph or _adj
    if hasattr(state_graph, "_graph"):
        try:
            for u, nbrs in state_graph._graph.items():
                for v in nbrs:
                    # nbr may be (dest, weight) or just dest
                    if isinstance(v, tuple):
                        adj.setdefault(u, []).append((v[0], float(v[1])))
                    else:
                        adj.setdefault(u, []).append((v, 1.0))
        except Exception:
            pass

    if hasattr(state_graph, "_adj") and isinstance(state_graph._adj, dict):
        try:
            for u, nbrs in state_graph._adj.items():
                for v in nbrs:
                    if isinstance(v, tuple):
                        adj.setdefault(u, []).append((v[0], float(v[1])))
                    else:
                        adj.setdefault(u, []).append((v, 1.0))
        except Exception:
            pass

    # fallback: if graph exposes nodes and a get_neighbors method
    if hasattr(state_graph, "nodes") and hasattr(state_graph, "get_neighbors"):
        for u in getattr(state_graph, "nodes", []):
            try:
                nb = state_graph.get_neighbors(u)
                for v in nb:
                    adj.setdefault(u, []).append((v, 1.0))
            except Exception:
                pass

    return adj

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
