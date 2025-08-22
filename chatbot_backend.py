from langchain_core.messages import BaseMessage, HumanMessage
# from langgraph.checkpoint.memory import InMemorySaver # used for in memory storage
from langgraph.checkpoint.sqlite import SqliteSaver # pip install langgraph-checkpoint-sqlite
from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.postgres import PostgresSaver # used for postgresql storage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import sqlite3 

load_dotenv()
llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # BaseMessage is the base message capable to storing all message incluing HumanMessage, AIMessage, SystemMessage etc.
    # add_messages is a Reducer function to prevent from preventing erasing when adding the new message

def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']
    # send to llm
    response = llm.invoke(messages)
    # response store state
    return {"messages": [response]}

def retrieve_all_threads():
    all_threads = set() # store only unique thread ids
    for checkpoint in checkpointer.list(None):# to get checkpoints of all threads ids
        all_threads.add(checkpoint.config['configurable']['thread_id']) # saving the thread id to the set

    return list(all_threads) # returning the list of unique thread ids from the database

# Checkpointer
# If we will ask 10 + 2 then it will tell 12. If we will aks again 5 * previous result then it won't give the right answer. Because we are calling the llm with current HumanMessage only.
# To solve this problem we have to use the concept of Persistent, where we store the past conversation data to a databases (like sqlite and postgresql) if data is largescale or RAM/memory if data is small.
# checkpointer = InMemorySaver() # to store the previous conversation to the memory
# postgres_saver = PostgresSaver.from_conn_string(
#     "postgresql://user:password@localhost:5432/mydatabase" # for production or multi-instance setups. Requires connection details like host, port, database name, user, and password.
# )
# sqlite_saver = SqliteSaver.from_conn_string("sqlite:///checkpoints.db") # Good for small to medium projects or local development. 
# If you want custom SQLite options (like WAL mode, PRAGMA tuning, shared cache). If you already have an existing SQLite connection being used elsewhere in your app. If you need thread-safe or async connection handling. to visualize the database you can use SQLite studio.
                    # or
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) # check_same_thread is set to False to allow multiple threads to use the same connection
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)