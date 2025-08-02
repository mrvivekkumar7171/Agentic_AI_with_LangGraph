from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

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

# Checkpointer
# If we will ask 10 + 2 then it will tell 12.
# If we will aks again 5 * previous result then it won't give the right answer.
# Because we are calling the llm with current HumanMessage only.
# To solve this problem we have to use the concept of Persistent, where we store the past conversation 
# data to a database(largescale) or RAM(small scale) i.e. memory.
checkpointer = InMemorySaver() # to store the previous conversation to the memory

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)