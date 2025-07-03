from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# initialize state
class AgentState(TypedDict):
    messages: List[HumanMessage]
    
llm = ChatOpenAI(model='gpt-4o')

# to run the llm, we can use invoke method
def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    print(f"\AI: {response.content}")  # print the content in the response
    return state

graph = StateGraph(AgentState) # graph initialize
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# while loop
user_input = input("Enter : ")
while user_input!= "exit": 
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input=input("Enter :")
