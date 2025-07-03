from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm = ChatOpenAI(model='gpt-4o')

def process(state:AgentState) -> AgentState:
    """this node will solve the request you input"""
    response = llm.invoke(state['messages'])
    
    # response.content - extract only the content part of the response
    state['messages'].append(AIMessage(content=response.content))
    print(f"\AI: {response.content}")
    
    return state

# create graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()    
    
# create history
conversation_history = []

user_input = input("Enter : ")
while user_input!= "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter : ")
    
# add messages into a txt file
with open("logging.txt", 'w') as file:
    file.write("your Conversation Log :\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You :  {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI :  {message.content}\n")
    file.write("\nEnd of Conversation")

print("Conversation saved to logging.txt")