from typing import Annotated, Sequence, TypedDict  
# annotate - use to attach metadata to a type
# sequence - This means the messages field is a list like sequence of BaseMessage objects.
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # reducer function...?
# how to mmatch new data into the current state
# without a reducer fumction, updates would have replaced the existing value entirely
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
# all are explained in the video
   
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
 
@tool  # a decorator
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""
    # without function description of docstring, code give an error
    # tell the llm what that do
    return a + b 

# tools
@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]  # list of tools

model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)
# bind_tools - give the model llm to use tools

# node initialize
def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"]) # system + query
    return {"messages": [response]} # update message with response (no + =)

# conditional for the loop
def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]  # last item of the message list
    if not last_message.tool_calls:   # if no more tools to call, then end
        return "end"
    else: 
        return "continue"
  
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)  # tool node
graph.add_node("tools", tool_node)  # give tools to tool node

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))