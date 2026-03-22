from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun  
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages


load_dotenv()



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


class researchState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]



# tools  
search_tool = DuckDuckGoSearchRun(region = 'us-en')

tools = [search_tool]

llm_with_tools = llm.bind_tools(tools=tools)



# nodes
def chat_node(state: researchState) -> researchState:
     
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': response}

tool_node = ToolNode(tools)


# graph
graph = StateGraph(researchState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)


graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile()
output = chatbot.invoke({'messages': [HumanMessage(content= "what is today's date ?")]})
print(output)