import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

llm=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b"))


def add(a: int,b: int):
    """Add two numbers"""
    return a+b

def multiply(a: int,b: int):
    """Multiply two numbers"""
    return a*b

tools=[add,multiply]
react_agent_auto = create_react_agent(
    model=llm,
    tools=tools
)

result_auto=react_agent_auto.invoke({
    "messages":[HumanMessage(content="What is 12+30? use tool.")]
})

print(result_auto)
