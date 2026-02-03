import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from PIL import Image
import io
llm=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b"))
def manager_node(state):
    task_input=state.get("task","")
    input=state.get("input","")
    prompt=f"""
    You are a task router. Based on the user request below, decide whether it is a:
    -translate
    -summarize
    -calculate
    
    Respond with only one word (translate, summarize, or calculate).

    Task: {task_input}
    """

    decision=llm.invoke(prompt).content.strip().lower()
    return {"agent":decision,"input":input}

def translator_node(state):
    text=state.get("input","")
    prompt=f"Act like You are a translator. Only respond with the English translation of the text below.\n\n{text}"
    result=llm.invoke(prompt).content
    return {"result":result}

def summarizer_node(state):
    text=state.get("input","")
    prompt=f"Summarize the following in 1-2 lines:\n\n{text}"
    result=llm.invoke(prompt).content
    return {"result":result}

def calculator_node(state):
    expression=state.get("input","")
    prompt=f"Please calculate and return the result of:\n{expression}"
    result=llm.invoke(prompt).content
    return {"result":result}

def route_by_agent(state):
    return{
        "translate":"translator",
        "summarize":"summarizer",
        "calculate":"calculator",
    }.get(state.get("agent",""),"default")

def default_node(state):
    return {"result":"Sorry, I couldn't understand the task."}

from langgraph.graph import StateGraph, END

g=StateGraph(dict)

g.add_node("manager",manager_node)
g.add_node("translator",translator_node)
g.add_node("summarizer",summarizer_node)
g.add_node("calculator",calculator_node)
g.add_node("default",default_node)

g.set_entry_point("manager")
g.add_conditional_edges("manager",route_by_agent)

g.add_edge("translator", END)
g.add_edge("summarizer", END)
g.add_edge("calculator", END)
g.add_edge("default", END)



app=g.compile()

if __name__ == "__main__":
    print(app.invoke({
        "task" :"Can you translate this ?",
        "input":"Bonjour le monde"
    }))

    print(app.invoke({
        "task":"Please summarize the following",
        "input":"Langgraph helps you build flexible multi-agent workflows in Python..."
    }))

    respcal = app.invoke({
        "task": "What is 12*8+5",
        "input": "12*8+5"
    })

    print(respcal['result'])

    print(app.invoke({
        "task": "Can you dance?",
        "input": "foo"
    }))

    image_data = app.get_graph().draw_mermaid_png()
    img = Image.open(io.BytesIO(image_data))
    img.show()
        