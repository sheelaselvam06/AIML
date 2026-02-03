import langgraph 
from typing import TypedDict
from langgraph.graph import StateGraph, END
from PIL import Image
import io

class Mystate(TypedDict):
    count: int

def increment(st: Mystate) -> Mystate:
    return {"count": st["count"] + 1} 

def double(st: Mystate) -> Mystate:
    return {"count": st["count"] * 2} 

# print(f"Increment test: {increment(ms)}")
# print(f"Double test: {double(ms)}")
graph = StateGraph(Mystate)

graph.add_node("increment", increment)
graph.add_node("double", double)
graph.set_entry_point("increment")
graph.add_edge("increment", "double")
graph.add_edge("double", END)

app = graph.compile()
result=app.invoke({"count": 3})
print(result)

image_data = app.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(image_data))
img.show()
