import streamlit as st
from newlanggraph import app
from langchain_core.runnables.graph import MermaidDrawMethod
import io
from PIL import Image

st.set_page_config(page_title="AI Task Router", page_icon="🤖")

st.title("LangGraph")



with st.sidebar:
    st.header("Graph Visualization")
    if st.button("Show Graph"):
        try:
             image_data = app.get_graph().draw_mermaid_png()
             img = Image.open(io.BytesIO(image_data))
             img.show()
        except Exception as e:
            st.error(f"Could not render graph: {e}")

col1, col2 = st.columns(2)

with col1:
    task_input = st.text_input("What is your task?", placeholder="e.g., Can you translate this?")

with col2:
    content_input = st.text_area("Content/Input", placeholder="Enter the text or expression here...")

if st.button("Submit", type="primary"):
    if task_input and content_input:
        with st.spinner("Processing..."):
            try:
                result = app.invoke({
                    "task": task_input,
                    "input": content_input
                })
                
                st.success("Task Completed!")
                
                # Display result
                st.subheader("Result")
                if "result" in result:
                    st.write(result["result"])
                else:
                    st.json(result)
                
                # Display trace info
                with st.expander("See Execution Details"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both a task and input content.")

