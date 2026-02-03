from aiohttp import request
import streamlit as st 
import requests

API_URL="http://localhost:8000/process"

st.title("Streamlit to FastAPI")

with st.form("text_form"):
    text=st.text_input("Enter text")
    submitted=st.form_submit_button("Send to FastAPI")

if submitted:
    response=requests.post(
        API_URL,
        json={"text": text}
    )

    if response.status_code==200:
        data=response.json()
        st.success("Response from FastAPI")
        st.json(data)
    else:
        st.error("API call failed")