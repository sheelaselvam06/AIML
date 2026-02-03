from fastapi import FastAPI
from pydantic import BaseModel
app=FastAPI()
class TextInput(BaseModel):
    text:str


@app.post("/process/")
def process_text(data: TextInput):
    return{
        "original":data.text,
        "length":len(data.text),
        "upper":data.text.upper()
    }