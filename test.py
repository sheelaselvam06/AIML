from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class Concat(BaseModel):
    text1:str
    text2:str

@app.post("/concat")
def concat_text(data: Concat):
    return data.text1+"  "+data.text2