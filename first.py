from huggingface_hub._webhooks_payload import BaseModel
from fastapi import FastAPI

appn=FastAPI()

@appn.get('/test')
def read_root():
    return "Testing Done"

@appn.post('/status')
def status():
    return {"status": "running"}

@appn.get("/items/{item_id}")
def read_item(item_id):
    return {"item_id": item_id}


class Item(BaseModel):
    name: str
    email: str

@appn.post("/create-items/")
def create_item(item: Item):
    return item.name+" "+item.email

    

