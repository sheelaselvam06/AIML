import os
from dotenv import load_dotenv
 
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
 
llm=AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)
 
response = llm.invoke("Hello, how are you?")
print("Response:", response.content)
 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
 
docs=[
    "Azure OpenAI provides enterprise-ready language models with Azure secure",
    "FAISS is a library for efficient similarity search and clustering of data",
    "RAG stands for Retrieval-Augmented Generation, combining search with generation",
    "LangChain helps buils LLM apps with chains, tools, and vector stores",
]
 
vector_db = FAISS.from_texts(docs, embeddings)
 
question = "what is RAG and how does FAISS help?"
top_k=2
retrieved = vector_db.similarity_search(question,k=top_k)
rag_prompt = (
    "Answer the question using onlt the context.\n\n"
    f"Context:\n{retrieved}\n\n"
    f"Question: {question}"
)
 
rag_response=llm.invoke(rag_prompt)
print("RAG answer:",rag_response.content)
 
 