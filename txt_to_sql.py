try:
    import streamlit as st
    import duckdb
    import pandas as pd
    from faker import Faker
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.prompts import PromptTemplate
    from dotenv import load_dotenv
    import os
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install with: pip install duckdb faker langchain-huggingface")
    exit(1)
 
# Load environment variables from .env file
load_dotenv()
 
def init_database():
    """
    Generates fake data, loads it into DuckDB, and exports schema.txt.
    Returns the DuckDB connection and the schema string.
    """
    fake = Faker()
 
    data = []
    for _ in range(50):
        data.append({
            "name": fake.name(),
            "job_title": fake.job(),
            "company": fake.company(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "city": fake.city(),
            "salary": fake.random_int(min=40000, max=150000),
            "hired_date": str(fake.date_between(start_date='-5y', end_date='today'))
        })
 
    df = pd.DataFrame(data)
 
    con = duckdb.connect(database=':memory:')
    con.execute("CREATE TABLE employees AS SELECT * FROM df")
 
    schema_info = con.execute("DESCRIBE employees").fetchall()
 
    schema_str = "Table: employees\nColumns:\n"
    for col in schema_info:
        schema_str += f"- {col[0]} ({col[1]})\n"
 
    with open("schema.txt", "w") as f:
        f.write(schema_str)
 
    return con, schema_str
 
 
# Streamlit UI setup
st.set_page_config(page_title="DuckDB English to SQL", layout="wide")
st.title(" Text-to-SQL Generator")
#st.markdown("Generates SQL for **DuckDB** using **Faker** data and **Hugging Face**.")
 
# Get Hugging Face API token from environment variable
hf_api_token = os.getenv("HF_TOKEN")
if not hf_api_token:
    st.error("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your environment.")
    st.stop()
 
# Initialize DuckDB connection
if 'db_connection' not in st.session_state:
    with st.spinner("Generating fake data and setting up DuckDB..."):
        con, schema_text = init_database()
        st.session_state['db_connection'] = con
        st.session_state['schema_text'] = schema_text
else:
    con = st.session_state['db_connection']
    schema_text = st.session_state['schema_text']
 
# Hugging Face model setup
repo_id = "openai/gpt-oss-120b"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=hf_api_token,
    temperature=0.1,
    max_new_tokens=512
)
chat_model = ChatHuggingFace(llm=llm)
 
# Prompt template
template = """
You are an expert in DuckDB SQL.
Your goal is to write a valid SQL query to answer the question based on the schema provided.
 
Table Schema:
{schema}
 
Question: {question}
 
Rules:
1. Return ONLY the SQL query.
2. No markdown (```sql). No explanations.
3. Use DuckDB syntax (Standard SQL usually works).
 
SQL Query:
"""
 
prompt = PromptTemplate.from_template(template)
 
# Layout
col1, col2 = st.columns([2, 1])
 
with col1:
    user_query = st.text_input(
        "Ask a question about the 'employees' table:",
        placeholder="e.g. Who has the highest salary in the company?"
    )
 
    if st.button("Generate & Run"):
        if user_query:
            with st.spinner("Thinking..."):
                try:
                    formatted_prompt = prompt.format(
                        schema=schema_text,
                        question=user_query
                    )
 
                    response = chat_model.invoke(formatted_prompt)
 
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
 
                    sql_query = response_text.strip().replace("```sql", "").replace("```", "").strip()
 
                    st.success("Generated SQL:")
                    st.code(sql_query, language="sql")
 
                    st.subheader("Query Results:")
                    try:
                        results = con.execute(sql_query).df()
                        st.dataframe(results)
                    except Exception as e:
                        st.error(f"SQL Execution Error: {e}")
 
                except Exception as e:
                    st.error(f"LLM Error: {e}")
 