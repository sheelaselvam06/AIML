import streamlit as st
import re
 
st.title("Text Utility App")
 
text = st.text_area("Enter text", height=200)
 
col1, col2, col3 = st.columns(3)
with col1:
    do_lower = st.checkbox("Lowercase")
with col2:
    remove_extra_spaces = st.checkbox("Remove extra spaces")
with col3:
    remove_punct = st.checkbox("Remove punctuation")
 
if st.button("Process"):
    out = text
 
    if do_lower:
        out = out.lower()
    if remove_extra_spaces:
        out = re.sub(r"\s+", " ", out).strip()
    if remove_punct:
        out = re.sub(r"[^\w\s]", "", out)
 
    st.subheader("Processed Text")
    st.write(out)
 
    st.subheader("Stats")
    st.write({
        "characters": len(out),
        "words": len(out.split()) if out else 0,
        "lines": out.count("\n") + 1 if out else 0
    })
    