import streamlit as st
from langchain_helper import get_few_shot_db_chain


@st.cache_resource
def load_chain():
    return get_few_shot_db_chain()


st.title("AtliQ T Shirts: Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    try:
        chain = load_chain()
        response = chain.invoke({"query": question})

        st.header("Answer")
        st.write(response["result"])
    except Exception as e:
        st.error(f"Something went wrong: {e}")