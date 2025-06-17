import streamlit as st

st.set_page_config(page_title="InkQuery", layout="wide")

# Load custom CSS


# 4) Now load other libraries and your modules
from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path

from ui.sidebar import render_sidebar
from modules.config import get_config
from modules.indexing import index_pdf
from modules.chat import query_pdf_chat, get_vector_store
from modules.quiz import generate_quiz

# 3) Sidebar + config
render_sidebar()
cfg = get_config()

st.title("InkQuery – AI-Powered PDF Conversations")
st.info("Set your preferences before uploading your PDF to begin chatting.")

# 4) PDF ingestion
uploaded = st.session_state.get("pdf_file")
if uploaded and st.button("Prepare this PDF"):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    pdf_path = data_dir / "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded.read())

    with st.spinner("Indexing PDF…"):
        index_pdf(pdf_path, cfg)
    st.session_state.ready = True
    st.success("✅ Ready to chat!")

# 5) Chat interface (fixed ordering)
if st.session_state.get("ready"):

    st.subheader("💬 Chat with your PDF")

    # initialize history
    if "history" not in st.session_state:
        st.session_state.history = []

    # 5a) Accept new user input and immediately process it
    user_input = st.chat_input("Ask a question…")
    if user_input:
        # append user question
        st.session_state.history.append(("user", user_input))
        # get answer
        answer = query_pdf_chat(user_input, cfg)
        # append assistant reply
        st.session_state.history.append(("assistant", answer))
        # no manual rerun needed

    # 5b) Now render the entire history (including what we just appended)
    for role, msg in st.session_state.history:
        st.chat_message(role).markdown(msg)

    # 6) Quiz generator button
    if st.button("Generate Quiz"):
        # Retrieve top chunks for context
        vs = get_vector_store()
        chunks = [r.page_content for r in vs.similarity_search(query="", k=5)]
        # Pass in cfg["model_name"], plus any other settings
        quiz_text = generate_quiz(
            "\n\n".join(chunks),
            num_q=5,
            model_name=cfg["model_name"],         # llama-3.1-8b-instant
            temperature=cfg["temperature"],       # reuse sidebar temp if you like
            max_tokens=400                        # shorter, to keep quiz concise
        )
        st.markdown("### 📝 Quiz Questions")
        st.markdown(quiz_text)