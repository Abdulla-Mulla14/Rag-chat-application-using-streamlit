# modules/config.py
import streamlit as st

def get_config():
    return {
        "model_name":    "llama-3.1-8b-instant",  # Fastest available on Groq
        "temperature":   st.session_state.get("temperature", 0.1),
        "max_tokens":    st.session_state.get("max_tokens", 3000),
        "chunk_size":    st.session_state.get("chunk_size", 1000),
        "chunk_overlap": st.session_state.get("chunk_overlap", 300),
    }
