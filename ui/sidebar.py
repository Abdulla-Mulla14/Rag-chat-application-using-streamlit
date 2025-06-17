# ui/sidebar.py
import streamlit as st

def render_sidebar():
    st.sidebar.title("⚙️ Settings")

    # Temperature
    st.sidebar.slider(
        "Temperature", 0.0, 1.0, 0.1, step=0.01, key="temperature",
        help="0 = deterministic; 1 = creative."
    )

    # Max output tokens
    st.sidebar.slider(
        "Max Output Tokens", 1000, 5000, 3000, step=100, key="max_tokens",
        help="Max length of each response."
    )

    # Chunk size & overlap
    st.sidebar.slider(
        "Chunk Size", 500, 5000, 1000, step=100, key="chunk_size",
        help="Characters per chunk for retrieval."
    )
    st.sidebar.slider(
        "Chunk Overlap", 0, 500, 300, step=50, key="chunk_overlap",
        help="Overlap between chunks to preserve context."
    )

    # PDF uploader
    st.sidebar.file_uploader(
        "Upload PDF", type=["pdf"], key="pdf_file",
        help="Drag & drop or click to browse PDF files."
    )
