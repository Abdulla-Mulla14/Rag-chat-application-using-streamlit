# modules/quiz.py

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# 1) Load environment variables
load_dotenv()

def generate_quiz(
    chunks_text: str,
    num_q: int = 5,
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    """
    Generate a multiple-choice quiz based on the provided text chunks.
    - chunks_text: concatenated chunks from the PDF
    - num_q: number of questions
    - model_name: which Groq model to use (must be valid)
    """
    # 2) Build the prompt
    prompt = (
        f"Create {num_q} multiple-choice quiz questions (with correct answers) "
        f"based on the text below:\n\n{chunks_text}"
    )

    # 3) Instantiate the same Groq API client used elsewhere
    chat = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
    )

    # 4) Call the LLM
    resp = chat.invoke([
        SystemMessage(content="You are a helpful quiz generator."),
        HumanMessage(content=prompt),
    ])

    # 5) Return just the text
    return resp.content
