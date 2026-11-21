import csv
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors, types

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "faq.csv")
ESCALATED_FILE = os.path.join(BASE_DIR, "escalated.csv")
MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(page_title="Customer Support FAQ Bot", page_icon="💬")
st.title("💬 Customer Support FAQ Bot")
st.caption("Answers from faq.csv and automatically logs unknown questions to escalated.csv.")


@st.cache_data
def load_faq_dataframe(filepath: str) -> pd.DataFrame:
    """Load FAQ rows from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Make sure faq.csv is in the project folder.")
    df = pd.read_csv(filepath, dtype=str)
    required_cols = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError("faq.csv must contain 'question' and 'answer' columns.")
    return df.dropna(subset=["question", "answer"])


@st.cache_data
def build_faq_prompt_block(df: pd.DataFrame) -> str:
    """Format FAQ entries compactly for fast prompt processing."""
    return "\n".join(
        f"Q: {str(row['question']).strip()} | A: {str(row['answer']).strip()}"
        for _, row in df.iterrows()
    )


def save_escalated_question(question: str, filepath: str = ESCALATED_FILE) -> None:
    """Append an unanswered question with timestamp to escalated.csv."""
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["timestamp", "question"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question.strip()])


def stream_faq_bot(client: genai.Client, user_question: str, faq_block: str, max_retries: int = 2):
    """Stream chunks from Gemini with short retry on temporary 503 load."""
    system_instruction = (
        "Answer ONLY using the FAQ below. Be brief and polite. "
        "If not in FAQ, reply strictly with: ESCALATE\n\n"
        f"{faq_block}"
    )

    for attempt in range(max_retries):
        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=user_question,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
        except errors.ServerError as exc:
            if exc.code == 503 and attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise


# --- Sidebar: API Key & Escalation Log ---
st.sidebar.header("Setup & Escalations")
env_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
ui_api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=env_api_key,
    type="password",
    help="Set GEMINI_API_KEY in your environment or paste your key here.",
)
api_key = ui_api_key.strip()

if not api_key:
    st.warning("Enter your Gemini API key in the sidebar (or set GEMINI_API_KEY) to start chatting.")
    st.stop()

client = genai.Client(api_key=api_key)

# Load FAQ CSV
try:
    faq_df = load_faq_dataframe(FAQ_FILE)
    faq_prompt_block = build_faq_prompt_block(faq_df)
except Exception as exc:
    st.error(f"Error loading FAQ file: {exc}")
    st.stop()

with st.expander("View FAQ Knowledge Base (faq.csv)"):
    st.dataframe(faq_df, use_container_width=True)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything from our FAQ, or ask a question if you need help."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your customer support question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""
        try:
            for piece in stream_faq_bot(client, user_input, faq_prompt_block):
                full_reply += piece
                # Avoid flashing raw ESCALATE token before checking
                if not full_reply.strip().upper().startswith("ESCALATE"):
                    placeholder.markdown(full_reply + "▌")
        except errors.ServerError as exc:
            st.error(f"Gemini service is temporarily busy ({exc.code}). Please try again in a moment.")
            st.stop()

        cleaned_reply = full_reply.strip()
        normalized_reply = cleaned_reply.upper().strip(" .!\"'")

        if normalized_reply == "ESCALATE" or normalized_reply.startswith("ESCALATE"):
            save_escalated_question(user_input, ESCALATED_FILE)
            assistant_reply = (
                "This question isn't covered in our FAQ, so I have saved it to **escalated.csv** "
                "for our support team to follow up."
            )
            placeholder.markdown(assistant_reply)
            st.toast("Question saved to escalated.csv", icon="📌")
        else:
            assistant_reply = cleaned_reply
            placeholder.markdown(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# Show escalated questions in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Escalated Questions (escalated.csv)")
if os.path.exists(ESCALATED_FILE):
    escalated_df = pd.read_csv(ESCALATED_FILE)
    st.sidebar.dataframe(escalated_df, use_container_width=True)
else:
    st.sidebar.info("No escalated questions saved yet.")