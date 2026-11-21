import csv
import os
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "faq.csv")
ESCALATED_FILE = os.path.join(BASE_DIR, "escalated.csv")
MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(
    page_title="Customer Support FAQ Bot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Reference-Inspired UI Styling ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
    }
    .top-banner {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 52%, #0f766e 100%);
        color: #ffffff;
        padding: 1.15rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.25rem;
    }
    .top-banner h1 {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
    }
    .top-banner p {
        margin: 0.25rem 0 0 0;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.65rem;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.15rem;
    }
    .metric-num {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.15;
    }
    .card-faq {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .card-esc {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .card-user {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .tag-faq {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.4rem;
    }
    .tag-esc {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.4rem;
    }
    .meta-muted {
        font-size: 0.78rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Data & Fast Retrieval Helpers ---
@st.cache_data
def load_faq_dataframe(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Make sure faq.csv is in the project folder.")
    df = pd.read_csv(filepath, dtype=str)
    required_cols = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError("faq.csv must contain 'question' and 'answer' columns.")
    cleaned = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
    cleaned["question"] = cleaned["question"].str.strip()
    cleaned["answer"] = cleaned["answer"].str.strip()
    return cleaned


@st.cache_data
def build_full_faq_block(df: pd.DataFrame) -> str:
    return "\n".join(f"Q: {r['question']} | A: {r['answer']}" for _, r in df.iterrows())


def select_relevant_faq_block(df: pd.DataFrame, user_question: str, top_k: int = 8) -> str:
    tokens = {w for w in re.findall(r"[a-z0-9]+", user_question.lower()) if len(w) > 2}
    if not tokens:
        return build_full_faq_block(df)

    scored = []
    for idx, row in df.iterrows():
        haystack = f"{row['question']} {row['answer']}".lower()
        score = sum(1 for tok in tokens if tok in haystack)
        scored.append((score, idx, row))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    top_rows = [row for score, _, row in scored[:top_k] if score > 0]

    if not top_rows:
        return build_full_faq_block(df)

    return "\n".join(f"Q: {r['question']} | A: {r['answer']}" for r in top_rows)


def load_escalated_dataframe(filepath: str = ESCALATED_FILE) -> pd.DataFrame:
    if os.path.exists(filepath):
        return pd.read_csv(filepath, dtype=str)
    return pd.DataFrame(columns=["timestamp", "question"])


def save_escalated_question(question: str, filepath: str = ESCALATED_FILE) -> None:
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["timestamp", "question"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question.strip()])


def stream_faq_bot(client: genai.Client, user_question: str, faq_block: str, max_retries: int = 2):
    system_instruction = (
        "Answer ONLY from the FAQ context below in 1-2 short sentences. "
        "If the answer is not clearly in the FAQ context, reply strictly with: ESCALATE\n\n"
        f"{faq_block}"
    )

    for attempt in range(max_retries):
        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=user_question,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
        except errors.ServerError as exc:
            if exc.code == 503 and attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            raise


# --- Session State ---
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []

if "draft_query" not in st.session_state:
    st.session_state.draft_query = "What is your return and refund policy?\nHow do I track my order?"

if "run_now_query" not in st.session_state:
    st.session_state.run_now_query = None

# --- Top Banner ---
st.markdown(
    """
    <div class="top-banner">
        <h1>💬 Customer Support FAQ Assistant</h1>
        <p>Transform customer inquiries into instant FAQ answers or logged support escalations</p>
    </div>
    """,
    unsafe_allow_html=True,
)

env_key = os.environ.get("GEMINI_API_KEY", "").strip()
with st.expander("🔑 API Key & Settings", expanded=not bool(env_key)):
    api_key = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        placeholder="Paste your Gemini API key or set GEMINI_API_KEY",
    ).strip()
    st.caption(f"Model: `{MODEL_NAME}`")

if not api_key:
    st.warning("Enter your Gemini API key above (or set `GEMINI_API_KEY`) to run queries.")
    st.stop()

try:
    faq_df = load_faq_dataframe(FAQ_FILE)
    _ = build_full_faq_block(faq_df)
except Exception as exc:
    st.error(f"Error loading FAQ file: {exc}")
    st.stop()

client = genai.Client(api_key=api_key)
escalated_df = load_escalated_dataframe(ESCALATED_FILE)

# --- Two-Column Main Workspace ---
left_col, right_col = st.columns([1.05, 1.35], gap="large")

with left_col:
    st.markdown('<div class="section-title">📝 1. Ask or Pick Sample Queries</div>', unsafe_allow_html=True)

    preset_cols = st.columns(3)
    if preset_cols[0].button("📦 Orders & Shipping", use_container_width=True):
        st.session_state.draft_query = "How do I track my order?"
        st.rerun()
    if preset_cols[1].button("💳 Refunds & Billing", use_container_width=True):
        st.session_state.draft_query = "What is your return and refund policy?"
        st.rerun()
    if preset_cols[2].button("🚨 Out-of-Scope Test", use_container_width=True):
        st.session_state.draft_query = "Can I book a custom private demo flight?"
        st.rerun()

    query_box = st.text_area(
        "Customer query (one question per line)",
        key="draft_query",
        height=145,
    )

    act_cols = st.columns([1.3, 1])
    ask_clicked = act_cols[0].button("⚡ Get Support Reply", type="primary", use_container_width=True)
    if act_cols[1].button("🧹 Clear Query", use_container_width=True):
        st.session_state.draft_query = ""
        st.rerun()

    queries_to_process = []
    if ask_clicked and query_box.strip():
        queries_to_process = [line.strip() for line in query_box.splitlines() if line.strip()]
    elif st.session_state.run_now_query:
        queries_to_process = [st.session_state.run_now_query]
        st.session_state.run_now_query = None

    if queries_to_process:
        for q_item in queries_to_process:
            relevant_block = select_relevant_faq_block(faq_df, q_item, top_k=8)
            raw_chunks = []
            try:
                for piece in stream_faq_bot(client, q_item, relevant_block):
                    raw_chunks.append(piece)
            except errors.ServerError as exc:
                st.error(f"Gemini service is temporarily busy ({exc.code}). Please try again.")
                st.stop()

            model_out = "".join(raw_chunks).strip()
            norm_out = model_out.upper().strip(" .!\"'")

            if norm_out == "ESCALATE" or norm_out.startswith("ESCALATE"):
                save_escalated_question(q_item, ESCALATED_FILE)
                status = "escalated"
                reply_text = (
                    "Not found in our FAQ knowledge base. Logged to **escalated.csv** "
                    "for human support follow-up."
                )
            else:
                status = "faq"
                reply_text = model_out

            st.session_state.qa_log.insert(
                0,
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "question": q_item,
                    "answer": reply_text,
                    "status": status,
                },
            )

        escalated_df = load_escalated_dataframe(ESCALATED_FILE)
        st.success(f"Processed {len(queries_to_process)} query(ies) instantaneously.")
        st.rerun()

    with st.expander(f"📚 Browse Knowledge Base ({len(faq_df)} FAQs)"):
        kb_search = st.text_input("Filter FAQs", placeholder="Type keyword...")
        kb_view = faq_df
        if kb_search.strip():
            mask = (
                faq_df["question"].str.contains(kb_search, case=False, na=False)
                | faq_df["answer"].str.contains(kb_search, case=False, na=False)
            )
            kb_view = faq_df[mask]
        st.dataframe(kb_view, use_container_width=True, hide_index=True)

with right_col:
    st.markdown('<div class="section-title">📋 2. Support Responses & Escalations</div>', unsafe_allow_html=True)

    total_asked = len(st.session_state.qa_log)
    faq_answered_count = sum(1 for item in st.session_state.qa_log if item["status"] == "faq")
    esc_count = sum(1 for item in st.session_state.qa_log if item["status"] == "escalated")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-label">📚 Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-num">{len(faq_df)}</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-label">✅ FAQ Answered</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-num">{faq_answered_count}</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-label">🚨 Escalated (Session)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-num">{esc_count}/{max(total_asked, 1) if total_asked else 0}</div>', unsafe_allow_html=True)

    ratio = (faq_answered_count / total_asked) if total_asked else 0.0
    st.progress(ratio)

    f1, f2, f3 = st.columns([1.2, 1, 1.3])
    status_filter = f1.selectbox("Filter by status", ["All", "Answered from FAQ", "Escalated"])
    keyword_filter = f3.text_input("Search responses", placeholder="Keyword...")
    if f2.button("Reset Log", use_container_width=True):
        st.session_state.qa_log = []
        st.rerun()

    filtered_items = st.session_state.qa_log
    if status_filter == "Answered from FAQ":
        filtered_items = [x for x in filtered_items if x["status"] == "faq"]
    elif status_filter == "Escalated":
        filtered_items = [x for x in filtered_items if x["status"] == "escalated"]

    if keyword_filter.strip():
        kw = keyword_filter.strip().lower()
        filtered_items = [
            x for x in filtered_items if kw in x["question"].lower() or kw in x["answer"].lower()
        ]

    if not filtered_items:
        st.info("No responses yet. Click **Get Support Reply** on the left to test a question.")
    else:
        for item in filtered_items:
            is_faq = item["status"] == "faq"
            card_cls = "card-faq" if is_faq else "card-esc"
            tag_html = (
                '<span class="tag-faq">✓ FAQ Answer</span>'
                if is_faq
                else '<span class="tag-esc">⚠ Escalated</span>'
            )
            st.markdown(
                f"""
                <div class="{card_cls}">
                    <div>{tag_html} <span class="meta-muted">🕒 {item["time"]}</span></div>
                    <div style="font-weight:700; margin-top:0.35rem;">Q: {item["question"]}</div>
                    <div style="margin-top:0.25rem;">{item["answer"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not escalated_df.empty:
        st.download_button(
            label=f"📥 Download escalated.csv ({len(escalated_df)} logged)",
            data=escalated_df.to_csv(index=False).encode("utf-8"),
            file_name="escalated.csv",
            mime="text/csv",
            use_container_width=True,
        )