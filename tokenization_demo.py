"""
Streamlit app demonstrating how LLMs tokenize text.
Supports tiktoken (OpenAI BPE) and HuggingFace tokenizers (BERT WordPiece, GPT-2 BPE, T5 SentencePiece).
"""

import streamlit as st
import tiktoken
import html
import colorsys
from transformers import AutoTokenizer


def _token_color(index: int, total: int) -> str:
    """Generate a distinct hue for each token index using golden ratio."""
    hue = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.4, 0.95)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


# ── Tokenizer backends ─────────────────────────────────────────────────────

TIKTOKEN_ENCODINGS = {
    "cl100k_base": "GPT-4, GPT-3.5-turbo",
    "o200k_base": "GPT-4o, GPT-4o-mini",
    "p50k_base": "Codex, text-davinci",
    "r50k_base": "GPT-3 (gpt2)",
}

HF_MODELS = {
    "bert-base-uncased": "BERT WordPiece (~30k vocab)",
    "gpt2": "GPT-2 BPE (~50k vocab)",
    "google-t5/t5-small": "T5 SentencePiece (~32k vocab)",
}

SAMPLE_TEXTS = {
    "Hello world!": "Hello world!",
    "Subword composition": "tokenization embedding uninteresting",
    "Code snippet": "def hello():\n    print('Hello, World!')",
    "With emoji": "Hello 👋 World 🌍",
    "Longer paragraph": (
        "Large language models process text by breaking it into tokens. "
        "Each token can be a word, subword, or even a single character depending on the tokenizer."
    ),
}


@st.cache_resource
def load_hf_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_tiktoken(text: str, encoding_name: str):
    """Returns list of (token_id, token_string) tuples."""
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)
    return [(tid, enc.decode([tid])) for tid in token_ids]


def tokenize_hf(text: str, model_name: str):
    """Returns list of (token_id, token_string) tuples."""
    tokenizer = load_hf_tokenizer(model_name)
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    tokens_str = tokenizer.convert_ids_to_tokens(token_ids)
    return list(zip(token_ids, tokens_str))


# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="LLM Tokenization Demo", page_icon="🔤", layout="wide")

st.title("🔤 LLM Tokenization Demo")
st.markdown(
    "Explore how Large Language Models convert text into **tokens**—the fundamental units "
    "they process. Type or paste text below to see token IDs, decoded strings, and statistics."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    backend = st.radio("Tokenizer library", ["tiktoken (OpenAI)", "HuggingFace Transformers"])

    if backend == "tiktoken (OpenAI)":
        tokenizer_key = st.selectbox(
            "Encoding",
            options=list(TIKTOKEN_ENCODINGS.keys()),
            format_func=lambda x: f"{x} ({TIKTOKEN_ENCODINGS[x]})",
        )
        st.caption(TIKTOKEN_ENCODINGS[tokenizer_key])
    else:
        tokenizer_key = st.selectbox(
            "Model tokenizer",
            options=list(HF_MODELS.keys()),
            format_func=lambda x: f"{x} ({HF_MODELS[x]})",
        )
        st.caption(HF_MODELS[tokenizer_key])


# ── Input ───────────────────────────────────────────────────────────────────

st.subheader("Input text")
sample_key = st.radio(
    "Quick samples:",
    options=list(SAMPLE_TEXTS.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
default_text = SAMPLE_TEXTS.get(sample_key, "")


text = st.text_area(
    "Enter or edit text to tokenize:",
    value=default_text,
    height=120,
    placeholder="Type or paste text here...",
    key=f"text_input_{sample_key}",
)

if not text.strip():
    st.info("Enter some text above to see how it gets tokenized.")
    st.stop()

# ── Tokenize ────────────────────────────────────────────────────────────────

try:
    if backend == "tiktoken (OpenAI)":
        token_pairs = tokenize_tiktoken(text, tokenizer_key)
    else:
        token_pairs = tokenize_hf(text, tokenizer_key)
except Exception as e:
    st.error(f"Tokenization failed: {e}")
    st.stop()

token_ids = [tid for tid, _ in token_pairs]
token_strings = [ts for _, ts in token_pairs]

# ── Metrics ─────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Token count", len(token_ids))
with col2:
    st.metric("Character count", len(text))
with col3:
    ratio = len(token_ids) / len(text) if len(text) > 0 else 0
    st.metric("Tokens per character", f"{ratio:.3f}")

# ── Visualization ───────────────────────────────────────────────────────────

st.subheader("Token visualization")

# 1. Highlighted text view
st.markdown("**Text with token boundaries** — each color = one token")
highlighted_parts = []
for i, (tid, ts) in enumerate(token_pairs):
    color = _token_color(i, len(token_pairs))
    display = html.escape(ts).replace("\n", "↵").replace(" ", "·")
    highlighted_parts.append(
        f'<span style="background:{color};padding:2px 4px;margin:1px;border-radius:3px;'
        f'font-family:monospace" title="Token {i} (ID: {tid})">{display}</span>'
    )
st.markdown(
    f'<div style="line-height:2;word-break:break-all;font-size:14px">{"".join(highlighted_parts)}</div>',
    unsafe_allow_html=True,
)

# 2. Token block strip
st.markdown("**Token sequence** — each block is one token")
block_html = []
for i, (tid, ts) in enumerate(token_pairs):
    display = html.escape(repr(ts)[1:-1]) if ts else "∅"
    color = _token_color(i, len(token_pairs))
    block_html.append(
        f'<span style="display:inline-block;background:{color};color:#333;padding:4px 8px;'
        f'margin:2px;border-radius:4px;font-family:monospace;font-size:12px" '
        f'title="Token {i} | ID: {tid}">{display}</span>'
    )
st.markdown(
    f'<div style="line-height:2.2">{"".join(block_html)}</div>',
    unsafe_allow_html=True,
)

# ── Token table ─────────────────────────────────────────────────────────────

st.subheader("Token breakdown")
token_data = []
for i, (tid, ts) in enumerate(token_pairs):
    display_str = repr(ts)[1:-1] if ts else ""
    token_data.append({"Index": i, "Token ID": tid, "Token string": display_str})

st.dataframe(token_data, width="stretch", hide_index=True)

# ── Character-to-token mapping ──────────────────────────────────────────────

st.subheader("Character-to-token mapping")
st.markdown(
    "Each token below shows its decoded string. "
    "Note: BPE/WordPiece tokens don't always align 1:1 with characters."
)

mapping_rows = []
for i, (tid, ts) in enumerate(token_pairs):
    mapping_rows.append(f"**Token {i}** (ID: {tid}) → `{repr(ts)}`")

st.markdown("\n".join(mapping_rows))

# ── Round-trip verification ─────────────────────────────────────────────────

st.subheader("Round-trip verification")
if backend == "tiktoken (OpenAI)":
    enc = tiktoken.get_encoding(tokenizer_key)
    reconstructed = enc.decode(token_ids)
else:
    tokenizer = load_hf_tokenizer(tokenizer_key)
    reconstructed = tokenizer.decode(token_ids, skip_special_tokens=True)

match = "✅ Match" if reconstructed == text else "⚠️ Mismatch (tokenizer may normalize input, e.g. lowercasing)"
st.code(reconstructed, language=None)
st.caption(match)
