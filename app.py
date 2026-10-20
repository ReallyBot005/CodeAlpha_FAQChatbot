"""
AskMate — Intelligent FAQ Assistant
====================================
CodeAlpha Internship Project

Approach:
  1. Load a predefined FAQ dataset from faq_data.json.
  2. Pre-process all FAQ questions and alternative phrasings with NLTK.
  3. Build a TF-IDF matrix from those processed texts.
  4. On each user query, pre-process it the same way and compute cosine
     similarity against every FAQ vector.
  5. Return the best-matching FAQ (question + answer + score) when the
     similarity meets the configurable threshold; otherwise return a
     helpful fallback message.

Note: TF-IDF similarity scores are NOT calibrated probabilities. A score
of 0.72 does not mean "72 % chance of being correct." They are relative
similarity measures — use them only for ranking, not as absolute confidence.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import json
import os
import re
import string

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import nltk
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# NLTK setup — download required corpora once, silently
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = {
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
}

for resource_path, resource_name in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAQ_FILE = os.path.join(os.path.dirname(__file__), "faq_data.json")
DEFAULT_THRESHOLD = 0.20          # Minimum cosine similarity to accept a match
MAX_RELATED = 3                    # Number of "related FAQ" suggestions to show
ENGLISH_STOPWORDS = set(stopwords.words("english"))

# Example questions shown in the UI to guide users
EXAMPLE_QUESTIONS = [
    "How do I track my order?",
    "I forgot my password, what do I do?",
    "What payment methods are accepted?",
    "How do I return a damaged item?",
    "Is there free shipping?",
    "How long does delivery take?",
    "Can I cancel my order?",
    "How do I get a refund?",
]

# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    """
    Normalise a text string for TF-IDF matching.

    Steps
    -----
    1. Guard against empty / non-string input.
    2. Lower-case the text.
    3. Remove punctuation (preserving apostrophes avoids splitting contractions
       like "don't" into "don" + "t", which would hurt matching).
    4. Tokenise with NLTK word_tokenize.
    5. Remove stop-words (common words like "the", "is" that carry little
       discriminating information for FAQ matching).
    6. Rejoin tokens with a single space and strip extra whitespace.

    Parameters
    ----------
    text : str
        Raw input text (user query or stored FAQ question).

    Returns
    -------
    str
        Cleaned, lower-cased, stop-word-filtered text ready for TF-IDF.
        Returns an empty string if input is invalid or empty.

    Notes
    -----
    We deliberately do NOT stem or lemmatize. Stemming can mangle words
    ("shipping" → "ship", "refund" → "refund" — fine; but "policies" →
    "polici" — odd), and TF-IDF with raw tokens already performs well for
    short FAQ queries. Keeping words recognisable also makes the code easier
    to understand and explain.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lower-case
    text = text.lower()

    # Remove punctuation characters (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenise
    tokens = word_tokenize(text)

    # Remove stop-words and pure-numeric tokens
    tokens = [
        token for token in tokens
        if token not in ENGLISH_STOPWORDS and not token.isdigit()
    ]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# FAQ data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_faq_data(filepath: str) -> list:
    """
    Load and validate the FAQ dataset from a JSON file.

    Returns a list of FAQ dicts. Each dict must contain at minimum:
      - "question"  : str
      - "answer"    : str
    Optionally:
      - "alternatives" : list[str]
      - "category"     : str
      - "id"           : int

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the JSON structure is not a dict with a "faqs" key, or if any
        FAQ entry is missing required fields.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"FAQ data file not found at: {filepath}\n"
            "Please ensure faq_data.json is in the same directory as app.py."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Support both {"faqs": [...]} and a bare list
    if isinstance(raw, dict):
        faqs = raw.get("faqs", [])
    elif isinstance(raw, list):
        faqs = raw
    else:
        raise ValueError("faq_data.json must contain either a JSON object with a 'faqs' key or a JSON array.")

    if not faqs:
        raise ValueError("The FAQ dataset is empty. Please add at least one FAQ entry.")

    validated = []
    for i, faq in enumerate(faqs):
        if not isinstance(faq, dict):
            continue
        q = faq.get("question", "").strip()
        a = faq.get("answer", "").strip()
        if not q or not a:
            # Skip malformed entries rather than crashing
            continue
        validated.append({
            "id":           faq.get("id", i + 1),
            "question":     q,
            "answer":       a,
            "category":     faq.get("category", "General"),
            "alternatives": [alt.strip() for alt in faq.get("alternatives", []) if alt.strip()],
        })

    if not validated:
        raise ValueError("No valid FAQ entries found after validation.")

    return validated


# ---------------------------------------------------------------------------
# TF-IDF engine
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_tfidf_engine(faq_filepath: str):
    """
    Build and cache the TF-IDF vectorizer and matrix from the FAQ dataset.

    How TF-IDF works
    ----------------
    TF-IDF (Term Frequency - Inverse Document Frequency) converts text
    into numerical vectors:

    * Term Frequency (TF): How often a word appears in a document
      (FAQ question/alternative) relative to that document's length.
    * Inverse Document Frequency (IDF): How rare a word is across
      ALL documents. Words that appear in every FAQ (e.g. "the") get a
      low IDF weight; rare, specific words (e.g. "invoice") get a higher
      weight -- making them more discriminating.

    The product TF x IDF gives each word a score that balances local
    frequency with global rarity. The resulting vector for each FAQ
    reflects its distinctive vocabulary.

    Construction strategy
    ---------------------
    We concatenate the primary question with all its alternative phrasings
    into a single "document" per FAQ. This enriches the vocabulary of each
    FAQ entry so that paraphrased user queries can still match correctly.

    Parameters
    ----------
    faq_filepath : str
        Path to faq_data.json.

    Returns
    -------
    tuple
        (faqs, vectorizer, tfidf_matrix, faq_documents)
        - faqs           : list of validated FAQ entries
        - vectorizer     : TfidfVectorizer (fitted)
        - tfidf_matrix   : sparse matrix, shape (n_faqs, n_features)
        - faq_documents  : list of pre-processed text used per FAQ
    """
    faqs = load_faq_data(faq_filepath)

    # Build one text document per FAQ: question + alternatives
    faq_documents = []
    for faq in faqs:
        parts = [faq["question"]] + faq["alternatives"]
        combined = " ".join(parts)
        faq_documents.append(preprocess(combined))

    # TfidfVectorizer settings:
    # - ngram_range=(1, 2): include both single words and two-word phrases
    #   so "free shipping" is treated as a meaningful unit.
    # - min_df=1: keep all terms (small dataset).
    # - sublinear_tf=True: apply log(1 + tf) to dampen the effect of
    #   high-frequency terms -- improves performance on short texts.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        preprocessor=None,   # we pre-process ourselves
        tokenizer=None,
    )
    tfidf_matrix = vectorizer.fit_transform(faq_documents)

    return faqs, vectorizer, tfidf_matrix, faq_documents


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def find_best_match(
    user_query: str,
    faqs: list,
    vectorizer,
    tfidf_matrix,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Find the best-matching FAQ for a user query using cosine similarity.

    How cosine similarity works
    ---------------------------
    Two TF-IDF vectors are compared by calculating the cosine of the angle
    between them. A score of 1.0 means identical direction (perfect match);
    0.0 means completely orthogonal (no shared vocabulary). Scores between
    0 and 1 indicate partial overlap -- the higher the score, the more
    semantically relevant the FAQ is to the query.

    Formula: similarity = (A . B) / (||A|| x ||B||)

    Parameters
    ----------
    user_query : str
        Raw question entered by the user.
    faqs : list
        Validated FAQ list returned by load_faq_data().
    vectorizer : TfidfVectorizer
        Fitted vectorizer from build_tfidf_engine().
    tfidf_matrix : sparse matrix
        Pre-computed FAQ vectors from build_tfidf_engine().
    threshold : float
        Minimum similarity score required to return a match.
        If the best score is below this value, a "no match" result
        is returned instead.

    Returns
    -------
    dict with keys:
        matched     : bool   -- whether a confident match was found
        score       : float  -- best cosine similarity score (0-1)
        faq         : dict   -- the best-matching FAQ entry (or None)
        related     : list   -- up to MAX_RELATED other relevant FAQs
        query_empty : bool   -- True if the user sent an empty query
    """
    result = {
        "matched": False,
        "score": 0.0,
        "faq": None,
        "related": [],
        "query_empty": False,
    }

    # --- Guard: empty query ---
    cleaned_query = user_query.strip() if isinstance(user_query, str) else ""
    if not cleaned_query:
        result["query_empty"] = True
        return result

    # --- Pre-process the query the same way as the FAQ documents ---
    processed_query = preprocess(cleaned_query)
    if not processed_query:
        # Query became empty after pre-processing (e.g., only punctuation)
        result["query_empty"] = True
        return result

    # --- Vectorize the query ---
    try:
        query_vector = vectorizer.transform([processed_query])
    except Exception:
        # Vectorizer failed -- return no match rather than crashing
        return result

    # --- Compute cosine similarity against all FAQ vectors ---
    try:
        scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    except Exception:
        return result

    if scores.size == 0:
        return result

    # --- Rank FAQs by score ---
    ranked_indices = np.argsort(scores)[::-1]   # descending
    best_idx = ranked_indices[0]
    best_score = float(scores[best_idx])

    result["score"] = best_score

    # --- Apply threshold ---
    if best_score < threshold:
        # Collect related suggestions even when no confident match found
        result["related"] = _collect_related(ranked_indices, scores, threshold / 2, faqs)
        return result

    result["matched"] = True
    result["faq"] = faqs[best_idx]

    # --- Collect related FAQs (exclude the best match itself) ---
    related_indices = [idx for idx in ranked_indices[1:] if scores[idx] > threshold / 2]
    result["related"] = [faqs[idx] for idx in related_indices[:MAX_RELATED]]

    return result


def _collect_related(ranked_indices, scores, min_score: float, faqs: list) -> list:
    """Return up to MAX_RELATED FAQs above min_score, in ranked order."""
    related = []
    for idx in ranked_indices:
        if len(related) >= MAX_RELATED:
            break
        if scores[idx] >= min_score:
            related.append(faqs[idx])
    return related


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AskMate -- Intelligent FAQ Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS -- clean, minimal, light theme with indigo accents
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 60%, #f5f0ff 100%);
}

.header-banner {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #4338ca 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 8px 32px rgba(79, 70, 229, 0.25);
}
.header-banner h1 {
    margin: 0 0 6px 0;
    font-size: 2.0rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.header-banner p {
    margin: 0;
    font-size: 0.95rem;
    opacity: 0.85;
    line-height: 1.5;
}
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    margin-top: 10px;
    letter-spacing: 0.3px;
}

.user-bubble {
    display: flex;
    justify-content: flex-end;
    margin: 8px 0;
}
.user-bubble-inner {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}

.bot-bubble {
    display: flex;
    justify-content: flex-start;
    margin: 8px 0;
    gap: 10px;
    align-items: flex-start;
}
.bot-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
}
.bot-bubble-inner {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    color: #1f2937;
}
.bot-bubble-inner .match-q {
    font-weight: 600;
    color: #4f46e5;
    font-size: 0.85rem;
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e8e4ff;
}
.score-chip {
    display: inline-block;
    background: #eef2ff;
    color: #4f46e5;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 8px;
}
.category-chip {
    display: inline-block;
    background: #f0fdf4;
    color: #16a34a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 4px;
    margin-left: 4px;
}
.no-match-bubble {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    color: #92400e;
}

.related-section {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #e8e4ff;
    font-size: 0.82rem;
    color: #6b7280;
}
.related-pill {
    display: inline-block;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.78rem;
    color: #4f46e5;
    margin: 3px 3px 0 0;
}

.msg-time {
    font-size: 0.72rem;
    color: #9ca3af;
    margin-top: 4px;
    text-align: right;
}

.example-section {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 18px;
    margin-top: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.example-section h4 {
    margin: 0 0 10px 0;
    color: #374151;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.sidebar-card {
    background: white;
    border-radius: 10px;
    padding: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 12px;
    font-size: 0.84rem;
    line-height: 1.6;
}
.sidebar-card h4 {
    color: #4f46e5;
    font-weight: 600;
    margin: 0 0 8px 0;
    font-size: 0.88rem;
    border-bottom: 1px solid #eef2ff;
    padding-bottom: 6px;
}
.sidebar-stat {
    color: #374151;
    margin: 4px 0;
}

.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #d1d5db, transparent);
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False

if "error_msg" not in st.session_state:
    st.session_state.error_msg = None


# ---------------------------------------------------------------------------
# Load FAQ engine
# ---------------------------------------------------------------------------

try:
    faqs, vectorizer, tfidf_matrix, faq_documents = build_tfidf_engine(FAQ_FILE)
    st.session_state.engine_ready = True
except FileNotFoundError as e:
    st.session_state.error_msg = str(e)
except ValueError as e:
    st.session_state.error_msg = str(e)
except Exception as e:
    st.session_state.error_msg = f"Unexpected error loading FAQ engine: {e}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px 0;">
        <div style="font-size:2.8rem; margin-bottom:4px;">💬</div>
        <div style="font-size:1.15rem; font-weight:700; color:#4f46e5;">AskMate</div>
        <div style="font-size:0.78rem; color:#6b7280;">Intelligent FAQ Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>📋 Project Info</h4>
        <div class="sidebar-stat">🎓 CodeAlpha Internship</div>
        <div class="sidebar-stat">📌 NLP FAQ Chatbot</div>
        <div class="sidebar-stat">🗂 Topic: Online Shopping Support</div>
    </div>
    """, unsafe_allow_html=True)

    faq_count = len(faqs) if st.session_state.engine_ready else 0
    category_count = len(set(f["category"] for f in faqs)) if st.session_state.engine_ready else 0
    alt_count = sum(len(f["alternatives"]) for f in faqs) if st.session_state.engine_ready else 0

    st.markdown(f"""
    <div class="sidebar-card">
        <h4>📊 Knowledge Base</h4>
        <div class="sidebar-stat">❓ {faq_count} FAQ entries</div>
        <div class="sidebar-stat">🗂 {category_count} categories</div>
        <div class="sidebar-stat">🔀 {alt_count} alternative phrasings</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>⚙️ How It Works</h4>
        This assistant uses <strong>TF-IDF vectorization</strong> and
        <strong>cosine similarity</strong> to find the most relevant answer
        from a predefined FAQ knowledge base.<br><br>
        It does <em>not</em> use generative AI or any paid LLM API.
        All answers come from the stored FAQ dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>🛠 Technologies</h4>
        <div class="sidebar-stat">🐍 Python 3.11+</div>
        <div class="sidebar-stat">📊 scikit-learn (TF-IDF)</div>
        <div class="sidebar-stat">📝 NLTK (preprocessing)</div>
        <div class="sidebar-stat">🖥 Streamlit (UI)</div>
        <div class="sidebar-stat">🔢 NumPy</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚖️ Similarity Threshold**")
    threshold = st.slider(
        label="Minimum similarity score",
        min_value=0.05,
        max_value=0.60,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help=(
            "If the best-matching FAQ scores below this value, AskMate will "
            "say it is not confident rather than returning an unrelated answer. "
            "Lower = more permissive. Higher = stricter. "
            "Note: TF-IDF scores are relative rankings, not calibrated probabilities."
        ),
    )
    st.caption(
        f"Current: **{threshold:.2f}**. Scores are relative similarity "
        "measures, not confidence percentages."
    )

    st.markdown("---")
    st.caption(
        "All answers are demonstration data and do not represent "
        "the policies of any real company."
    )


# ---------------------------------------------------------------------------
# Helper: render a single chat turn
# ---------------------------------------------------------------------------

def render_message(msg: dict) -> None:
    """Render a single message bubble (user or assistant)."""
    import html as html_lib

    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta", {})
    timestamp = msg.get("timestamp", "")

    if role == "user":
        # Escape user text so special HTML chars don't break the layout
        safe_content = html_lib.escape(content)
        html_block = (
            '<div class="user-bubble">'
            '<div>'
            f'<div class="user-bubble-inner">{safe_content}</div>'
            f'<div class="msg-time">{timestamp} \U0001f9d1</div>'
            '</div>'
            '</div>'
        )
        st.markdown(html_block, unsafe_allow_html=True)
        return

    if role != "assistant":
        return

    matched = meta.get("matched", False)
    score = meta.get("score", 0.0)
    matched_q = html_lib.escape(meta.get("matched_q", ""))
    category = html_lib.escape(meta.get("category", ""))
    related = meta.get("related", [])
    query_empty = meta.get("query_empty", False)

    if query_empty:
        safe_content = html_lib.escape(content)
        html_block = (
            '<div class="bot-bubble">'
            '<div class="bot-avatar">\U0001f4ac</div>'
            '<div>'
            f'<div class="bot-bubble-inner" style="font-style:italic;">{safe_content}</div>'
            f'<div class="msg-time">\U0001f916 {timestamp}</div>'
            '</div>'
            '</div>'
        )
        st.markdown(html_block, unsafe_allow_html=True)
        return

    # Build related-pills HTML (question text is escaped)
    related_html = ""
    if related:
        pills = "".join(
            '<span class="related-pill">\U0001f4ce '
            + html_lib.escape(r["question"][:55])
            + ("\u2026" if len(r["question"]) > 55 else "")
            + "</span>"
            for r in related
        )
        if matched:
            related_html = (
                '<div class="related-section">'
                '<strong>You might also want to know:</strong><br>'
                + pills
                + '</div>'
            )
        else:
            related_html = "<br><strong>Suggested FAQs:</strong><br>" + pills

    if matched:
        score_pct = int(score * 100)
        # content is FAQ answer text — escape it so stray angle-brackets don't break HTML
        safe_content = html_lib.escape(content)
        html_block = (
            '<div class="bot-bubble">'
            '<div class="bot-avatar">\U0001f4ac</div>'
            '<div style="max-width:72%;">'
            '<div class="bot-bubble-inner">'
            f'<div class="match-q">\U0001f4cc {matched_q}</div>'
            f'{safe_content}'
            '<br>'
            f'<span class="score-chip">Match: {score_pct}%</span>'
            f'<span class="category-chip">{category}</span>'
            f'{related_html}'
            '</div>'
            f'<div class="msg-time">\U0001f916 {timestamp}</div>'
            '</div>'
            '</div>'
        )
    else:
        # For no-match responses the content already has <strong>/<br> tags
        # so we keep it as-is (it is authored by the app, not user input)
        html_block = (
            '<div class="bot-bubble">'
            '<div class="bot-avatar">\U0001f914</div>'
            '<div style="max-width:72%;">'
            f'<div class="no-match-bubble">{content}{related_html}</div>'
            f'<div class="msg-time">\U0001f916 {timestamp}</div>'
            '</div>'
            '</div>'
        )

    st.markdown(html_block, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.markdown("""
<div class="header-banner">
    <h1>💬 AskMate</h1>
    <p>Your Intelligent FAQ Assistant for online shopping and customer support questions.</p>
    <span class="header-badge">🔬 Powered by TF-IDF + Cosine Similarity — No generative AI</span>
</div>
""", unsafe_allow_html=True)

# Fatal error (engine failed to load)
if st.session_state.error_msg:
    st.error(f"Failed to load FAQ engine:\n\n{st.session_state.error_msg}")
    st.info("Please ensure faq_data.json exists in the project directory and is valid JSON.")
    st.stop()

# Top controls row
col_title, col_btn = st.columns([6, 1])
with col_btn:
    if st.button("🗑 Clear", help="Clear conversation history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown("""
    <div class="bot-bubble">
        <div class="bot-avatar">💬</div>
        <div class="bot-bubble-inner">
            👋 <strong>Hi! I'm AskMate.</strong><br><br>
            I can answer questions about our online shopping and customer support services.
            Type your question below, or pick one of the example questions to get started.<br><br>
            <em>Note: I retrieve answers from a predefined FAQ knowledge base using NLP similarity
            matching — I do not generate new answers.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_message(msg)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Example questions (shown when conversation is short)
# ---------------------------------------------------------------------------

if len(st.session_state.messages) < 4:
    st.markdown("""
    <div class="example-section">
        <h4>💡 Try asking one of these:</h4>
    </div>
    """, unsafe_allow_html=True)

    ex_cols = st.columns(4)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with ex_cols[i % 4]:
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state["_pending_query"] = q
                st.rerun()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_input = st.chat_input(
    placeholder="Ask me anything about orders, returns, shipping, accounts…",
)

# Handle pending query from example button click
if "_pending_query" in st.session_state:
    user_input = st.session_state.pop("_pending_query")

# ---------------------------------------------------------------------------
# Process user input
# ---------------------------------------------------------------------------

if user_input:
    from datetime import datetime
    now_str = datetime.now().strftime("%H:%M")

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": now_str,
    })

    # Run matching
    with st.spinner("Searching knowledge base…"):
        result = find_best_match(
            user_query=user_input,
            faqs=faqs,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            threshold=threshold,
        )

    # Build assistant response
    if result["query_empty"]:
        answer_text = (
            "It looks like you sent an empty message. "
            "Please type a question and I will do my best to help! 😊"
        )
        meta = {"query_empty": True, "matched": False, "score": 0.0, "related": []}

    elif result["matched"]:
        faq = result["faq"]
        answer_text = faq["answer"]
        meta = {
            "matched": True,
            "score": result["score"],
            "matched_q": faq["question"],
            "category": faq["category"],
            "related": result["related"],
            "query_empty": False,
        }

    else:
        score = result["score"]
        related = result["related"]
        answer_text = (
            "I'm not confident I found the right answer for your question. "
            f"The best match scored only <strong>{int(score*100)}%</strong> similarity, "
            f"which is below my confidence threshold of <strong>{int(threshold*100)}%</strong>.<br><br>"
            "Try rephrasing your question, or browse the suggested FAQs below."
        )
        meta = {
            "matched": False,
            "score": score,
            "related": related,
            "query_empty": False,
        }

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "meta": meta,
        "timestamp": now_str,
    })

    st.rerun()
