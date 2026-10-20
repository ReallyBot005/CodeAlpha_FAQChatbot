"""
Test script for AskMate NLP matching logic.
Run with: .venv\Scripts\python.exe test_matching.py
"""
import json
import os
import re
import sys

import nltk

# Ensure NLTK data is present
for rpath, rname in {"tokenizers/punkt_tab": "punkt_tab", "corpora/stopwords": "stopwords"}.items():
    try:
        nltk.data.find(rpath)
    except LookupError:
        nltk.download(rname, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

STOPWORDS = set(stopwords.words("english"))
THRESHOLD = 0.20


def preprocess(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
    return " ".join(tokens)


# Load FAQ dataset
try:
    with open("faq_data.json", encoding="utf-8") as f:
        raw = json.load(f)
    faqs_raw = raw["faqs"]
    validated = [
        {
            "question": f["question"],
            "answer": f["answer"],
            "alternatives": f.get("alternatives", []),
            "category": f.get("category", "General"),
        }
        for f in faqs_raw
        if f.get("question") and f.get("answer")
    ]
    print(f"[OK] Loaded {len(validated)} FAQs from faq_data.json")
except Exception as e:
    print(f"[ERROR] Could not load faq_data.json: {e}")
    sys.exit(1)

# Build TF-IDF engine
docs = []
for faq in validated:
    combined = " ".join([faq["question"]] + faq["alternatives"])
    docs.append(preprocess(combined))

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(docs)
print(f"[OK] TF-IDF matrix built: {tfidf_matrix.shape[0]} FAQs x {tfidf_matrix.shape[1]} features")


def match(query):
    pq = preprocess(query)
    if not pq:
        return None, 0.0
    qv = vectorizer.transform([pq])
    scores = cosine_similarity(qv, tfidf_matrix).flatten()
    idx = int(np.argmax(scores))
    return validated[idx], float(scores[idx])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

tests = [
    # (query, description, expect_match)
    ("How do I track my order?",           "Exact match — order tracking",             True),
    ("Where is my package?",               "Alternative phrasing — order tracking",    True),
    ("My item arrived broken",             "Alternative phrasing — damaged product",   True),
    ("I forgot my password",               "Rephrased — password reset",               True),
    ("Is there free delivery?",            "Rephrased — shipping cost",                True),
    ("Can I get my money back?",           "Rephrased — refund policy",                True),
    ("How do I close my account?",         "Rephrased — account deletion",             True),
    ("WHY WAS MY CARD DECLINED???",        "Unusual caps + punctuation — failed payment", True),
    ("Tell me about quantum physics",      "Unrelated query — should NOT match",       False),
    ("!!! ???",                            "Punctuation only — should NOT match",      False),
    ("",                                   "Empty input — should NOT match",           False),
    ("How long until I get my order?",     "Follow-up style question — delivery time", True),
    ("what are your opening hours",        "Lowercase no punctuation — working hours", True),
]

SEP = "=" * 72
print(f"\n{SEP}")
print("  AskMate — NLP Matching Test Results")
print(SEP)

pass_count = 0
fail_count = 0

for query, description, expect_match in tests:
    faq, score = match(query)
    actually_matched = score >= THRESHOLD

    status_ok = (actually_matched == expect_match)
    status_icon = "PASS" if status_ok else "FAIL"

    if status_ok:
        pass_count += 1
    else:
        fail_count += 1

    matched_q = ""
    if faq:
        q = faq["question"]
        matched_q = (q[:58] + "...") if len(q) > 60 else q

    print(f"\n  [{status_icon}] {description}")
    print(f"        Query   : {repr(query[:55])}")
    print(f"        Score   : {score:.4f}  (threshold={THRESHOLD})  => {'MATCH' if actually_matched else 'NO MATCH'}")
    if matched_q:
        print(f"        Matched : {matched_q}")

print(f"\n{SEP}")
print(f"  Results: {pass_count} passed, {fail_count} failed out of {len(tests)} tests")
print(SEP)
print()

if fail_count > 0:
    sys.exit(1)
