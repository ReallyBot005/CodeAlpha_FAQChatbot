# 💬 AskMate — Intelligent FAQ Assistant

> **CodeAlpha Internship Project | NLP FAQ Chatbot**

AskMate is a lightweight, intelligent FAQ chatbot built with Python and Streamlit. It accepts natural-language questions from users and retrieves the most relevant answer from a predefined FAQ knowledge base using **TF-IDF vectorization** and **cosine similarity** — two fundamental NLP techniques.

> ⚠️ **Transparency notice:** AskMate does *not* use a generative AI model or any large language model (LLM). It does *not* generate new text. All answers are retrieved from a predefined FAQ dataset. The NLP techniques are used purely for *matching* a user's question to the best stored answer.

---

## 📸 Screenshots

> *Add your screenshots here after running the application.*

| Chat Interface | Sidebar |
|---|---|
| *(screenshot placeholder)* | *(screenshot placeholder)* |

---

## ✨ Features

- **Natural-language question matching** — ask questions in your own words, not just exact keyword searches
- **TF-IDF vectorization** — converts text into numerical vectors that capture word importance
- **Cosine similarity** — measures how closely your question matches each stored FAQ
- **Configurable similarity threshold** — prevents low-confidence matches from being returned as though they were correct
- **30 FAQ entries** with alternative phrasings across 6 categories
- **Related FAQ suggestions** — when a match is found, other relevant FAQs are displayed
- **Helpful fallback** — when no confident match is found, the chatbot says so and suggests alternatives
- **Conversation history** — maintained with Streamlit session state
- **Clean, professional UI** — light theme with indigo accents, chat bubbles, match score chips, and category labels
- **Clear conversation button** — reset the chat at any time
- **Example questions** — clickable prompts to help users get started

---

## 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11+ | Core programming language |
| Streamlit | Web UI framework |
| scikit-learn | TF-IDF vectorization and cosine similarity |
| NLTK | Text tokenization and stop-word removal |
| NumPy | Array operations for similarity ranking |

---

## 🧠 How It Works

### 1. Text Preprocessing

Before any text is compared, it is normalised using the `preprocess()` function:

1. **Lowercase conversion** — "How do I TRACK my Order?" → "how do i track my order?"
2. **Punctuation removal** — removes non-alphanumeric characters with regex
3. **Whitespace normalisation** — collapses multiple spaces into one
4. **Tokenization** — splits text into individual word tokens using `nltk.word_tokenize`
5. **Stop-word removal** — discards common English words ("the", "is", "a", etc.) that carry little discriminating information

> **Why no stemming?** Stemming (reducing words to their root form) is intentionally omitted. For short FAQ queries, raw tokens already perform well, and stemming can produce ugly or incorrect root forms that make the code harder to understand and explain.

The same preprocessing function is applied to **both** the stored FAQ texts and every user query, ensuring the comparison is always like-for-like.

---

### 2. TF-IDF Vectorization

**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**. It converts text into numerical vectors that capture the *importance* of each word.

| Component | What it measures | Effect |
|---|---|---|
| **Term Frequency (TF)** | How often a word appears in *this* FAQ | Common words in one FAQ get higher weight |
| **Inverse Document Frequency (IDF)** | How rare a word is across *all* FAQs | Words unique to one FAQ get boosted; words appearing everywhere get penalised |
| **TF × IDF** | Importance of a word to a specific FAQ | Distinctive vocabulary is weighted more than generic vocabulary |

**Implementation details:**
- `ngram_range=(1, 2)` — includes both single words ("shipping") and two-word phrases ("free shipping"), so multi-word queries match more accurately
- `sublinear_tf=True` — applies `log(1 + tf)` to dampen the effect of repeated terms
- Each FAQ's primary question **and all its alternative phrasings** are combined into a single document, so the vocabulary of each FAQ is richer and paraphrased queries are more likely to match

---

### 3. Cosine Similarity

Once both the FAQ vectors and the user query vector are computed, **cosine similarity** measures how closely they match.

```
similarity = (A · B) / (||A|| × ||B||)
```

- **Score = 1.0** → perfectly identical direction (identical vocabulary distribution)
- **Score = 0.0** → completely orthogonal (no shared vocabulary at all)
- **Score between 0 and 1** → partial overlap

The FAQ with the **highest cosine similarity score** is selected as the best match.

> ⚠️ **Important:** TF-IDF cosine similarity scores are **not calibrated probabilities**. A score of 0.65 does not mean "65% chance this is correct." They are relative ranking measures. Always interpret them as "FAQ A is more similar to the query than FAQ B," not as absolute confidence values.

---

### 4. Similarity Threshold

A **configurable threshold** prevents the chatbot from returning a weakly-matching answer as though it were a confident result.

- If the best cosine similarity score is **≥ threshold** → the FAQ answer is returned
- If the best score is **< threshold** → AskMate responds with a helpful "not confident" message and suggests related FAQs

The threshold can be adjusted in the sidebar (default: **0.20**). Lowering it makes the chatbot more permissive; raising it makes it stricter.

---

## 📁 Project Structure

```
CodeAlpha_FAQChatbot/
│
├── app.py              # Main Streamlit application
├── faq_data.json       # FAQ knowledge base (30 entries, 6 categories)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git exclusions
```

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.11 or higher
- pip

### Step 1 — Clone or download the repository

```bash
git clone https://github.com/your-username/CodeAlpha_FAQChatbot.git
cd CodeAlpha_FAQChatbot
```

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3 — Activate the virtual environment

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

> NLTK resources (`punkt_tab` and `stopwords`) are downloaded automatically the first time the application starts.

### Step 5 — Run the application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 💬 Example Questions

Try these questions in AskMate to see different matching scenarios:

| Question | Expected match |
|---|---|
| "How do I track my order?" | Order tracking FAQ |
| "I forgot my password" | Password reset FAQ |
| "Where is my package?" | Order tracking FAQ (via alternative phrasing) |
| "My item arrived broken" | Damaged product FAQ |
| "Is there free delivery?" | Shipping cost FAQ |
| "Can I get my money back?" | Refund FAQ |
| "How do I close my account?" | Account deletion FAQ |
| "What ways can I pay?" | Payment methods FAQ |
| "Why was my card declined?" | Failed payment FAQ |
| "How do I cancel my purchase?" | Order cancellation FAQ |
| "Tell me about quantum physics" | No match (threshold not met) |
| *(empty message)* | Friendly prompt to type a question |

---

## ⚙️ FAQ Knowledge Base

The FAQ dataset (`faq_data.json`) contains **30 entries** across **6 categories**:

| Category | Topics covered |
|---|---|
| Account | Registration, login, password reset, profile updates, account deletion, privacy |
| Orders | Tracking, cancellation, damaged items, wrong items, address changes |
| Payments | Accepted methods, failed payments, duplicate charges, gift cards |
| Returns | Return policy, refunds, exchanges |
| Shipping | Shipping cost, delivery time, international shipping |
| Promotions | Coupons, loyalty rewards |

Each FAQ entry includes:
- A primary question
- A detailed answer
- Several alternative phrasings (used to enrich the TF-IDF representation)
- A category label

---

## ⚠️ Limitations

1. **Vocabulary dependency** — TF-IDF relies on word overlap. Semantically similar words with different spellings ("cheap" vs "affordable") will not match unless covered by alternative phrasings.
2. **No semantic understanding** — The model does not understand meaning; it matches vocabulary distributions. A question worded very differently from all stored FAQs may not match even if the intent is identical.
3. **Fixed knowledge base** — AskMate can only answer questions covered by the stored FAQs. It cannot answer questions outside the dataset.
4. **No conversation memory** — Each question is matched independently. AskMate does not use the context of previous questions.
5. **Scores are not probabilities** — A high similarity score does not guarantee a correct match; it means the vocabulary overlaps significantly.
6. **Small dataset** — With 30 FAQs, IDF weights are computed on a very small corpus. On larger datasets, TF-IDF performance generally improves.

---

## 📄 License

This project is created for educational purposes as part of the **CodeAlpha internship programme**. The FAQ data is entirely fictional and does not represent the policies of any real company.

---

*Built with ❤️ using Python, Streamlit, scikit-learn, and NLTK.*
