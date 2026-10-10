# 🎓 2IS Master's Program Assistant

A chatbot that answers students' questions about the **Master "Innovative Information Systems" (2IS)** at **Université Toulouse Capitole**: admissions, courses, teachers, ECTS, prerequisites, career prospects.

It answers **only from official program documents**, and shows its sources under every answer.

👉 **Try it:** https://rag-chatbot-2is-qcthvwdzoerdnj4xappyfjd.streamlit.app/

> Internship project (Licence 3 MIAGE, 2024-2025) at **IRIT**, supervised by Prof. Chihab Hanachi.

<img width="1454" height="872" alt="image" src="https://github.com/user-attachments/assets/9006ca27-ed37-4133-9c2e-d6f94b411034" />


---

## Why this project?

Every year, the 2IS master receives many international students who ask the same questions again and again. This assistant answers them instantly, any time, in plain language.

## What it can do

- **Answer questions about the program**, using the official website, PDF brochures and the syllabus.
- **List things exactly**: for questions like "list all courses of semester 1", it filters the syllabus with code instead of guessing, so nothing is forgotten.
- **Follow a conversation**: "And who teaches it?" is understood from the previous messages.
- **Stay on topic**: greetings get a friendly reply, and off-topic questions get a polite refusal.
- **Avoid making things up**: the model is told to answer only with the retrieved information, and it shows its sources.

## How it works

The idea is called **RAG** (Retrieval-Augmented Generation): instead of letting the AI answer from memory, we first **find** the relevant passages, then ask the AI to **write the answer from them only**.

**Step 1: Build the knowledge base** (`build_index.py`)
1. Collect the website pages, PDFs and the syllabus (`Syllabus.json`).
2. Cut long texts into small chunks (each course in the syllabus is kept whole).
3. Turn each chunk into numbers (an *embedding*) with `all-MiniLM-L6-v2`.
4. Store them in a **FAISS** index (`vector_db/`) for fast search.

**Step 2: Answer a question** (`streamlit_app.py`)
1. Understand the question: chat, off-topic, list request, or normal question. Rewrite it if it depends on earlier messages.
2. Find the best matching chunks (or filter the syllabus for lists).
3. Ask the AI to write the answer using only those chunks.
4. Show the answer with its sources.

The flow between these steps is managed with **LangGraph**.

## Built with

| What | Tool |
|---|---|
| Flow of the conversation | LangGraph |
| AI model | A free model via [OpenRouter](https://openrouter.ai) (set in `config.py`) |
| Understanding text | `sentence-transformers/all-MiniLM-L6-v2` |
| Search | FAISS |
| Chat history | Upstash Redis |
| Interface | Streamlit |
| Hosting | GitHub + Streamlit Community Cloud |

---

## Run it yourself

### 1. Get the code
```bash
git clone https://github.com/ayaelmajjouti/RAG-Chatbot-2IS.git
cd RAG-Chatbot-2IS
```

### 2. Install
Python **3.11 or 3.12** is recommended.
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
python -m pip install -r requirements.txt
```

### 3. Add your secrets
The app needs three values. **Never write them in the code or commit them.**

| Name | Where to get it |
|---|---|
| `OPENROUTER_API_KEY` | openrouter.ai → API Keys |
| `UPSTASH_REDIS_REST_URL` | console.upstash.com → your database → REST API |
| `UPSTASH_REDIS_REST_TOKEN` | same place |

**Locally**, set them as environment variables. For example in PowerShell:
```powershell
$env:OPENROUTER_API_KEY="your_key"
$env:UPSTASH_REDIS_REST_URL="your_url"
$env:UPSTASH_REDIS_REST_TOKEN="your_token"
```

**On Streamlit Cloud**, open your app → ⋮ → **Settings → Secrets** and paste:
```toml
OPENROUTER_API_KEY = "your_key"
UPSTASH_REDIS_REST_URL = "your_url"
UPSTASH_REDIS_REST_TOKEN = "your_token"
```

### 4. Build the knowledge base
```bash
python build_index.py
```

### 5. Start the app
```bash
streamlit run streamlit_app.py
```
It opens at http://localhost:8501.

---

## Project files

```
.
├── streamlit_app.py     # The web interface
├── main.py              # Main program
├── base_rag.py          # RAG logic
├── config.py            # Settings (model name, secrets read from environment)
├── build_index.py       # Builds the knowledge base and FAISS index
├── evaluate.py          # Automatic tests of the chatbot
├── Syllabus.json        # Course data
├── scraped_content.json # Content collected from the website
├── rag/                 # RAG components
├── scraper/             # Website scraping
├── vector_db/           # FAISS index (needed by the app)
└── requirements.txt     # Python packages
```

## Keeping it up to date

The website changes, so the knowledge base is rebuilt every 4 months (January, April, July, October) by a scheduled task that runs `build_index.py`.

To update by hand: edit `Syllabus.json` or add PDFs, run `python build_index.py`, then push to GitHub. Streamlit redeploys automatically.

## How it was tested

- **Automatic tests** on five types of questions: about courses, lists, greetings, off-topic, and trick questions with no answer in the documents. They check the routing, the response time, and the quality of the answers.
- **Real users**: fellow students tried the deployed app and gave feedback.

## Known limits

- It only knows what is in its documents: no timetables, conferences, or private spaces (ENT, Moodle).
- For urgent or personal administrative questions, contact the school office.
- It uses a free AI model, so it can be slow, or show a "too many requests" error. Wait a moment and retry.

## Ideas for the future

- Adapt it to other university programs.
- Make it faster (caching, parallel search).
- Add timetables and events.
- Handle more complex data.

---

## Author

**Aya El Majjouti**, Licence 3 MIAGE, Université Toulouse Capitole
Supervisor: **Chihab Hanachi**

Thanks to M. Chihab Hanachi, M. Alain Berro and Mme Leila Moudjari.
