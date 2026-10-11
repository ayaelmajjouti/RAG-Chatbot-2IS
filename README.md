# 🎓 2IS Master's Program Assistant

A chatbot that answers students' questions about the **Master "Innovative Information Systems" (2IS)** at **Université Toulouse Capitole**: admissions, courses, teachers, ECTS, prerequisites, career prospects.

It answers **only from official program documents**, and shows its sources under every answer.

👉 **Try it:** <https://rag-chatbot-2is-qcthvwdzoerdnj4xappyfjd.streamlit.app/>
> Internship project (Licence 3 MIAGE, 2024-2025) at **IRIT**, supervised by Prof. Chihab Hanachi.

<img width="1436" height="883" alt="image" src="https://github.com/user-attachments/assets/fae95a3d-6852-4f54-a70f-d86adf36e86e" />

---

## Why this project?

Every year, the 2IS master receives many international students who ask the same questions again and again. This assistant answers them instantly, any time, in plain language.

## What it can do

- **Answer questions about the program**, using the official website page, two PDF brochures (flyer and student welcome booklet) and the syllabus (`Syllabus.json`, 28 courses).
- **List things exactly**: for questions like "list all courses of semester 1", it filters the syllabus with code instead of relying on search, so nothing is forgotten.
- **Use the conversation**: the chat history is sent to the model, and on the course-list path the question is rewritten as a standalone question.
- **Stay on topic**: greetings get a friendly reply, and off-topic questions get a polite refusal.
- **Limit made-up answers**: the model is told to answer only from the retrieved information, to say so when it cannot find the answer, and the sources are shown.

## How it works

The idea is called **RAG** (Retrieval-Augmented Generation): instead of letting the AI answer from memory, we first **find** the relevant passages, then ask the AI to **write the answer from them only**.

**Step 1: Build the knowledge base** (`build_index.py`)

1. Collect the website page, the PDFs and the syllabus (`Syllabus.json`). The syllabus PDF is skipped on purpose, because the same content is available in cleaner, structured form in the JSON.
2. Clean the text and cut long texts into chunks (1500 characters, 200 overlap). Each course of the syllabus is kept whole as one chunk.
3. Turn each chunk into numbers (an *embedding*) with `all-MiniLM-L6-v2` (384 dimensions).
4. Store them in a **FAISS** index (`vector_db/`), with the chunk texts saved alongside.

**Step 2: Answer a question** (`streamlit_app.py` → `rag/graph.py`)

1. **Analyze** the question in a single model call: chat, off-topic, or on-topic; is it a list request; rewrite it as a standalone question.
2. **Route**:
   - chat or off-topic → a short, friendly reply, no search;
   - list request → the syllabus JSON is filtered in Python (period, fields), and the model only formats the result;
   - other on-topic question → the 15 closest chunks are retrieved from FAISS.
3. Ask the model to write the answer using only the retrieved context.
4. Show the answer with its sources.

The flow between these steps is managed with **LangGraph**.

```
analyze_query ─┬─ conversational ──► conversational_response ──► END
               ├─ off_topic ───────► off_topic_response ───────► END
               ├─ on-topic list ───► find_and_list_courses ─┐
               └─ on-topic ────────► retrieve ─┬─ found ─► format_docs ─► generate ─┤
                                               └─ none ──► no_docs_fallback ───────┤
                                                              evaluate_answer ◄─────┘ ─► END
```

`evaluate_answer` is an LLM-as-judge step (faithfulness and relevancy, 1 to 5). The web app shows the answer as soon as it is generated; the scores are used in the command line and in `evaluate.py`.

## Built with

| What                     | Tool                                                                      |
| ------------------------ | ------------------------------------------------------------------------- |
| Flow of the conversation | LangGraph                                                                 |
| AI model                 | A free model via [OpenRouter](https://openrouter.ai) (set in `config.py`) |
| Understanding text       | `sentence-transformers/all-MiniLM-L6-v2`                                  |
| Search                   | FAISS (`IndexFlatL2`, exact search)                                       |
| Website and PDF reading  | BeautifulSoup, PyMuPDF                                                    |
| Text splitting           | LangChain `RecursiveCharacterTextSplitter`                                |
| Chat history             | JSON files in `chat_histories/` (one per session)                         |
| Interface                | Streamlit                                                                 |
| Hosting                  | GitHub + Streamlit Community Cloud                                        |

---

## Run it yourself

### 1. Get the code

```
git clone https://github.com/ayaelmajjouti/RAG-Chatbot-2IS.git
cd RAG-Chatbot-2IS
```

### 2. Install

Python **3.11 or 3.12** is recommended.

```
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
python -m pip install -r requirements.txt
```

### 3. Add your secret

The app needs one value. **Never write it in the code or commit it.**

| Name                 | Where to get it            |
| -------------------- | -------------------------- |
| `OPENROUTER_API_KEY` | openrouter.ai → API Keys   |

**Locally**, put it in a `.env` file (already git-ignored) or set it as an environment variable. For example in PowerShell:

```
$env:OPENROUTER_API_KEY="your_key"
```

**On Streamlit Cloud**, open your app → ⋮ → **Settings → Secrets** and paste:

```
OPENROUTER_API_KEY = "your_key"
```

### 4. Build the knowledge base

```
python build_index.py
```

Use `python build_index.py --force` to overwrite an existing index without being asked.

### 5. Start the app

```
streamlit run streamlit_app.py
```

It opens at <http://localhost:8501>. You can also chat in the terminal with `python main.py`.

---

## Project files

```
.
├── streamlit_app.py       # The web interface
├── main.py                # Starts all components; terminal chat
├── base_rag.py            # Abstract interface for model providers
├── config.py              # Settings (URLs, chunk sizes, model name; key read from environment)
├── build_index.py         # Builds the knowledge base and FAISS index
├── evaluate.py            # Automatic tests of the chatbot
├── Syllabus.json          # Course data (28 courses)
├── scraped_content.json   # Content collected from the website and PDFs
├── rag/
│   ├── graph.py           # LangGraph workflow: routing, retrieval, generation, evaluation
│   ├── openrouter_rag.py  # OpenRouter model handler (prompts, context formatting)
│   └── gemini_rag.py      # Unused alternative handler (Gemini)
├── scraper/
│   ├── website_scraper.py # Reads the web page, PDFs and syllabus
│   └── content_processor.py # Cleaning, chunking, embeddings
├── vector_db/             # FAISS index and chunk store (needed by the app)
└── requirements.txt       # Python packages
```

## Keeping it up to date

The website changes, so the knowledge base should be rebuilt from time to time. Rebuilding is manual by default, and `python build_index.py --force` runs without any prompt, so it can be started by a scheduler (for example cron or a GitHub Action).

To update by hand: edit `Syllabus.json` or add PDF links in `config.py`, run `python build_index.py`, then push to GitHub. Streamlit redeploys automatically.

## How it was tested

- **Automatic tests** (`evaluate.py`): 10 questions covering course questions, lists, greetings, off-topic, and a trick question with no answer in the documents. They measure:
  - classification accuracy (is the question sent to the right path?);
  - response time;
  - faithfulness and relevancy (1 to 5, scored by an LLM judge);
  - similarity to a reference answer (0 to 1, scored by an LLM judge).
- **Real users**: fellow students tried the deployed app and gave feedback.

## Known limits

- It only knows what is in its documents: no timetables, conferences, or private spaces (ENT, Moodle).
- For urgent or personal administrative questions, contact the school office.
- It uses a free AI model, so it can be slow, or show a "too many requests" error. Wait a moment and retry.
- The test set is small and the scores come from an LLM judge, so they are indicative, not definitive.
- The embedding model is mainly English, and long chunks are only partly seen by it.

## Ideas for the future

- Use the rewritten (standalone) question for retrieval too, so follow-up questions search better.
- Combine keyword and semantic search, and re-rank the results.
- Test retrieval itself on a larger set of questions.
- Rebuild the knowledge base automatically on a schedule.
- Adapt it to other university programs and add timetables and events.

---

## Author

**Aya El Majjouti**, Licence 3 MIAGE, Université Toulouse Capitole
Supervisor: **Chihab Hanachi**

Thanks to M. Chihab Hanachi, M. Alain Berro and Mme Leila Moudjari.
