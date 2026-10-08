#  2IS Master's Program Assistant (RAG Chatbot)

A conversational assistant for the **Master "Innovative Information Systems" (2IS)** at **Université Toulouse Capitole**. It answers questions about admissions, courses, teachers, ECTS, prerequisites and career prospects, using only official program documents.

> Internship project (Licence 3 MIAGE, 2024-2025) carried out at **IRIT**, supervised by Prof. Chihab Hanachi.

👉 **Live demo:** https://rag-chatbot-2is-qcthvwdzoerdnj4xappyfjd.streamlit.app/

<!-- TODO: add a screenshot of the interface, e.g. ![Interface](docs/interface.png) -->

---

## Features

- **Hybrid RAG**: semantic search (FAISS) for open questions + deterministic Python filtering for exhaustive list questions ("list all courses of semester 1").
- **Adaptive routing** with LangGraph. Each question is classified and sent down the right path:
  1. Simple conversation (greetings, thanks)
  2. Polite refusal for off-topic questions
  3. Specialized list handling (`find_and_list_courses`)
  4. Standard RAG pipeline
- **Conversation memory**: follow-up questions ("And who teaches it?") are rewritten into standalone questions.
- **Anti-hallucination**: strict prompting ("answer ONLY with the provided information") and cited sources under every answer.
- **Automatic evaluation** (LLM-as-a-Judge): classification accuracy, response time, faithfulness, relevancy, reference-based score.
- **Auto-updated knowledge base** via a scheduled scraping task.

---

##  Architecture

### Phase 1: Building the knowledge base
1. **Collection**: website scraping (up to 5 levels deep), PDF documents (brochures, booklets), and a structured `syllabus.json`.
2. **Processing**: long texts are cleaned and split into chunks; each syllabus course stays a complete entity (never fragmented).
3. **Embedding**: chunks are converted to 384-dimensional vectors with `sentence-transformers/all-MiniLM-L6-v2`.
4. **Indexing**: vectors are stored in a **FAISS** index for fast similarity search.

### Phase 2: Answering a question
1. **Analyze query**: classification, question rewriting, list detection.
2. **Retrieve**: FAISS semantic search, or exhaustive syllabus filtering for list requests.
3. **Generate**: the LLM writes a natural answer from the retrieved context only.
4. **Display**: answer plus sources in the Streamlit interface.

<!-- TODO: add the architecture diagram (Figure 4.4 of the report), e.g. ![Architecture](docs/architecture.png) -->

---

##  Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | DeepSeek-R1 (`deepseek/deepseek-r1-0528:free`) via API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Interface | Streamlit |
| Deployment | GitHub + Streamlit Community Cloud |
| Language | Python |

---

##  Getting Started

### Prerequisites
- Python 3.10+
- An API key for the LLM provider <!-- TODO: name the provider you use (e.g. OpenRouter) -->

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### Configuration

Create a `.env` file (or Streamlit secrets) with your API key:

```
API_KEY=your_key_here
```

> ⚠️ Never commit your API key. Add `.env` to your `.gitignore`.

### Build the knowledge base

```bash
python build_index.py
```

This scrapes the website, processes the PDFs and `syllabus.json`, creates the chunks and builds the FAISS index.

### Run the app

```bash
streamlit run app.py
```

<!-- TODO: replace app.py with the real name of your Streamlit file -->

---

##  Project Structure

<!-- TODO: adjust to match your real repository -->
```
.
├── app.py              # Streamlit interface + LangGraph agent
├── build_index.py      # Scraping, chunking, embeddings, FAISS index
├── syllabus.json       # Structured course data
├── requirements.txt
└── README.md
```

---

## 🔄 Updating the Data

The knowledge base is rebuilt automatically every 4 months (January, April, July, October) by a scheduled task running `build_index.py`. To update manually:

1. Add new PDFs and/or edit `syllabus.json`
2. Run `python build_index.py`
3. Push to GitHub, which triggers an automatic redeploy on Streamlit

---

##  Evaluation

- **Automated tests** on 5 question types: course questions, list questions, greetings, off-topic questions, and trap questions (no answer in the documents).
- **User validation** with fellow students through the deployed app.

---

## ⚠️ Limitations

- Only answers from information present in its knowledge base (no timetables or conferences yet).
- Cannot access private student spaces (ENT, Moodle).
- Free-tier LLM: slow responses or "too many requests" errors are possible.

##  Future Work

- Extend to other university programs (modular architecture)
- Performance optimization (caching, parallel vector search)
- Conversation history, timetables, events
- Handling of more complex data

---

## 👤 Author

**Aya El Majjouti**, Licence 3 MIAGE, Université Toulouse Capitole
Supervisor: **Chihab Hanachi**

## 🙏 Acknowledgements

Thanks to M. Chihab Hanachi, M. Alain Berro and Mme Leila Moudjari.
