# Self-Improving Agentic RAG — React + FastAPI

A text RAG agent that retrieves, generates, judges its own answer, and self-corrects —
with a React frontend and a FastAPI backend wrapping the LangGraph agent.

```
agentic-rag-app/
├── backend/          FastAPI + the agent (LangGraph, Groq)
│   ├── agent.py      agent logic (graph, nodes, judge, memory)
│   ├── main.py       API: POST /build, POST /ask
│   └── requirements.txt
└── frontend/         React (Vite) single page
    └── src/App.jsx
```

## Run it (two terminals)

### 1. Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"                   # Windows: set GROQ_API_KEY=...
# optional: export TAVILY_API_KEY="..."  (enables web fallback)
uvicorn main:app --reload --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (usually http://localhost:5173).
The frontend proxies `/api/*` to the backend on port 8000 (see `vite.config.js`).

## Use
1. Upload one or more PDFs → **Build index**
2. Type a question → **Ask**
3. The answer shows with its faithfulness **score**, an expandable **decision trace**
   (where you can watch the agent self-correct), and the retrieved context.

## Notes
- Generation runs on the **Groq API** — no local GPU needed.
- Chroma is **in-memory** and lessons are in SQLite at the system temp path;
  both are session-scoped and reset on restart.
