"""
FastAPI server — exposes the agent over HTTP for the React frontend.
Endpoints:  POST /build (upload PDFs)   POST /ask (run the agent)
"""
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent

app = FastAPI(title="Agentic RAG API")

# allow the React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# single in-memory session (one index + history). Fine for a demo / single user.
STATE = {"retriever": None, "history": []}


class AskRequest(BaseModel):
    question: str


@app.post("/build")
async def build(files: list[UploadFile] = File(...)):
    paths, names = [], []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(await f.read())
        tmp.close()
        paths.append(tmp.name)
        names.append(f.filename)
    if not paths:
        raise HTTPException(400, "No PDF files provided.")
    STATE["retriever"] = agent.build_retriever(paths, names)
    STATE["history"] = []
    n_chunks = len(STATE["retriever"]["chunks"])
    return {"chunks": n_chunks, "sources": sorted(set(names))}


@app.post("/ask")
def ask(req: AskRequest):
    if STATE["retriever"] is None:
        raise HTTPException(400, "No index yet. Upload PDFs and build first.")
    result = agent.run_agent(req.question, STATE["retriever"], STATE["history"])
    STATE["history"].append({"q": req.question, "a": result["answer"]})
    return result


@app.get("/health")
def health():
    return {"ok": True, "indexed": STATE["retriever"] is not None}
