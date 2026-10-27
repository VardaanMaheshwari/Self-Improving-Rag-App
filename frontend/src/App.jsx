import { useState } from "react";

// score badge colored by threshold
function Score({ value }) {
  const good = value >= 0.7;
  return (
    <span className={`score ${good ? "good" : "warn"}`}>
      <span className="dot" /> score {value.toFixed(2)}
    </span>
  );
}

// collapsible section
function Expander({ title, open = false, children }) {
  const [show, setShow] = useState(open);
  return (
    <div className="exp">
      <button className="exp-head" onClick={() => setShow((s) => !s)}>
        <span className="caret">{show ? "▾" : "▸"}</span> {title}
      </button>
      {show && <div className="exp-body">{children}</div>}
    </div>
  );
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState(null);   // {chunks, sources}
  const [building, setBuilding] = useState(false);

  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [result, setResult] = useState(null);   // {answer, score, trace, context}
  const [error, setError] = useState(null);

  async function buildIndex() {
    if (!files.length) return;
    setBuilding(true); setError(null); setStatus(null); setResult(null);
    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const res = await fetch("/api/build", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail || "Build failed");
      setStatus(await res.json());
    } catch (e) { setError(e.message); }
    finally { setBuilding(false); }
  }

  async function ask() {
    if (!question.trim() || !status) return;
    setAsking(true); setError(null);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Ask failed");
      setResult(await res.json());
    } catch (e) { setError(e.message); }
    finally { setAsking(false); }
  }

  return (
    <div className="wrap">
      <header>
        <h1>Self-Improving Agentic RAG</h1>
        <div className="flow">contextualize → triage → retrieve → generate → judge → self-correct</div>
      </header>

      {/* 1 Documents */}
      <section className="card">
        <p className="step">1 · Documents</p>
        <label className="drop">
          Drop PDFs here, or <b>click to upload</b> · one or many
          <input type="file" accept=".pdf" multiple hidden
            onChange={(e) => setFiles([...e.target.files])} />
        </label>
        <div className="filerow">
          {files.map((f) => (
            <span className="chip" key={f.name}><span className="tick">✓</span> {f.name}</span>
          ))}
          <span className="spacer" />
          <button onClick={buildIndex} disabled={!files.length || building}>
            {building ? "Building…" : "Build index"}
          </button>
        </div>
        {status && (
          <div className="status">
            Indexed {status.chunks} chunks from {status.sources.length} PDF
            {status.sources.length > 1 ? "s" : ""}
          </div>
        )}
      </section>

      {/* 2 Ask */}
      <section className="card">
        <p className="step">2 · Ask</p>
        <div className="askrow">
          <input
            placeholder="What is the main contribution?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
          />
          <button onClick={ask} disabled={!status || !question.trim() || asking}>
            {asking ? "Running…" : "Ask"}
          </button>
        </div>
        {error && <div className="err">{error}</div>}
      </section>

      {/* 3 Result */}
      {result && (
        <section className="card">
          <div className="ansrow">
            <h2>Answer</h2>
            <Score value={result.score} />
          </div>
          <p className="answer">{result.answer}</p>

          <Expander title="Decision trace" open>
            <ul className="trace">
              {result.trace.map(([node, detail], i) => (
                <li key={i} className={/reflect|rewrite|widen|regenerate/.test(node) ? "loop" : ""}>
                  <span className="node">{node}</span>
                  <span className="detail">{detail}</span>
                </li>
              ))}
            </ul>
          </Expander>

          <Expander title="Retrieved context">
            <ul className="ctx">
              {result.context.map((c, i) => (
                <li key={i}>{c.slice(0, 500)}…</li>
              ))}
            </ul>
          </Expander>
        </section>
      )}

      <footer>The decision trace shows the agent correcting itself — the core of how it works.</footer>
    </div>
  );
}
