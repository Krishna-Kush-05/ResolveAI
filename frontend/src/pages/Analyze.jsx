import { useEffect, useState } from "react";
import { api } from "../api.js";
import { Alert, Card, EmptyState, Spinner, InputPanel } from "../components.jsx";
import Results from "./Results.jsx";

export default function Analyze() {
  const [text, setText] = useState("");
  const [source, setSource] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [single, setSingle] = useState(null);
  const [batch, setBatch] = useState(null);
  const [llmUnavailable, setLlmUnavailable] = useState(false);
  const [howOpen, setHowOpen] = useState(() => {
    try { return localStorage.getItem("grievance-how-works") !== "closed"; }
    catch { return true; }
  });

  useEffect(() => { api.health().then((h) => setLlmUnavailable(!h.llm_configured)).catch(() => {}); }, []);

  const toggleHow = () => setHowOpen((open) => {
    const next = !open;
    try { localStorage.setItem("grievance-how-works", next ? "open" : "closed"); } catch {}
    return next;
  });

  const analyze = async () => {
    if (!source && text.trim().length < 5) {
      setError(text.trim() ? "Please enter at least 5 characters so we can understand the grievance." : "Enter a grievance or choose a sample or file before analysing.");
      return;
    }
    setBusy(true); setError(null); setSingle(null); setBatch(null);
    try {
      if (source) setBatch(await api.analyzeBatch(source.texts));
      else {
        const result = await api.analyze(text.trim());
        setSingle(result);
        if (result.reply?.source === "offline template" && result.reply?.note?.includes("unavailable")) setLlmUnavailable(true);
      }
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  return (
    <div className="stack">
      <div><h1>Analyze grievances</h1><p className="muted">Classifies a student grievance by category and priority, detects sentiment and language, extracts key entities, and drafts a reply.</p></div>
      {llmUnavailable && <Alert kind="info">AI-drafted replies are currently unavailable. Classification and sentiment still work.</Alert>}
      <Card className="how-card">
        <button className="how-toggle" type="button" onClick={toggleHow} aria-expanded={howOpen}>How it works <span aria-hidden="true">{howOpen ? "−" : "+"}</span></button>
        {howOpen && <div className="how-content">
          <div className="how-steps"><div><b>1</b><span>Paste text, upload a .txt/.pdf/.csv file, or try a sample.</span></div><div><b>2</b><span>Click Analyze.</span></div><div><b>3</b><span>Review the results, then mark each grievance Resolved or Pending in History.</span></div></div>
          <h3>What you get</h3>
          <ul className="how-list"><li>Category, priority, sentiment, detected language, key entities, and a suggested reply.</li><li>Supports TXT, PDF, and CSV batch input (up to 2 MB per file and 100 grievances per batch).</li><li>Non-English input is automatically translated when a translation service is available.</li><li className="muted">Submitted grievances are saved in History on this server; you can delete them anytime.</li></ul>
        </div>}
      </Card>
      <InputPanel text={text} setText={setText} source={source} setSource={setSource} onAnalyze={analyze} busy={busy} />
      {error && <Alert onClose={() => setError(null)}>{error}</Alert>}
      {busy && <Card><div className="center"><Spinner label="Analysing" /><p className="muted">Analysing{source ? ` ${source.texts.length} grievances` : ""}...</p></div></Card>}
      {!busy && (single || batch) && <Results single={single} batch={batch} />}
      {!busy && !single && !batch && !error && <Card><EmptyState title="No results yet" text="Enter a grievance or load a sample, then press Analyze." /></Card>}
      <p className="analyze-footer">AI-assisted suggestions. Review before acting on any grievance. <a href="https://github.com/Krishna-Kush-05/ResolveAI" target="_blank" rel="noopener noreferrer">ResolveAI on GitHub</a></p>
    </div>
  );
}
