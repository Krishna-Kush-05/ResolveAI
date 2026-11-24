// Reusable UI building blocks — used by every page for consistent look.
import { useState, useEffect, useRef } from "react";
import { api } from "./api.js";

/* ---- Generic components ---- */
export const Card = ({ title, subtitle, actions, children, className = "" }) => (
  <section className={`card ${className}`}>
    {(title || actions) && <div className="card-head"><div>{title && <h2>{title}</h2>}{subtitle && <p className="muted">{subtitle}</p>}</div>{actions}</div>}
    {children}
  </section>
);

export const Button = ({ variant = "primary", loading, children, ...p }) => (
  <button className={`btn btn-${variant}`} disabled={loading || p.disabled} {...p}>
    {loading && <Spinner small />}{children}
  </button>
);

export const Spinner = ({ small, label }) => (
  <span className={`spinner ${small ? "small" : ""}`} role="status" aria-label={label || "Loading"} />
);

export const Alert = ({ kind = "error", children, onClose }) => (
  <div className={`alert alert-${kind}`} role={kind === "error" ? "alert" : "status"}>
    <span>{children}</span>
    {onClose && <button className="alert-x" onClick={onClose} aria-label="Dismiss">&times;</button>}
  </div>
);

export const EmptyState = ({ title, text }) => (
  <div className="empty"><div className="empty-icon" aria-hidden="true">&#9676;</div><h3>{title}</h3><p className="muted">{text}</p></div>
);

export const Priority = ({ value }) => (
  <span className={`badge p-${String(value).toLowerCase()}`}>{value}</span>
);

export const StatusBadge = ({ value = "pending" }) => (
  <span className={`status-badge status-${value}`}>{value === "resolved" ? "Resolved" : "Pending"}</span>
);

function HighlightTerms({ text, terms }) {
  const patterns = [...new Set((terms || []).map((t) => t.term).filter(Boolean))]
    .sort((a, b) => b.length - a.length)
    .map((term) => term.split(/\s+/).map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?:ing|ed|s)?").join("\\s+"));
  if (!patterns.length) return <p className="highlight">{text}</p>;
  const regex = new RegExp(`(${patterns.join("|")})`, "gi");
  return <p className="highlight contribution-text">{text.split(regex).map((part, i) => { regex.lastIndex = 0; return regex.test(part) ? <mark key={i} className="contribution-highlight">{part}</mark> : part; })}</p>;
}

function TermWeights({ title, terms = [] }) {
  const max = Math.max(0.0001, ...terms.map((t) => Math.abs(t.weight)));
  return <div className="term-group"><h4>{title}</h4>{terms.length ? terms.map((item, i) => (
    <div className="term-weight" key={`${item.term}-${i}`}><span className="term-chip">{item.term}</span><div className="term-track"><span style={{ width: `${Math.abs(item.weight) / max * 100}%` }} /></div><code>{item.weight > 0 ? "+" : ""}{item.weight.toFixed(3)}</code></div>
  )) : <p className="muted small">No vocabulary terms contributed for this text.</p>}</div>;
}

function SimilarContent({ matches = [] }) {
  return <>
    {matches.some((m) => m.similarity >= 0.7) && <Alert kind="info">Possible duplicate: a past grievance is highly similar.</Alert>}
    {matches.length ? <div className="similar-list">{matches.map((m) => <div className="similar-item" key={m.id}>
      <div><StatusBadge value={m.status} /><span className="similar-score">{Math.round(m.similarity * 100)}% similar</span></div>
      <p>{m.english_text}</p><span className="muted small">{m.category} · #{m.id}</span>
    </div>)}</div> : <p className="muted small">No close matches found in saved grievances.</p>}
  </>;
}

export const Stat = ({ label, value, note }) => (
  <div className="stat"><div className="stat-label">{label}</div><div className="stat-value">{value}</div>{note && <div className="stat-note">{note}</div>}</div>
);

export const LANG = { en: "English", hi: "Hindi", mr: "Marathi", ne: "Nepali" };
export const PRIORITY_ORDER = { Critical: 0, High: 1, Medium: 2, Low: 3 };

/* ---- Entity-highlighted text ---- */
export function Highlight({ text, entities = [] }) {
  const parts = [];
  let pos = 0;
  [...entities].sort((a, b) => a.start - b.start).forEach((e, i) => {
    if (e.start < pos) return;
    parts.push(text.slice(pos, e.start));
    parts.push(<mark key={i}>{e.text}<small>{e.label}</small></mark>);
    pos = e.end;
  });
  parts.push(text.slice(pos));
  return <p className="highlight">{parts}</p>;
}

/* ---- Input panel: text box + upload + sample dropdown ---- */
const MAX_CHARS = 2000, MAX_BYTES = 2 * 1024 * 1024, OK_EXT = ["txt", "pdf", "csv"];

export function InputPanel({ text, setText, source, setSource, onAnalyze, busy }) {
  const [samples, setSamples] = useState([]);
  const [msg, setMsg] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef(null);

  useEffect(() => { api.samples().then(setSamples).catch(() => setSamples([])); }, []);

  const apply = (d) => {
    if (d.kind === "table") {
      setSource(d); setText("");
      setMsg(d.truncated ? { kind: "info", text: `Only the first ${d.texts.length} of ${d.total_rows} rows will be analysed.` } : null);
    } else {
      setSource(null); setText(d.text);
      setMsg({ kind: "info", text: `Text extracted from ${d.name}. You can edit it before analysing.` });
    }
  };
  const run = async (fn) => {
    setLoading(true); setMsg(null);
    try { apply(await fn()); } catch (e) { setMsg({ kind: "error", text: e.message }); }
    finally { setLoading(false); }
  };
  const onFile = (e) => {
    const f = e.target.files[0]; e.target.value = ""; if (!f) return;
    const ext = f.name.split(".").pop().toLowerCase();
    if (!OK_EXT.includes(ext)) return setMsg({ kind: "error", text: "Unsupported file type. Use TXT, PDF or CSV." });
    if (f.size > MAX_BYTES) return setMsg({ kind: "error", text: "File is too large (max 2 MB)." });
    run(() => api.extract(f));
  };
  const clear = () => { setText(""); setSource(null); setMsg(null); };
  const tooShort = !source && text.trim().length < 5;

  return (
    <Card title="Grievance" subtitle="Type or paste a grievance (English, Hindi or Marathi), upload a TXT / PDF / CSV file, or load a sample.">
      {source ? (
        <div className="file-chip"><div><strong>{source.name}</strong><span className="muted"> &middot; {source.texts.length} grievances ready</span></div><Button variant="ghost" onClick={clear}>Remove</Button></div>
      ) : (
        <>
          <label className="sr" htmlFor="g-text">Grievance text</label>
          <textarea id="g-text" value={text} maxLength={MAX_CHARS} rows={5} placeholder="e.g. There has been no hot water in Cedar Hall for three days." onChange={(e) => setText(e.target.value)} aria-label="Grievance text" />
          <div className="input-meta"><span className={text.trim().length > 0 && text.trim().length < 5 ? "input-hint" : "muted small"}>{text.trim().length === 0 ? "Enter at least 5 characters to analyze." : text.trim().length < 5 ? "This grievance is too short. Add a few more details." : ""}</span><span className="muted small">{text.length} / {MAX_CHARS}</span></div>
        </>
      )}
      <div className="row">
        <input ref={fileRef} type="file" accept=".txt,.pdf,.csv" hidden onChange={onFile} />
        <Button variant="secondary" loading={loading} onClick={() => fileRef.current.click()}>Upload file</Button>
        <select aria-label="Load a sample" value="" disabled={loading || !samples.length} onChange={(e) => e.target.value && run(() => api.sample(e.target.value))}>
          <option value="">{samples.length ? "Load a sample..." : "Samples unavailable"}</option>
          {samples.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
        </select>
        {(text || source) && <Button variant="ghost" onClick={clear}>Clear</Button>}
        <span className="spacer" />
        <Button loading={busy} disabled={tooShort || loading} onClick={() => onAnalyze()}>
          {source ? `Analyze ${source.texts.length} grievances` : "Analyze"}
        </Button>
      </div>
      {msg && <Alert kind={msg.kind} onClose={() => setMsg(null)}>{msg.text}</Alert>}
    </Card>
  );
}

/* ---- Single result card ---- */
export function ResultCard({ r }) {
  const [copied, setCopied] = useState(false);
  const [status, setStatus] = useState(r.status || "pending");
  const [statusError, setStatusError] = useState("");
  const rep = r.reply;
  const copy = async () => {
    try { await navigator.clipboard.writeText(rep.reply); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch {}
  };
  const toggleStatus = async () => {
    const previous = status, next = status === "resolved" ? "pending" : "resolved";
    setStatus(next); setStatusError("");
    try { await api.updateStatus(r.id, next); }
    catch (e) { setStatus(previous); setStatusError(e.message); }
  };
  return (
    <div className="stack">
      {r.warning && <Alert kind="error">{r.warning}</Alert>}
      {r.needs_review && <div className="review-row"><span className="review-badge">Needs human review</span><div className="review-categories">{r.top_categories.map((item) => <span className="review-chip" key={item.category}>{item.category} · {Math.round(item.probability * 100)}%</span>)}</div></div>}
      {r.priority_source === "rule" && <p className="rule-note">Escalated by safety rule: {r.matched_rules.map((m) => m.term).join(", ")}</p>}
      <div className="status-row"><StatusBadge value={status} /><Button variant="secondary" onClick={toggleStatus}>{status === "resolved" ? "Reopen" : "Mark resolved"}</Button></div>
      {statusError && <Alert>{statusError}</Alert>}
      <div className="grid-4">
        <Stat label="Category" value={r.category} note={`confidence ${Math.round(r.confidence * 100)}%`} />
        <Stat label="Priority" value={<Priority value={r.priority} />} />
        <Stat label="Sentiment" value={r.sentiment} note={`score ${r.sentiment_score > 0 ? "+" : ""}${r.sentiment_score}`} />
        <Stat label="Language" value={LANG[r.language] || r.language.toUpperCase()} note={r.translated_by ? `translated by ${r.translated_by}` : "no translation needed"} />
      </div>
      <details className="card explanation-card"><summary>Why this result?</summary><div className="explanation-content">
        <div className="grid-2"><TermWeights title="Category terms" terms={r.explanation?.category_terms} /><TermWeights title="Priority terms" terms={r.explanation?.priority_terms} /></div>
        <div><h4>Contributing words in the English text</h4><HighlightTerms text={r.english_text} terms={[...(r.explanation?.category_terms || []), ...(r.explanation?.priority_terms || [])]} /></div>
      </div></details>
      <Card title="Similar grievances already reported"><SimilarContent matches={r.similar} /></Card>
      <div className="grid-2">
        <Card title="English text with entities" subtitle="Named entities found by our rule-based NER are highlighted.">
          <Highlight text={r.english_text} entities={r.entities} />
          {r.language !== "en" && <p className="muted small">Original: {r.text}</p>}
          {r.entities.length === 0 && <p className="muted small">No specific entities (courses, buildings, times, days) were found.</p>}
        </Card>
        <Card title="Suggested response" actions={<Button variant="ghost" onClick={copy}>{copied ? "Copied" : "Copy reply"}</Button>}>
          <dl className="facts">
            <dt>Summary</dt><dd>{rep.summary}</dd>
            <dt>Handled by</dt><dd>{rep.department}</dd>
            <dt>Next step</dt><dd>{rep.action}</dd>
          </dl>
          <blockquote className="reply">{rep.reply}</blockquote>
          <p className="muted small">Source: {rep.source}{rep.note ? ` · ${rep.note}` : ""}. Replies are drafts for human review.</p>
        </Card>
      </div>
    </div>
  );
}

/* ---- Batch result table ---- */
export function BatchResults({ data }) {
  const [rows, setRows] = useState(data.results.map((r) => ({ ...r, status: r.status || "pending" })));
  const [selected, setSelected] = useState([]);
  const [statusError, setStatusError] = useState("");
  const orderedRows = [...rows].sort((a, b) => PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority]);
  const top = Object.entries(data.by_category).sort((a, b) => b[1] - a[1])[0];
  const neg = data.results.filter((r) => r.sentiment === "Negative").length;
  const updateSelected = async (ids, status) => {
    const before = rows;
    setRows(rows.map((r) => ids.includes(r.id) ? { ...r, status } : r)); setStatusError("");
    try { await api.updateStatuses(ids, status); }
    catch (e) { setRows(before); setStatusError(e.message); }
  };
  return (
    <div className="stack">
      <div className="grid-4">
        <Stat label="Grievances" value={data.count} />
        <Stat label="Critical + High" value={(data.by_priority.Critical || 0) + (data.by_priority.High || 0)} note="need attention first" />
        <Stat label="Top category" value={top[0]} note={`${top[1]} grievances`} />
        <Stat label="Negative tone" value={`${Math.round((neg / data.count) * 100)}%`} />
      </div>
      <Card title="Results" subtitle="Most urgent first.">
        <div className="row status-actions">
          <label><input type="checkbox" checked={selected.length === rows.length && rows.length > 0} onChange={(e) => setSelected(e.target.checked ? rows.map((r) => r.id) : [])} /> Select all</label>
          <Button variant="secondary" disabled={!selected.length} onClick={() => updateSelected(selected, "resolved")}>Mark selected resolved</Button>
          <Button variant="secondary" disabled={!selected.length} onClick={() => updateSelected(selected, "pending")}>Mark selected pending</Button>
          <Button variant="ghost" disabled={!rows.length} onClick={() => updateSelected(rows.map((r) => r.id), "resolved")}>Mark all resolved</Button>
        </div>
        {statusError && <Alert>{statusError}</Alert>}
        <div className="table-wrap"><table>
          <thead><tr><th>Select</th><th>Grievance</th><th>Category</th><th>Priority</th><th>Sentiment</th><th>Status</th><th>Similar</th></tr></thead>
          <tbody>{orderedRows.map((r) => (
            <tr key={r.id}><td><input type="checkbox" checked={selected.includes(r.id)} onChange={(e) => setSelected(e.target.checked ? [...selected, r.id] : selected.filter((id) => id !== r.id))} aria-label={`Select grievance ${r.id}`} /></td><td className="cell-text" title={r.text}>{r.text}</td><td>{r.category}</td><td><Priority value={r.priority} /></td><td>{r.sentiment}</td><td><StatusBadge value={r.status} /></td><td><details><summary>{r.similar?.length || 0} matches</summary><SimilarContent matches={r.similar} /></details></td></tr>
          ))}</tbody>
        </table></div>
      </Card>
    </div>
  );
}
