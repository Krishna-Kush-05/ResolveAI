import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { Alert, Button, Card, EmptyState, Priority, Spinner, StatusBadge } from "../components.jsx";

export default function History() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("all");
  const load = useCallback(() =>
    api.history().then((r) => { setRows(r); setError(null); }).catch((e) => { setRows([]); setError(e.message); })
  , []);
  useEffect(() => { load(); }, [load]);

  const remove = async (id) => {
    try { await api.deleteOne(id); setRows((r) => r.filter((x) => x.id !== id)); }
    catch (e) { setError(e.message); }
  };
  const changeStatus = async (id, status) => {
    const before = rows;
    setRows(rows.map((r) => r.id === id ? { ...r, status } : r));
    try { await api.updateStatus(id, status); }
    catch (e) { setRows(before); setError(e.message); }
  };
  const clearAll = async () => {
    if (!window.confirm("Delete all saved grievances?")) return;
    try { await api.clearHistory(); setRows([]); } catch (e) { setError(e.message); }
  };

  return (
    <div className="stack">
      <div><h1>History</h1><p className="muted">Every grievance you analyse is saved here.</p></div>
      {error && <Alert onClose={() => setError(null)}>{error}</Alert>}
      <Card title={rows ? `${rows.filter((r) => r.status === "pending").length} pending · ${rows.filter((r) => r.status === "resolved").length} resolved` : "Saved grievances"}
            actions={rows?.length ? <Button variant="ghost" onClick={clearAll}>Clear all</Button> : null}>
        {rows === null ? <div className="center"><Spinner /></div> : rows.length === 0 ? (
          <EmptyState title="Nothing saved yet" text="Analysed grievances will appear here." />
        ) : (
          <>
          <div className="status-filters" role="group" aria-label="Filter history by status">{["all", "pending", "resolved"].map((value) => <Button key={value} variant={filter === value ? "primary" : "secondary"} onClick={() => setFilter(value)}>{value[0].toUpperCase() + value.slice(1)}</Button>)}</div>
          <div className="table-wrap"><table>
            <thead><tr><th>When</th><th>Grievance</th><th>Category</th><th>Priority</th><th>Sentiment</th><th>Status</th><th></th></tr></thead>
            <tbody>{rows.filter((r) => filter === "all" || r.status === filter).map((r) => (
              <tr key={r.id}>
                <td className="nowrap">{new Date(r.created_at).toLocaleString()}</td>
                <td className="cell-text" title={r.text}>{r.text}</td>
                <td>{r.category}</td>
                <td><Priority value={r.priority} /></td>
                <td>{r.sentiment}</td>
                <td><div className="status-cell"><StatusBadge value={r.status} /><Button variant="secondary" onClick={() => changeStatus(r.id, r.status === "resolved" ? "pending" : "resolved")}>{r.status === "resolved" ? "Reopen" : "Mark resolved"}</Button></div></td>
                <td><Button variant="ghost" onClick={() => remove(r.id)} aria-label={`Delete record ${r.id}`}>Delete</Button></td>
              </tr>
            ))}</tbody>
          </table></div>
          </>
        )}
      </Card>
    </div>
  );
}
