import { useEffect, useState } from "react";
import { api } from "../api.js";
import { Alert, Card, EmptyState, Spinner, Stat } from "../components.jsx";

function Bars({ title, values }) {
  const entries = Object.entries(values || {}).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, count]) => count));
  return <Card title={title}><div className="stats-bars">{entries.map(([name, count]) => (
    <div className="stats-bar-row" key={name}><span>{name}</span><div className="stats-bar-track"><div style={{ width: `${count / max * 100}%` }} /></div><b>{count}</b></div>
  ))}</div></Card>;
}

function Trend({ days }) {
  const width = 720, height = 230, left = 28, right = 16, top = 16, bottom = 32;
  const max = Math.max(1, ...days.map((d) => d.count));
  const points = days.map((d, i) => {
    const x = left + i * (width - left - right) / Math.max(1, days.length - 1);
    const y = height - bottom - d.count / max * (height - top - bottom);
    return `${x},${y}`;
  }).join(" ");
  return <Card title="Grievances received · last 14 days"><div className="trend-wrap"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Daily grievance counts for the last 14 days">
    {[0, 1, 2, 3].map((i) => <line key={i} x1={left} x2={width - right} y1={top + i * (height - top - bottom) / 3} y2={top + i * (height - top - bottom) / 3} className="chart-grid" />)}
    <polyline points={points} className="trend-line" />
    {days.map((d, i) => {
      const x = left + i * (width - left - right) / Math.max(1, days.length - 1);
      const y = height - bottom - d.count / max * (height - top - bottom);
      return <g key={d.date}><circle cx={x} cy={y} r="3.5" className="trend-dot"><title>{d.date}: {d.count}</title></circle>{i % 2 === 0 && <text x={x} y={height - 8} textAnchor="middle" className="chart-label">{d.date.slice(5)}</text>}</g>;
    })}
  </svg></div></Card>;
}

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { api.stats().then(setData).catch((e) => setError(e.message)); }, []);
  if (error) return <Alert>{error}</Alert>;
  if (!data) return <div className="center"><Spinner label="Loading dashboard" /></div>;
  if (!data.total) return <div className="stack"><div><h1>Dashboard</h1><p className="muted">A quick view of grievance activity and resolution progress.</p></div><Card><EmptyState title="No grievances yet" text="Analysed grievances will appear here." /></Card></div>;
  const resolutionRate = Math.round(data.resolved / data.total * 100);
  const pendingWidth = data.total ? data.pending / data.total * 100 : 0;
  return <div className="stack">
    <div><h1>Dashboard</h1><p className="muted">A quick view of grievance activity and resolution progress.</p></div>
    <div className="grid-4"><Stat label="Total" value={data.total} /><Stat label="Pending" value={data.pending} /><Stat label="Resolved" value={data.resolved} /><Stat label="Resolution rate" value={`${resolutionRate}%`} /></div>
    <div className="grid-2"><Bars title="By category" values={data.by_category} /><Bars title="By priority" values={data.by_priority} /></div>
    <Trend days={data.per_day} />
    <Card title="Status split"><div className="status-split" aria-label={`${data.pending} pending, ${data.resolved} resolved`}><span style={{ width: `${pendingWidth}%` }} /><span style={{ width: `${100 - pendingWidth}%` }} /></div><div className="split-legend"><span>Pending · {data.pending}</span><span>Resolved · {data.resolved}</span></div><p className="muted small">Average sentiment score: {data.average_sentiment_score}</p></Card>
  </div>;
}
