// One place for every backend call.
// All functions throw Error(message) with a user-friendly message.
const BASE = "/api";

async function request(path, options = {}, timeoutMs = 120000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(BASE + path, { ...options, signal: ctrl.signal });
    const data = await res.json().catch(() => null);
    if (!res.ok) throw new Error(data?.detail || `Request failed (${res.status}).`);
    return data;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("The server took too long to respond. Please try again.");
    if (e instanceof TypeError) throw new Error("Cannot reach the server. Check your connection and try again.");
    throw e;
  } finally { clearTimeout(timer); }
}

const json = (body, method = "POST") => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
});

export const api = {
  health:         ()     => request("/health"),
  metrics:        ()     => request("/metrics"),
  stats:          ()     => request("/stats"),
  similar:        (id)   => request(`/similar?id=${encodeURIComponent(id)}`),
  samples:        ()     => request("/samples"),
  sample:         (name) => request(`/samples/${encodeURIComponent(name)}`),
  extract:        (file) => { const f = new FormData(); f.append("file", file); return request("/extract", { method: "POST", body: f }); },
  analyze:        (text) => request("/analyze", json({ text }), 60000),
  analyzeBatch:   (texts)=> request("/analyze/batch", json({ texts }), 180000),
  history:        (status) => request(`/history?limit=200${status ? `&status=${encodeURIComponent(status)}` : ""}`),
  updateStatus:   (id, status) => request(`/history/${id}`, json({ status }, "PATCH")),
  updateStatuses: (ids, status) => request("/history", json({ ids, status }, "PATCH")),
  deleteOne:      (id)   => request(`/history/${id}`, { method: "DELETE" }),
  clearHistory:   ()     => request("/history", { method: "DELETE" }),
  steps:          (text) => request(`/steps?text=${encodeURIComponent(text)}`),
  source:         (name) => request(`/source/${name}`),
  sentimentCheck: (n=20) => request(`/sentiment-check?n=${n}`),
  promptCompare:  ()     => request("/prompt-compare", { method: "POST" }),
};
