import { useState, useEffect } from "react";
import Analyze from "./pages/Analyze.jsx";
import Walkthrough from "./pages/Walkthrough.jsx";
import History from "./pages/History.jsx";
import Dashboard from "./pages/Dashboard.jsx";

/* Hash-based page switch: #/, #/walkthrough, #/history.
   No router library needed. Refresh and back button work. */
function useHash() {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const onHash = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return hash;
}

const NAV = [
  { hash: "#/", label: "Analyze" },
  { hash: "#/dashboard", label: "Dashboard" },
  { hash: "#/walkthrough", label: "Walkthrough" },
  { hash: "#/history", label: "History" },
];

export default function App() {
  const hash = useHash();
  const page = hash.replace("#", "") || "/";
  const [dark, setDark] = useState(() => {
    try { return localStorage.getItem("resolveai-theme") === "dark"; }
    catch { return false; }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    try { localStorage.setItem("resolveai-theme", dark ? "dark" : "light"); } catch {}
  }, [dark]);

  return (
    <div className="shell">
      <header className="nav">
        <div className="container nav-inner">
          <a href="#/" className="brand" aria-label="ResolveAI home">
            <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
              <rect width="32" height="32" rx="8" fill="#4F46E5" />
              <path d="M8 21V12l8-4 8 4v9M12 21v-6m4 6v-6m4 6v-6" stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span>ResolveAI</span>
          </a>
          <nav className="nav-links">
            {NAV.map((n) => (
              <a key={n.hash} href={n.hash} className={hash === n.hash || (n.hash === "#/" && (hash === "" || hash === "#")) ? "active" : ""}>
                {n.label}
              </a>
            ))}
          </nav>
          <button className="theme-toggle" type="button" onClick={() => setDark((value) => !value)} aria-label={dark ? "Use light mode" : "Use dark mode"}>{dark ? "☀ Light" : "☾ Dark"}</button>
        </div>
      </header>
      <main className="container main">
        {page === "/" || page === "" ? <Analyze />
         : page === "/walkthrough" ? <Walkthrough />
         : page === "/dashboard" ? <Dashboard />
         : page === "/history" ? <History />
         : <Analyze />}
      </main>
      <footer className="footer container">ResolveAI &middot; Student grievance analysis and resolution tracking</footer>
    </div>
  );
}
