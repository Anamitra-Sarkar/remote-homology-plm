import React, { useEffect, useState } from "react";
import { signIn } from "../firebase.js";

const API_BASE = import.meta.env.VITE_API_BASE || "";

export default function SearchPage({ user, authChecked }) {
  const [health, setHealth] = useState(null);
  const [sequence, setSequence] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!API_BASE) return;
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  async function runSearch(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const token = await user.getIdToken();
      const resp = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ sequence: sequence.trim(), top_k: 10 }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data?.detail || "Something went wrong running that search. Please try again.");
      } else {
        setResult(data);
      }
    } catch (err) {
      setError("Something went wrong running that search. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  const modelReady = health?.model_loaded === true;

  return (
    <div className="container search-page">
      <div className="page-header">
        <h1>Search</h1>
        <p>Paste a protein sequence to find its most likely distant relatives.</p>
      </div>

      {health && !modelReady && (
        <div className="notice">
          <strong>Search isn't available right now.</strong> No reviewed reference database is currently
          live — check back shortly.
        </div>
      )}

      {!authChecked ? (
        <p className="muted">Checking sign-in status…</p>
      ) : !user ? (
        <div className="panel">
          <p className="muted">Sign in to run a real search.</p>
          <button className="button button-primary" onClick={() => signIn().catch((err) => setError(String(err)))}>
            Sign in with Google
          </button>
        </div>
      ) : (
        <div className="panel">
          <h2>Paste a sequence</h2>
          <form onSubmit={runSearch}>
            <textarea
              rows={4}
              placeholder="e.g. MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAP..."
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
            />
            <button className="button button-primary" type="submit" disabled={loading || sequence.trim().length < 10 || !modelReady}>
              {loading ? "Searching…" : "Search"}
            </button>
          </form>
          {error && <p className="error">{error}</p>}
          {result && (
            <div className="result">
              <p className="disclaimer">{result.notice}</p>
              <table>
                <thead>
                  <tr>
                    <th>Match</th>
                    <th>Similarity</th>
                    <th>Evolutionary group</th>
                  </tr>
                </thead>
                <tbody>
                  {result.hits.map((h) => (
                    <tr key={h.domain_id}>
                      <td>{h.domain_id}</td>
                      <td>{(h.similarity * 100).toFixed(1)}%</td>
                      <td style={{ fontSize: 12, color: "var(--text-dim)" }}>{h.superfamily}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
