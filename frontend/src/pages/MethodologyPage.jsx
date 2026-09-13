import React, { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "";

export default function MethodologyPage() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!API_BASE) return;
    fetch(`${API_BASE}/report`)
      .then((r) => {
        if (!r.ok) throw new Error(`backend returned ${r.status}`);
        return r.json();
      })
      .then((d) => setReport(d.summary || d))
      .catch((e) => setError(String(e)));
  }, []);

  const emb = report?.mean_auroc?.embedding;
  const kmer = report?.mean_auroc?.kmer_baseline;
  const better = emb !== undefined && kmer !== undefined ? emb > kmer : null;

  return (
    <div className="container">
      <div className="page-header">
        <h1>Methodology</h1>
        <p>How this is tested, and the real result — reported honestly either way.</p>
      </div>

      <section className="panel">
        <h2>The test</h2>
        <p className="lede">
          Real classified protein domains (SCOPe) were split so that, for each test protein, its true
          distant relatives — proteins in the same broad evolutionary group but a genuinely different,
          low-similarity sub-family — were held out and never seen as "easy" near-duplicates. Both this
          tool's protein-language-model search and a conventional sequence-similarity search were then
          asked to find those same relatives, on the exact same real test cases.
        </p>
        {error && <p className="caveat">Could not reach the backend ({error}). The API may still be starting up.</p>}
        {report && (
          <>
            <p style={{ fontSize: 15, fontWeight: 600, marginTop: 8 }}>
              {better === null
                ? "Result not yet available."
                : better
                ? "On this real test, the protein-language-model search found more true distant relatives than a conventional sequence search."
                : "On this real test, the protein-language-model search did not clearly beat a conventional sequence search — reported honestly, not hidden."}
            </p>
            <table className="metrics">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Accuracy at telling relatives apart</th>
                  <th>Found in top 10 results</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Protein language model (this tool)</td>
                  <td className="num">{(emb * 100).toFixed(1)}%</td>
                  <td className="num">{(report.mean_recall_at_10?.embedding * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                  <td>Conventional sequence similarity (baseline)</td>
                  <td className="num">{(kmer * 100).toFixed(1)}%</td>
                  <td className="num">{(report.mean_recall_at_10?.kmer_baseline * 100).toFixed(1)}%</td>
                </tr>
              </tbody>
            </table>
            <p className="caveat">
              Tested on {report.n_queries_scored} real held-out query proteins against a reference set of{" "}
              {report.n_domains_total} real classified protein domains. "Accuracy at telling relatives apart"
              is the chance the method correctly ranks a true distant relative above an unrelated protein.
            </p>
          </>
        )}
      </section>

      <section className="panel">
        <h2>Data sources (real, public)</h2>
        <div className="source-grid">
          <div className="source-card"><b>SCOPe 2.08</b><span>Structural classification of proteins, extended · public research database</span></div>
          <div className="source-card"><b>ESM-2</b><span>Protein language model (35M parameters) · Meta AI, public checkpoint</span></div>
        </div>
      </section>
    </div>
  );
}
