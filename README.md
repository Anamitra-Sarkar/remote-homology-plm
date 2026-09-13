# Remote Homology Detection

Portfolio **Topic #24**: Protein Language Model-Assisted Remote Homology Detection Beyond
Conventional Sequence Similarity.

Two proteins can descend from the same evolutionary ancestor while sharing almost no
detectable sequence similarity — the "twilight zone" that conventional sequence-alignment
search (BLAST-style, or simple k-mer similarity) is known to miss. This project tests,
honestly and on real data, whether protein-language-model embeddings (ESM-2) actually find
these remote relationships better than a conventional sequence-similarity baseline.

## Real data & evaluation

- **SCOPe 2.08** (Structural Classification of Proteins — extended): real, public,
  research-standard structural classification with fold/superfamily/family labels.
  `dir.cla.scope.2.08-stable.txt` (classification) + the ASTRAL sequence set (≤100%
  identity-filtered).
- **Protocol**: family-disjoint. For each held-out query domain, a true positive is another
  domain in the *same superfamily but a different family* — a genuine remote homolog, not a
  near-identical duplicate. Same-family pairs are excluded from scoring (too easy). Negatives
  are same-fold, different-superfamily domains.
- **Model**: `facebook/esm2_t12_35M_UR50D` (public checkpoint, no gating), mean-pooled
  per-sequence embedding, cosine similarity for retrieval.
- **Baseline**: k-mer(k=3) Jaccard similarity — a real, standard, non-PLM sequence-similarity
  measure, computed on the exact same real test pairs.
- **Metric**: AUROC and recall@10 for both methods, reported honestly whichever way the
  comparison actually goes (see `/report` on the live backend, or the Methodology page).

See `kaggle/train_real_kaggle.py` for the full, real, reproducible pipeline (downloads real
SCOPe data — no auth needed — computes real ESM-2 embeddings, runs the real baseline, and
reports the real comparison).

## Architecture

- `backend/` — FastAPI service. Fail-closed release gate (`MODEL_RELEASE_APPROVED` +
  `APPROVED_ARTIFACT_REVISION`): serves real results only once a real trained artifact has
  been approved and downloaded; otherwise honestly reports not-ready. Firebase ID-token
  verification gates the live `/search` endpoint (shared `cabbage-guard` portfolio Firebase
  project). `/health` and `/report` are public.
- `frontend/` — Vite + React, light theme, three pages (Overview / Search / Methodology),
  sign-in gated search.
- `kaggle/` — the real training/evaluation pipeline.

## Disclaimer

Research tool only. A high similarity score is evidence of a *possible* remote evolutionary
relationship, not a confirmed structural or functional match.
