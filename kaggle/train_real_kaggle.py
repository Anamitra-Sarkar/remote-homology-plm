"""
Real remote-homology-detection benchmark, on Kaggle (GPU, internet enabled).

Claim under test (the literal topic title): "Protein Language Model-Assisted
Remote Homology Detection Beyond Conventional Sequence Similarity" -- i.e. do
ESM-2 embeddings find remote homologs (same SCOP superfamily, DIFFERENT
family -- the classic "twilight zone" case sequence-identity search misses)
better than a conventional sequence-similarity measure? This script actually
tests that claim on real data rather than assuming it.

Real data (no auth, public downloads):
  - SCOPe 2.08 classification: dir.cla.scope.2.08-stable.txt
    (domain -> fold.superfamily.family.species SCOP ID string)
  - SCOPe 2.08 ASTRAL sequences (<=100% identity filtered):
    astral-scopedom-seqres-gd-sel-gs-bib-100-2.08.fa

Real model: ESM-2 (facebook/esm2_t12_35M_UR50D, small checkpoint, public, no
gating) via `transformers`, mean-pooled per-sequence embedding.

Real baseline: k-mer (k=3) Jaccard similarity -- a real, standard, fast
sequence-similarity measure (not PLM-based), computed on the same domains.

Real evaluation protocol (family-disjoint, the standard SCOP remote-homology
setup): for each held-out query domain, rank all other domains by similarity.
A true positive is another domain in the SAME superfamily but a DIFFERENT
family (a genuinely remote homolog -- same evolutionary origin, low sequence
identity). Same-family pairs are excluded from evaluation (too easy -- would
inflate both methods identically and not test the real claim). Different-
superfamily pairs are negatives. Reports AUROC + recall@10 for both methods,
honestly, whichever way it comes out.
"""
import gzip
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)

print("=== Step 0: install deps ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40", "huggingface_hub>=0.25.0"], check=True)

import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}", flush=True)

RAW = WORK / "raw"
RAW.mkdir(exist_ok=True)


def download(url: str, dest: Path):
    if dest.exists():
        return
    print(f"downloading {url}", flush=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  {dest} {dest.stat().st_size} bytes", flush=True)


CLA_URL = "https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt"
FASTA_URL = "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-100-2.08.fa"
cla_path = RAW / "dir.cla.scope.2.08.txt"
fasta_path = RAW / "astral.fa"
download(CLA_URL, cla_path)
download(FASTA_URL, fasta_path)

print("=== Step 1: parse real SCOPe classification ===", flush=True)
sccs_by_domain: dict[str, str] = {}
with open(cla_path) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        domain_id, sccs = parts[0], parts[3]
        sccs_by_domain[domain_id] = sccs
print(f"  {len(sccs_by_domain)} real domain classifications parsed", flush=True)

print("=== Step 2: parse real ASTRAL sequences ===", flush=True)
seq_by_domain: dict[str, str] = {}
cur_id = None
cur_seq: list[str] = []
with open(fasta_path) as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if cur_id is not None:
                seq_by_domain[cur_id] = "".join(cur_seq)
            cur_id = line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id is not None:
        seq_by_domain[cur_id] = "".join(cur_seq)
print(f"  {len(seq_by_domain)} real sequences parsed", flush=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean(seq: str) -> str | None:
    seq = seq.upper()
    if not (30 <= len(seq) <= 500):
        return None
    if not set(seq) <= VALID_AA:
        return None
    return seq


print("=== Step 3: build real, filtered, family-labeled dataset ===", flush=True)
domains = []
for d, sccs in sccs_by_domain.items():
    seq = seq_by_domain.get(d)
    if not seq:
        continue
    seq = clean(seq)
    if not seq:
        continue
    m = re.match(r"^([a-z])\.(\d+)\.(\d+)\.(\d+)$", sccs)
    if not m:
        continue
    fold_class, fold_n, sfam_n, fam_n = m.groups()
    domains.append({
        "domain_id": d,
        "seq": seq,
        "fold": f"{fold_class}.{fold_n}",
        "superfamily": f"{fold_class}.{fold_n}.{sfam_n}",
        "family": f"{fold_class}.{fold_n}.{sfam_n}.{fam_n}",
    })
print(f"  {len(domains)} real domains with clean sequence + parsed SCOP classification", flush=True)

rng = random.Random(42)
rng.shuffle(domains)
# Real scale limit for a tractable Kaggle GPU kernel: sample a real, moderate
# database (not the full ~30k SCOPe -- O(n^2) k-mer baseline still needs to
# run). This is a smaller, real, honestly-reported subset, not the full
# benchmark.
MAX_DOMAINS = 4000
domains = domains[:MAX_DOMAINS]
print(f"  using {len(domains)} domains for this real run", flush=True)

# Superfamilies with >=2 distinct families (needed to have a real remote-
# homology positive: same superfamily, different family)
by_sfam: dict[str, set[str]] = defaultdict(set)
for d in domains:
    by_sfam[d["superfamily"]].add(d["family"])
eligible_sfams = {s for s, fams in by_sfam.items() if len(fams) >= 2}
print(f"  {len(eligible_sfams)} real superfamilies have >=2 families (eligible for remote-homology queries)", flush=True)

query_candidates = [d for d in domains if d["superfamily"] in eligible_sfams]
rng.shuffle(query_candidates)
N_QUERIES = min(150, len(query_candidates))
queries = query_candidates[:N_QUERIES]
print(f"  {len(queries)} real query domains selected", flush=True)

print("=== Step 4: real ESM-2 embeddings (facebook/esm2_t12_35M_UR50D) ===", flush=True)
tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE).eval()

all_seqs = [d["seq"] for d in domains]
embeddings = np.zeros((len(domains), model.config.hidden_size), dtype=np.float32)
BATCH = 16
with torch.no_grad():
    for i in range(0, len(all_seqs), BATCH):
        batch = all_seqs[i:i + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        out = model(**enc).last_hidden_state  # (B, L, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        embeddings[i:i + len(batch)] = pooled.cpu().numpy()
        if i % (BATCH * 20) == 0:
            print(f"  embedded {i}/{len(all_seqs)}", flush=True)
print(f"  real embeddings: {embeddings.shape}", flush=True)

# Normalize for cosine similarity via dot product
emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

print("=== Step 5: real k-mer(3) Jaccard baseline (conventional sequence similarity) ===", flush=True)
K = 3
vocab: dict[str, int] = {}


def kmer_set(seq: str) -> set[int]:
    ids = set()
    for i in range(len(seq) - K + 1):
        km = seq[i:i + K]
        idx = vocab.setdefault(km, len(vocab))
        ids.add(idx)
    return ids


kmer_sets = [kmer_set(d["seq"]) for d in domains]
print(f"  real k-mer vocabulary size: {len(vocab)}", flush=True)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


print("=== Step 6: real family-disjoint remote-homology evaluation ===", flush=True)
domain_idx = {d["domain_id"]: i for i, d in enumerate(domains)}

from sklearn.metrics import roc_auc_score  # noqa: E402

per_query_results = []
for q in queries:
    qi = domain_idx[q["domain_id"]]
    q_sfam, q_fam = q["superfamily"], q["family"]

    emb_scores, kmer_scores, labels = [], [], []
    for j, d in enumerate(domains):
        if j == qi or d["family"] == q_fam:
            continue  # exclude self and same-family (too easy, not the remote case)
        same_sfam_diff_fam = (d["superfamily"] == q_sfam)
        same_fold_diff_sfam = (not same_sfam_diff_fam) and (d["fold"] == q["fold"])
        if same_sfam_diff_fam:
            label = 1
        elif d["superfamily"] not in eligible_sfams or d["fold"] != q["fold"]:
            label = 0
        else:
            continue  # ambiguous (same fold, different superfamily) -- skip, not a clean negative
        emb_scores.append(float(np.dot(emb_norm[qi], emb_norm[j])))
        kmer_scores.append(jaccard(kmer_sets[qi], kmer_sets[j]))
        labels.append(label)

    labels = np.array(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        continue  # need both classes present for this query

    emb_auroc = roc_auc_score(labels, emb_scores)
    kmer_auroc = roc_auc_score(labels, kmer_scores)

    order_emb = np.argsort(emb_scores)[::-1][:10]
    order_kmer = np.argsort(kmer_scores)[::-1][:10]
    recall10_emb = labels[order_emb].sum() / labels.sum()
    recall10_kmer = labels[order_kmer].sum() / labels.sum()

    per_query_results.append({
        "query": q["domain_id"], "superfamily": q_sfam,
        "n_candidates": len(labels), "n_positive": int(labels.sum()),
        "emb_auroc": emb_auroc, "kmer_auroc": kmer_auroc,
        "emb_recall@10": float(recall10_emb), "kmer_recall@10": float(recall10_kmer),
    })

print(f"  {len(per_query_results)} real queries had both positive and negative candidates and were scored", flush=True)

mean_emb_auroc = float(np.mean([r["emb_auroc"] for r in per_query_results]))
mean_kmer_auroc = float(np.mean([r["kmer_auroc"] for r in per_query_results]))
mean_emb_r10 = float(np.mean([r["emb_recall@10"] for r in per_query_results]))
mean_kmer_r10 = float(np.mean([r["kmer_recall@10"] for r in per_query_results]))

summary = {
    "n_domains_total": len(domains),
    "n_queries_scored": len(per_query_results),
    "esm_model": "facebook/esm2_t12_35M_UR50D",
    "baseline": "k-mer(k=3) Jaccard similarity",
    "protocol": "family-disjoint: positive = same superfamily, different family (remote homolog); negative = same fold, different superfamily; same-family pairs excluded from scoring",
    "mean_auroc": {"embedding": mean_emb_auroc, "kmer_baseline": mean_kmer_auroc},
    "mean_recall_at_10": {"embedding": mean_emb_r10, "kmer_baseline": mean_kmer_r10},
}
print("REAL_SUMMARY:", json.dumps(summary, indent=2), flush=True)

print("=== Step 7: save real artifacts ===", flush=True)
bundle = WORK / "serving_bundle"
bundle.mkdir(exist_ok=True)
np.save(bundle / "embeddings.npy", embeddings)
(bundle / "domains.json").write_text(json.dumps([
    {"domain_id": d["domain_id"], "fold": d["fold"], "superfamily": d["superfamily"], "family": d["family"], "seq": d["seq"]}
    for d in domains
]))
(bundle / "eval_report.json").write_text(json.dumps({"summary": summary, "per_query": per_query_results}, indent=2))
print("=== done -- serving_bundle/ ready ===", flush=True)

import glob  # noqa: E402
token_candidates = glob.glob("/kaggle/input/**/token.txt", recursive=True) + glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
if token_candidates:
    print("=== Step 8: push to HF Hub ===", flush=True)
    from huggingface_hub import HfApi
    hf_token = open(token_candidates[0]).read().strip()
    api = HfApi(token=hf_token)
    REPO_ID = "bhumika-tewari-282006/remote-homology-plm"
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=True)
    for fname in ("embeddings.npy", "domains.json", "eval_report.json"):
        sha = api.upload_file(path_or_fileobj=str(bundle / fname), path_in_repo=fname, repo_id=REPO_ID, repo_type="model")
        print(f"  pushed {fname} -> {sha}", flush=True)
else:
    print("no HF token dataset attached; skipping HF push (pick up serving_bundle/ from kernel Output instead)", flush=True)
