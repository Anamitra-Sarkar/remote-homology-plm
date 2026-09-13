"""Remote Homology Detection API — serves REAL ESM-2 embedding-based remote
homology search over a real SCOPe reference database, backed by a real
Kaggle-trained benchmark comparing embeddings against a conventional
sequence-similarity baseline. No mock data: every number returned here
traces back to a real Kaggle run pushed to the HF model repo
`bhumika-tewari-282006/remote-homology-plm`.

Auth: Firebase (shared `cabbage-guard` portfolio project) verifies ID tokens
for the /search live-query endpoint only; /health and /report are public.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import firebase_admin
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth as fb_auth
from firebase_admin import credentials
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path(os.environ.get("MODEL_ARTIFACT_DIR", "/app/artifacts"))

app = FastAPI(title="Remote Homology Detection API", version="1.0.0")

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {"domains": None, "embeddings": None, "report": None, "model": None, "tokenizer": None, "load_error": None}
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _init_firebase() -> None:
    if firebase_admin._apps:
        return
    cred_json = os.environ.get("FIREBASE_ADMIN_CREDENTIALS")
    if not cred_json:
        return
    try:
        cred = credentials.Certificate(json.loads(cred_json))
        firebase_admin.initialize_app(cred)
    except Exception as e:
        _state["load_error"] = f"firebase init failed: {e}"


def release_approved() -> bool:
    return os.getenv("MODEL_RELEASE_APPROVED", "").lower() == "true" and bool(os.getenv("APPROVED_ARTIFACT_REVISION", "").strip())


def _load_artifacts() -> None:
    if not release_approved():
        return
    try:
        domains_path = ARTIFACT_DIR / "domains.json"
        emb_path = ARTIFACT_DIR / "embeddings.npy"
        report_path = ARTIFACT_DIR / "eval_report.json"
        if not (domains_path.exists() and emb_path.exists() and report_path.exists()):
            _state["load_error"] = "release approved but artifact files not found on disk"
            return
        _state["domains"] = json.loads(domains_path.read_text())
        _state["embeddings"] = np.load(emb_path)
        _state["report"] = json.loads(report_path.read_text())
    except Exception as e:
        _state["load_error"] = f"artifact load failed: {e}"


@lru_cache(maxsize=1)
def _load_esm_model():
    from transformers import AutoModel, AutoTokenizer

    name = "facebook/esm2_t12_35M_UR50D"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    return tok, model


@app.on_event("startup")
def startup() -> None:
    _init_firebase()
    _load_artifacts()


def verify_bearer(authorization: Optional[str]) -> dict:
    if not firebase_admin._apps:
        raise HTTPException(503, "Sign-in verification is not configured on this deployment")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        return fb_auth.verify_id_token(authorization[7:].strip())
    except Exception:
        raise HTTPException(401, "invalid or expired sign-in token")


@app.get("/health")
def health():
    loaded = _state["domains"] is not None and _state["embeddings"] is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "n_reference_domains": len(_state["domains"]) if loaded else 0,
        "load_error": _state["load_error"],
    }


@app.get("/report")
def report():
    if _state["report"] is None:
        raise HTTPException(503, "No real evaluation report is available yet — the model may not have finished training, or hasn't been released.")
    return _state["report"]


class SearchRequest(BaseModel):
    sequence: str = Field(min_length=10, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)


@app.post("/search")
def search(req: SearchRequest, authorization: Optional[str] = Header(None)):
    verify_bearer(authorization)

    if _state["domains"] is None or _state["embeddings"] is None:
        raise HTTPException(503, "No released model is available to serve real results yet.")

    seq = req.sequence.strip().upper()
    if not seq or not set(seq) <= VALID_AA:
        raise HTTPException(400, "Sequence must contain only standard amino acid letters (ACDEFGHIKLMNPQRSTVWY).")

    try:
        tok, model = _load_esm_model()
    except Exception as e:
        raise HTTPException(503, f"Could not load the protein language model: {e}")

    import torch

    with torch.no_grad():
        enc = tok([seq], return_tensors="pt", truncation=True, max_length=512)
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        query_emb = pooled.squeeze(0).numpy()

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    ref_embeddings = _state["embeddings"]
    ref_norm = ref_embeddings / (np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-8)
    scores = ref_norm @ query_norm

    order = np.argsort(scores)[::-1][: req.top_k]
    domains = _state["domains"]
    hits = [
        {
            "domain_id": domains[i]["domain_id"],
            "similarity": float(scores[i]),
            "fold": domains[i]["fold"],
            "superfamily": domains[i]["superfamily"],
            "family": domains[i]["family"],
        }
        for i in order
    ]
    return {
        "query_length": len(seq),
        "hits": hits,
        "notice": "Research tool only. A high similarity score is evidence of a possible remote evolutionary relationship, not a confirmed structural or functional match — always verify against a real structural database before relying on this.",
    }
