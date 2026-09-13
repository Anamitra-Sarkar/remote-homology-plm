#!/bin/sh
set -e
if [ "$MODEL_RELEASE_APPROVED" = "true" ] && [ -n "$APPROVED_ARTIFACT_REVISION" ] && [ -n "$MODEL_ARTIFACT_REPO_ID" ]; then
  echo "[entrypoint] Downloading approved artifacts from $MODEL_ARTIFACT_REPO_ID"
  python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
dest_dir = os.environ.get('MODEL_ARTIFACT_DIR', '/app/artifacts')
os.makedirs(dest_dir, exist_ok=True)
for fname in ('embeddings.npy', 'domains.json', 'eval_report.json'):
    downloaded = hf_hub_download(
        repo_id=os.environ['MODEL_ARTIFACT_REPO_ID'],
        filename=fname,
        repo_type='model',
        token=os.environ.get('HF_TOKEN'),
    )
    shutil.copy(downloaded, os.path.join(dest_dir, fname))
    print('[entrypoint] downloaded', fname)
"
else
  echo "[entrypoint] No approved release configured; starting in fail-closed abstention mode."
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 7860
