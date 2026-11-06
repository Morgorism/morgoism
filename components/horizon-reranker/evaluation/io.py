"""Read predictions produced by test.py and optional test-set metadata."""
import hashlib
import json
from pathlib import Path

# test.py writes: id, source_en, reference_ko, prediction_ko, status, hit_max_tokens, adapter_path
REQUIRED = ("id", "source_en", "reference_ko", "prediction_ko")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
    return rows


def load_predictions(path, testset=None):
    rows = read_jsonl(path)
    for i, r in enumerate(rows, 1):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise ValueError(f"{path} record {i}: missing fields {missing}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids found; one prediction per id is required")

    meta = {}
    if testset:
        for t in read_jsonl(testset):
            meta[t["id"]] = t
        unknown = [i for i in ids if i not in meta]
        if unknown:
            raise ValueError(f"{len(unknown)} prediction ids not in test set, e.g. {unknown[:3]}")
        for r in rows:
            t = meta[r["id"]]
            if t["source_en"].strip() != r["source_en"].strip():
                raise ValueError(f"id {r['id']}: source_en differs from test set (misaligned file?)")
            for k in ("domain", "journal", "paper_id"):
                r.setdefault(k, t.get(k))
        missing_ids = set(meta) - set(ids)
    else:
        missing_ids = set()
    return rows, missing_ids


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
