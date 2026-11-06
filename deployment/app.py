"""Self-hosted Docker deployment: persistent translation engine + queued paper pipeline.

No model is downloaded at import time. The E_all reranker is the repository's
original pipeline, initialized lazily and reused for text requests.
"""
from __future__ import annotations

import gc
import hmac
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import UUID, uuid4
import zipfile

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
RERANKER = ROOT / "components" / "horizon-reranker"
if str(RERANKER) not in sys.path:
    sys.path.insert(0, str(RERANKER))

# Model loading and pipeline execution must not overlap on a single demo GPU.
_gpu_lock = threading.Lock()
_engine_lock = threading.Lock()
_engine = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-jobs")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _config_path() -> Path:
    return Path(os.environ.get("MODEL_CONFIG", "/workspace/rerank_config.json"))


def _job_root() -> Path:
    return Path(os.environ.get("JOB_DIR", "/workspace/jobs"))


def _limits() -> tuple[int, int]:
    return (int(os.environ.get("MAX_TEXT_CHARS", "12000")),
            int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024)


def _require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("API_KEY", "")
    if not expected:
        raise HTTPException(503, "API_KEY is not configured on the server")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Invalid API key")


def _configuration_status() -> dict:
    from run_pipeline import check_real_models
    try:
        paths = check_real_models(
            _config_path(), RERANKER,
            RERANKER / "results/final/weights.json",
            ROOT / "components/glossary_curated.json",
        )
        return {"configured": True, "adapter_paths": paths["adapters"]}
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {"configured": False, "reason": str(exc)}


class TranslationEngine:
    """Preserves the team's exact generate -> score -> E_all decode logic."""

    def __init__(self) -> None:
        from evaluation import glossary as G
        from reranker import config as C
        from reranker import pipeline as P
        from reranker.model import Translator
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected. Run Docker on a compatible NVIDIA GPU host.")
        status = _configuration_status()
        if not status["configured"]:
            raise RuntimeError(status["reason"])
        cfg = C.load(str(_config_path()))
        self.cfg = cfg
        self.P = P
        self.tr = Translator(cfg["adapter"], cfg.get("base_model"),
                             backward_adapter=cfg.get("backward_adapter"))
        self.gloss = G.load(str(ROOT / "components/glossary_curated.json"))
        weights = json.loads(
            (RERANKER / "results/final/weights.json").read_text(encoding="utf-8")
        )
        self.weights = weights["E_all"]["weights"]
        if not isinstance(self.weights, dict):
            raise RuntimeError("E_all reranker weights are missing")

    def translate(self, text: str) -> tuple[str, int]:
        from reranker.markdown_io import parse, render
        from run_pipeline import protect_markdown, restore_markdown

        guarded, replacements, _ = protect_markdown(text)
        blocks, units = parse(guarded)
        if not units:
            return text, 0
        doc = {"doc_id": "api", "source_en": " ".join(units),
               "reference_ko": "", "sentences": units}
        self.P.generate([doc], self.tr, self.cfg, self.gloss)
        self.P.score([doc], self.tr, self.cfg, self.gloss)
        chosen, _picked = self.P.decode([doc], self.tr, self.cfg,
                                       self.weights, self.gloss)
        result = render(blocks, chosen["api"])
        return restore_markdown(result, replacements).strip(), len(units)


def _get_engine() -> TranslationEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TranslationEngine()
    return _engine


def _unload_text_engine() -> None:
    """Release the API's persistent 7B model before the PDF CLI loads its own."""
    global _engine
    if _engine is None:
        return
    _engine = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_job(job_id: str, **updates) -> dict:
    with _jobs_lock:
        item = _jobs[job_id]
        item.update(updates)
        item["updated_at"] = _now()
        snapshot = item.copy()
    jobdir = _job_root() / job_id
    jobdir.mkdir(parents=True, exist_ok=True)
    temp = jobdir / "status.tmp.json"
    temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(jobdir / "status.json")
    return snapshot


def _find_job(job_id: str) -> dict:
    try:
        UUID(job_id)
    except ValueError:
        raise HTTPException(404, "Job not found")
    with _jobs_lock:
        item = _jobs.get(job_id)
        if item:
            return item.copy()
    saved = _job_root() / job_id / "status.json"
    if not saved.is_file():
        raise HTTPException(404, "Job not found")
    item = json.loads(saved.read_text(encoding="utf-8"))
    # Tasks are not persisted across Pod restarts; do not claim a phantom job is running.
    if item["status"] in {"queued", "running"}:
        item["status"] = "interrupted"
        item["error"] = "Host/container was restarted before the job completed"
    return item


def _run_job(job_id: str, ext: str, fmt: str) -> None:
    jobdir = _job_root() / job_id
    input_path = jobdir / ("input" + ext)
    output_dir = jobdir / "output"
    log_path = jobdir / "pipeline.log"
    try:
        _save_job(job_id, status="running")
        argv = [sys.executable, str(ROOT / "run_pipeline.py"),
                "--input", str(input_path), "--out", str(output_dir),
                "--name", "result", "--mode", "real",
                "--config", str(_config_path()),
                "--renderer", "real" if fmt == "pdf" else "skip",
                "--main-font", "Noto Serif CJK KR",
                "--sans-font", "Noto Sans CJK KR", "--no-previews"]
        if ext == ".pdf" and not shutil.which("mineru-kit"):
            raise RuntimeError("PDF parsing requires mineru-kit; upload MinerU ZIP or Markdown instead")
        # A dedicated subprocess runs the unmodified full pipeline, including its
        # own model load, while the API avoids competing for GPU memory.
        with _gpu_lock:
            _unload_text_engine()
            with log_path.open("wb") as log:
                proc = subprocess.run(
                    argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    timeout=int(os.environ.get("MAX_JOB_SECONDS", "7200")),
                    check=False,
                )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
            raise RuntimeError(f"Pipeline exited with {proc.returncode}: {tail}")
        output = (output_dir / "result.ko.pdf" if fmt == "pdf"
                  else output_dir / "document" / "paper.ko.md")
        if not output.is_file():
            raise RuntimeError("Pipeline completed without the expected output")
        _save_job(job_id, status="completed", result_name=output.name)
    except Exception as exc:
        _save_job(job_id, status="failed", error=str(exc)[:3000])


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)


app = FastAPI(title="CS Paper Translation API", version="1.0.0",
              description="Stage4 LoRA + Bomin backward LoRA + E_all reranker")


@app.get("/health")
def health():
    return {"ok": True, "service": "cs-paper-api", "model_loaded": _engine is not None}


@app.get("/ready", dependencies=[Depends(_require_key)])
def ready():
    status = _configuration_status()
    try:
        import torch
        status["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        status["cuda_available"] = False
    status["model_loaded"] = _engine is not None
    status["pdf_parser_available"] = bool(shutil.which("mineru-kit"))
    status["pdf_renderer_available"] = bool(shutil.which("xelatex") and shutil.which("pandoc"))
    return status


@app.post("/translate", dependencies=[Depends(_require_key)])
def translate(req: TranslateRequest):
    max_chars, _ = _limits()
    if len(req.text) > max_chars:
        raise HTTPException(413, f"Text exceeds {max_chars} characters")
    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(429, "GPU is processing another request")
    try:
        try:
            engine = _get_engine()
            result, n = engine.translate(req.text)
            return {"translation": result, "units": n, "system": "Stage4+E_all"}
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise HTTPException(503, f"Translation model unavailable: {exc}")
    finally:
        _gpu_lock.release()


@app.post("/jobs", status_code=202, dependencies=[Depends(_require_key)])
async def create_job(file: UploadFile = File(...), output_format: str = "pdf"):
    if output_format not in {"pdf", "markdown"}:
        raise HTTPException(422, "output_format must be pdf or markdown")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".pdf", ".zip", ".md"}:
        raise HTTPException(415, "Only PDF, MinerU ZIP, and Markdown files are accepted")
    if ext == ".pdf" and not shutil.which("mineru-kit"):
        raise HTTPException(503, "mineru-kit is not installed; upload MinerU ZIP or Markdown")
    if output_format == "pdf" and not (shutil.which("pandoc") and shutil.which("xelatex")):
        raise HTTPException(503, "PDF renderer is not installed")
    if not _configuration_status()["configured"]:
        raise HTTPException(503, "Missing Stage4/Bomin LoRA adapter weights; see /ready")
    with _jobs_lock:
        pending = sum(j["status"] in {"queued", "running"} for j in _jobs.values())
    if pending >= int(os.environ.get("MAX_PENDING_JOBS", "2")):
        raise HTTPException(429, "Job queue is full; retry later")
    job_id = str(uuid4())
    folder = _job_root() / job_id
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / ("input" + ext)
    _, max_bytes = _limits()
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File exceeds {max_bytes // (1024 * 1024)} MB")
                out.write(chunk)
        if ext == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    items = zf.infolist()
                    if len(items) > 5000 or sum(i.file_size for i in items) > 400 * 1024 * 1024:
                        raise HTTPException(413, "ZIP expanded contents exceed safety limit")
            except zipfile.BadZipFile:
                raise HTTPException(422, "Invalid ZIP file")
        with _jobs_lock:
            _jobs[job_id] = {"id": job_id, "status": "queued", "created_at": _now(),
                             "updated_at": _now(), "output_format": output_format,
                             "input_type": ext, "error": None, "result_name": None}
        _save_job(job_id)
        _executor.submit(_run_job, job_id, ext, output_format)
        return {"job_id": job_id, "status": "queued",
                "status_url": f"/jobs/{job_id}", "download_url": f"/jobs/{job_id}/download"}
    except Exception:
        if job_id not in _jobs:
            shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        await file.close()


@app.get("/jobs/{job_id}", dependencies=[Depends(_require_key)])
def job_status(job_id: str):
    return _find_job(job_id)


@app.get("/jobs/{job_id}/download", dependencies=[Depends(_require_key)])
def job_download(job_id: str):
    item = _find_job(job_id)
    if item["status"] != "completed":
        raise HTTPException(409, f"Job status: {item['status']}")
    folder = _job_root() / job_id / "output"
    path = (folder / "result.ko.pdf" if item["output_format"] == "pdf"
            else folder / "document" / "paper.ko.md")
    if not path.is_file():
        raise HTTPException(404, "Result file missing")
    media = "application/pdf" if path.suffix == ".pdf" else "text/markdown; charset=utf-8"
    return FileResponse(path, media_type=media, filename=path.name)
