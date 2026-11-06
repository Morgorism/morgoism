"""GPU-free API smoke tests. No neural translation is claimed."""
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment import app as module


def test_health_and_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "secret-test-token")
    monkeypatch.setenv("JOB_DIR", str(tmp_path / "jobs"))
    client = TestClient(module.app)
    res = client.get("/health")
    assert res.status_code == 200 and res.json()["ok"]
    assert client.get("/ready").status_code == 401
    assert client.get("/ready", headers={"X-API-Key": "incorrect"}).status_code == 401
    assert client.get("/ready", headers={"X-API-Key": "secret-test-token"}).status_code == 200
    assert client.post("/translate", json={"text": "Hello"}).status_code == 401
    assert client.get("/jobs/bad-id", headers={"X-API-Key": "secret-test-token"}).status_code == 404


def test_fake_engine_route(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-test-token")
    class Dummy:
        def translate(self, text):
            return "한국어 번역 (테스트)", 1
    monkeypatch.setattr(module, "_get_engine", lambda: Dummy())
    client = TestClient(module.app)
    res = client.post("/translate", json={"text": "Hello"},
                      headers={"X-API-Key": "secret-test-token"})
    assert res.status_code == 200
    assert res.json()["translation"] == "한국어 번역 (테스트)"


def test_upload_rejects_missing_parser_and_extension(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-test-token")
    monkeypatch.setattr(module.shutil, "which", lambda exe: None)
    client = TestClient(module.app)
    headers = {"X-API-Key": "secret-test-token"}
    resp = client.post("/jobs", files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
                       headers=headers)
    assert resp.status_code == 503
    resp = client.post("/jobs", files={"file": ("hack.exe", b"!", "application/octet-stream")},
                       headers=headers)
    assert resp.status_code == 415


def test_job_lifecycle_simulated(monkeypatch, tmp_path):
    """Test HTTP upload/poll/download, substituting the neural CLI only."""
    monkeypatch.setenv("API_KEY", "secret-test-token")
    monkeypatch.setenv("JOB_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(module, "_configuration_status", lambda: {"configured": True})
    monkeypatch.setattr(module.shutil, "which", lambda program: "/usr/bin/" + program)
    # No real GPU subprocess; create a fake PDF file under expected output dir.
    def pretend_run(argv, **kwargs):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.ko.pdf").write_bytes(b"%PDF-1.4\nfake-test-pdf")
        class Process:
            returncode = 0
        return Process()
    monkeypatch.setattr(module.subprocess, "run", pretend_run)
    monkeypatch.setattr(module, "_unload_text_engine", lambda: None)
    client = TestClient(module.app)
    h = {"X-API-Key": "secret-test-token"}
    response = client.post("/jobs", files={"file": ("paper.md", b"# Paper\n\nHello.", "text/markdown")},
                           headers=h)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    future_state = "queued"
    import time
    for _ in range(40):
        result = client.get(f"/jobs/{job_id}", headers=h)
        assert result.status_code == 200
        future_state = result.json()["status"]
        if future_state == "completed":
            break
        time.sleep(0.05)
    assert future_state == "completed"
    downloaded = client.get(f"/jobs/{job_id}/download", headers=h)
    assert downloaded.status_code == 200
    assert downloaded.content.startswith(b"%PDF")
