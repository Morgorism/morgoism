"""Tests for the client-side MinerU bridge (no GPU, network, or MinerU needed)."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment import submit_pdf_local as bridge


class FakeResponse:
    def __init__(self, payload=None, data=b"%PDF-1.4\ncompleted", status=200):
        self.payload = payload or {}
        self.data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, n):
        yield self.data


def test_local_mineru_to_api(monkeypatch, tmp_path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\noriginal")
    output = tmp_path / "paper.ko.pdf"
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert cmd[0] == "/fake/bin/mineru-kit"
        assert cmd[-4:] == ["--format", "zip", "--tier", "standard"]
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"PK\x03\x04fakezip")

    class Requests:
        def __init__(self):
            self.gets = 0
        def post(self, url, *, params, headers, files, timeout):
            assert url == "http://127.0.0.1:8000/jobs"
            assert params == {"output_format": "pdf"}
            assert headers == {"X-API-Key": "secret"}
            assert files["file"][0].endswith(".mineru.zip")
            return FakeResponse({"job_id": "abc-123"})
        def get(self, url, *, headers, timeout=None, stream=False):
            if stream:
                assert url.endswith("/download")
                return FakeResponse()
            self.gets += 1
            return FakeResponse({"status": "running" if self.gets == 1 else "completed"})

    monkeypatch.setattr(bridge.shutil, "which", lambda x: "/fake/bin/mineru-kit")
    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    monkeypatch.setattr(bridge.time, "sleep", lambda x: None)
    monkeypatch.setitem(sys.modules, "requests", Requests())
    result = bridge.run(paper, api_url="http://127.0.0.1:8000", api_key="secret",
                        output=output, poll_seconds=1)
    assert result == output.resolve()
    assert output.read_bytes().startswith(b"%PDF-")
    assert len(calls) == 1


def test_missing_mineru(monkeypatch, tmp_path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(bridge.shutil, "which", lambda x: None)
    try:
        bridge.run(paper, api_url="http://127.0.0.1:8000", api_key="secret",
                   output=tmp_path / "out.pdf")
    except RuntimeError as exc:
        assert "mineru-kit" in str(exc)
    else:
        raise AssertionError("Must reject missing MinerU CLI")


def test_missing_key(monkeypatch, tmp_path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4")
    try:
        bridge.run(paper, api_url="http://127.0.0.1:8000", api_key="",
                   output=tmp_path / "out.pdf")
    except ValueError as exc:
        assert "API_KEY" in str(exc)
    else:
        raise AssertionError("Must reject missing key")
