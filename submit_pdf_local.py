#!/usr/bin/env python3
"""Run MinerU Standard on the caller's computer, then submit its ZIP to the local API.

MinerU is intentionally external to the LLM/Docker container. This script is
not used inside the API container; it runs on a client machine with mineru-kit.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def run(pdf: Path, *, api_url: str, api_key: str, output: Path,
        output_format: str = "pdf", poll_seconds: int = 5,
        wait_seconds: int = 7200, keep_zip: bool = False) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests가 없습니다. python -m pip install requests") from exc

    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError(f"PDF 파일을 찾을 수 없습니다: {pdf}")
    if not api_key or api_key == "REPLACE_WITH_LONG_RANDOM_TOKEN":
        raise ValueError("API_KEY 환경 변수를 설정하세요 (.env와 동일한 값)")
    mineru = shutil.which("mineru-kit")
    if not mineru:
        raise RuntimeError(
            "이 PC에 mineru-kit 명령이 없습니다. MinerU Standard를 별도 환경에 설치하고 "
            "`mineru-kit --help`가 실행되는지 확인하세요."
        )
    root_url = api_url.rstrip("/")
    auth = {"X-API-Key": api_key}
    dest = output.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cs_paper_mineru_") as temporary:
        zip_path = Path(temporary) / (pdf.stem + ".mineru.zip")
        cmd = [mineru, "parse", str(pdf.resolve()), "-o", str(zip_path),
               "--format", "zip", "--tier", "standard"]
        print("[1/3] 로컬 MinerU Standard로 PDF 분석 중...", flush=True)
        subprocess.run(cmd, check=True)
        if not zip_path.is_file() or zip_path.stat().st_size == 0:
            raise RuntimeError("MinerU 결과 ZIP을 생성하지 못했습니다")
        if keep_zip:
            kept = pdf.with_name(pdf.stem + ".mineru.zip")
            if kept.resolve() == zip_path.resolve():
                raise RuntimeError("임시 ZIP 경로가 원본과 충돌합니다")
            shutil.copy2(zip_path, kept)
            print("MinerU ZIP 보관:", kept)

        print("[2/3] 번역 API로 ZIP 제출 중...", flush=True)
        with zip_path.open("rb") as f:
            response = requests.post(
                root_url + "/jobs",
                params={"output_format": output_format}, headers=auth,
                files={"file": (zip_path.name, f, "application/zip")}, timeout=120,
            )
        response.raise_for_status()
        job_id = response.json().get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("API 응답에 job_id가 없습니다")
        print("작업 ID:", job_id, flush=True)
        print("[3/3] GPU 번역 작업 완료 대기 중...", flush=True)
        deadline = time.monotonic() + wait_seconds
        previous = None
        while time.monotonic() < deadline:
            state_res = requests.get(root_url + f"/jobs/{job_id}",
                                     headers=auth, timeout=30)
            state_res.raise_for_status()
            state = state_res.json()
            status = state.get("status")
            if status != previous:
                print("작업 상태:", status, flush=True)
                previous = status
            if status in {"failed", "interrupted"}:
                raise RuntimeError(f"번역 실패: {state.get('error', '상세 내용 없음')}")
            if status == "completed":
                with requests.get(root_url + f"/jobs/{job_id}/download",
                                  headers=auth, stream=True, timeout=180) as download:
                    download.raise_for_status()
                    with dest.open("wb") as out:
                        for chunk in download.iter_content(1024 * 1024):
                            if chunk:
                                out.write(chunk)
                if not dest.is_file() or dest.stat().st_size == 0:
                    raise RuntimeError("다운로드 파일이 비어 있습니다")
                print("완료!", dest)
                return dest
            if status not in {"queued", "running"}:
                raise RuntimeError(f"예상하지 못한 작업 상태: {status!r}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"{wait_seconds}초 안에 완료되지 않았습니다. job_id={job_id}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="호스트 MinerU Standard로 PDF를 ZIP으로 변환한 뒤 Docker 번역 API에 제출"
    )
    p.add_argument("pdf", type=Path, help="영어 원본 PDF")
    p.add_argument("--url", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    p.add_argument("--out", type=Path, help="결과 경로. 기본: 원본파일명.ko.pdf")
    p.add_argument("--output-format", choices=("pdf", "markdown"), default="pdf")
    p.add_argument("--keep-mineru-zip", action="store_true", help="변환 중 생성된 ZIP도 원본 PDF 옆에 저장")
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--wait-seconds", type=int, default=7200)
    args = p.parse_args()
    if args.poll_seconds < 1 or args.wait_seconds < 1:
        p.error("poll-seconds와 wait-seconds는 양수여야 합니다")
    suffix = ".ko.pdf" if args.output_format == "pdf" else ".ko.md"
    output = args.out or args.pdf.with_name(args.pdf.stem + suffix)
    try:
        run(args.pdf, api_url=args.url, api_key=os.environ.get("API_KEY", ""),
            output=output, output_format=args.output_format,
            poll_seconds=args.poll_seconds, wait_seconds=args.wait_seconds,
            keep_zip=args.keep_mineru_zip)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError,
            TimeoutError) as exc:
        print("[실패]", exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print("[API 오류]", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
