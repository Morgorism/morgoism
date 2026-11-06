"""Example API client. Requires: pip install requests

Examples:
  python deployment/client.py text --text "The algorithm converges."
  python deployment/client.py paper examples/mineru_result.zip --output-format markdown
"""
import argparse
import os
from pathlib import Path
import time
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("API_URL", "http://127.0.0.1:8000"),
                        help="Example: http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    sub = parser.add_subparsers(dest="cmd", required=True)
    text = sub.add_parser("text")
    text.add_argument("--text", required=True)
    paper = sub.add_parser("paper")
    paper.add_argument("path", type=Path)
    paper.add_argument("--output-format", choices=["pdf", "markdown"], default="pdf")
    args = parser.parse_args()
    if not args.url or not args.api_key:
        parser.error("Set --url and --api-key (or API_URL and API_KEY environment variables)")
    url = args.url.rstrip("/")
    headers = {"X-API-Key": args.api_key}
    if args.cmd == "text":
        r = requests.post(url + "/translate", headers=headers, json={"text": args.text}, timeout=1800)
        r.raise_for_status()
        print(r.json()["translation"])
        return
    with args.path.open("rb") as f:
        r = requests.post(url + "/jobs", params={"output_format": args.output_format},
                          headers=headers, files={"file": (args.path.name, f)}, timeout=60)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print("Job:", job_id)
    while True:
        time.sleep(10)
        r = requests.get(url + "/jobs/" + job_id, headers=headers, timeout=20)
        r.raise_for_status()
        info = r.json()
        print("Status:", info["status"])
        if info["status"] in {"failed", "interrupted"}:
            raise RuntimeError(info.get("error") or "Pipeline failed")
        if info["status"] == "completed":
            break
    r = requests.get(url + "/jobs/" + job_id + "/download", headers=headers, timeout=180)
    r.raise_for_status()
    ext = ".pdf" if args.output_format == "pdf" else ".ko.md"
    output = args.path.with_name(args.path.stem + ext)
    output.write_bytes(r.content)
    print("Saved:", output)


if __name__ == "__main__":
    main()
