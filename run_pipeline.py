#!/usr/bin/env python3
"""Single-command PDF -> Markdown -> Stage4+E_all Reranker -> Markdown -> PDF.

This is an orchestration and structural-protection layer over the teammates'
original components. It does not train, download or simulate neural weights.
'--mode demo' is explicitly labelled and never claimed as actual translation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
CONVERTER = ROOT / "components" / "pdf_to_markdown" / "make_paper_md.py"
RENDERER = ROOT / "components" / "markdown_to_pdf" / "pdf.py"
RERANKER = ROOT / "components" / "horizon-reranker"
GLOSSARY = ROOT / "components" / "glossary_curated.json"
MARKER = re.compile(r"⟦PIPE[A-Z0-9_]+⟧")
IMAGE = re.compile(r"!\[[^\]\n]*\]\(\s*(?:<[^>\n]+>|[^)\n]+)\s*\)")
IMG_HTML = re.compile(r"<img\b[^>]*>", re.I)
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
HTML_TABLE_OPEN = re.compile(r"^\s*<table\b", re.I)
HTML_TABLE_CLOSE = re.compile(r"</table\s*>", re.I)
# Keep specific syntax in place, don't let the model rewrite the syntax/paths.
INLINE = re.compile(
    r"!\[[^\]\n]*\]\(\s*(?:<[^>\n]+>|[^)\n]+)\s*\)"
    r"|\[[^\]\n]+\]\([^\n)]+\)"
    r"|(?<!\\)\$(?!\$)[^$\n]+?(?<!\\)\$(?!\$)"
    r"|\\\([^\n]*?\\\)"
    r"|`+[^`\n]+`+"
    r"|https?://[^\s<>]+"
    r"|\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]"
    r"|</?[A-Za-z][^>\n]*>"
)


class PipelineError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def execute(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log("$ " + " ".join(str(x) for x in command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def token(number: int, kind: str) -> str:
    return f"⟦PIPE{kind}{number:06d}⟧"


def protect_markdown(text: str) -> tuple[str, dict[str, str], dict[str, int]]:
    """Prevent model calls on structural blocks and protect inline math/links.

    Raw blocks are represented by Markdown HTML comments that teammate's
    markdown_io.parse explicitly treats as 'raw'. Inline spans use the exact
    placeholder style recognised by the team's evaluation/structure.py.
    """
    if "⟦PIPE" in text or "<!--PIPERAW" in text:
        raise PipelineError("원본 Markdown에 내부 보호 표시와 충돌하는 문자열이 있습니다.")
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    replacements: dict[str, str] = {}
    kinds: Counter[str] = Counter()
    num = 0
    i = 0

    def remember(fragment: str, kind: str, block: bool = False) -> str:
        nonlocal num
        num += 1
        key = token(num, kind)
        replacements[key] = fragment
        kinds[kind] += 1
        return f"<!--PIPERAW{key}-->" if block else key

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            out.append(line)
            i += 1
            continue

        # YAML front matter must not be translated.
        if i == 0 and s == "---":
            j = i + 1
            while j < len(lines) and lines[j].strip() not in {"---", "..."}:
                j += 1
            if j < len(lines):
                j += 1
            out.append(remember("".join(lines[i:j]), "META", block=True) + "\n")
            i = j
            continue

        # Code fences; also support ~~~ which teammate's markdown parser doesn't.
        m = FENCE.match(line)
        if m:
            char, count = m.group(1)[0], len(m.group(1))
            j = i + 1
            while j < len(lines):
                c = FENCE.match(lines[j])
                if c and c.group(1)[0] == char and len(c.group(1)) >= count and not lines[j][c.end():].strip():
                    j += 1
                    break
                j += 1
            out.append(remember("".join(lines[i:j]), "CODE", block=True) + "\n")
            i = j
            continue

        # Display LaTeX; protect complete block, regardless of its syntax.
        if s.startswith("$$") or s.startswith(r"\["):
            end = "$$" if s.startswith("$$") else r"\]"
            same_line = (end == "$$" and s.count("$$") >= 2) or (end == r"\]" and r"\]" in s)
            j = i + 1
            if not same_line:
                while j < len(lines):
                    if end in lines[j]:
                        j += 1
                        break
                    j += 1
            out.append(remember("".join(lines[i:j]), "MATH", block=True) + "\n")
            i = j
            continue

        # Multiline HTML tables and their math must be passed through exactly.
        if HTML_TABLE_OPEN.match(line):
            j = i + 1
            if not HTML_TABLE_CLOSE.search(line):
                while j < len(lines):
                    if HTML_TABLE_CLOSE.search(lines[j]):
                        j += 1
                        break
                    j += 1
            out.append(remember("".join(lines[i:j]), "HTMLTABLE", block=True) + "\n")
            i = j
            continue

        # Pipe table rows: raw already in teammate's parser; leave unchanged.
        if s.startswith("|") and s.endswith("|"):
            out.append(line)
            i += 1
            continue

        # HTML/image-only lines are not translation sentences.
        if (IMAGE.fullmatch(s) or IMG_HTML.fullmatch(s) or s.startswith("<!--")
                or re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", s)):
            out.append(remember(line, "RAW", block=True) + "\n")
            i += 1
            continue

        # Preserve link targets, inline math, code, URL, citation, etc.
        body = INLINE.sub(lambda match: remember(match.group(0), "INLINE"), line)
        out.append(body)
        i += 1
    return "".join(out), replacements, dict(kinds)


def restore_markdown(translated: str, replacements: dict[str, str]) -> str:
    text = translated
    # Restore raw first; each placeholder must exist exactly once.
    for key, original in replacements.items():
        raw = f"<!--PIPERAW{key}-->"
        if raw in text:
            if text.count(raw) != 1:
                raise PipelineError(f"원본 블록 위치가 중복됐습니다: {key}")
            text = text.replace(raw, original.rstrip("\r\n"), 1)
            continue
        if text.count(key) != 1:
            raise PipelineError(f"번역 모델이 보호 표시를 훼손했습니다: {key}")
        text = text.replace(key, original, 1)
    if "⟦PIPE" in text or "<!--PIPERAW" in text:
        raise PipelineError("번역 결과에 복원되지 않은 보호 표시가 남아 있습니다.")
    return text.rstrip() + "\n"


def local_image_refs(markdown: str) -> list[str]:
    paths: list[str] = []
    for m in IMAGE.finditer(markdown):
        value = m.group(0)
        inner = value[value.index("](") + 2:-1].strip()
        if inner.startswith("<") and inner.endswith(">"):
            inner = inner[1:-1]
        else:
            inner = re.sub(r'\s+["\'][^"\']*["\']$', '', inner)
        paths.append(unquote(inner))
    for m in re.finditer(r'<img\b[^>]*\bsrc=["\']([^"\']+)["\']', markdown, re.I):
        paths.append(unquote(m.group(1)))
    return paths


def verify_document(source_md: str, final_md: str, doc_root: Path) -> dict[str, Any]:
    old_paths = Counter(local_image_refs(source_md))
    new_paths = Counter(local_image_refs(final_md))
    if old_paths != new_paths:
        raise PipelineError(f"이미지 참조가 변형됐습니다: 전 {dict(old_paths)} / 후 {dict(new_paths)}")
    missing = []
    for image in new_paths:
        if image.startswith(("http://", "https://", "data:")):
            continue
        relative = Path(image.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in image:
            raise PipelineError("안전하지 않은 이미지 경로: " + image)
        path = (doc_root / relative).resolve()
        if not path.is_relative_to(doc_root.resolve()) or not path.is_file():
            missing.append(image)
    if missing:
        raise PipelineError("이미지 누락: " + ", ".join(missing))
    # Teammate's markdown_io leaves pipe tables raw. Verify that remains true.
    pipe = lambda content: [line for line in content.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
    if pipe(source_md) != pipe(final_md):
        raise PipelineError("Markdown 표의 행이나 값이 변경됐습니다.")
    return {"source_image_references": sum(old_paths.values()),
            "image_missing": missing, "pipe_table_rows": len(pipe(source_md)),
            "source_characters": len(source_md), "final_characters": len(final_md)}


def mark_mock_translation(prepared: str, repo: Path) -> tuple[str, int]:
    # Exercise the team's REAL markdown parser/render API without loading a GPU model.
    sys.path.insert(0, str(repo))
    try:
        from reranker.markdown_io import parse, render
        blocks, units = parse(prepared)
        if not units:
            return render(blocks, []), 0
        result = render(blocks, ["【데모: 실제 AI 번역 아님】 " + unit for unit in units])
        return result, len(units)
    finally:
        sys.path.pop(0)


def resolve_existing(path: Path, message: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PipelineError(message + ": " + str(path))
    return path


def check_real_models(config: Path, repo: Path, weights: Path, glossary: Path) -> dict[str, Any]:
    config = resolve_existing(config, "Reranker 설정 파일 없음")
    resolve_existing(weights, "Reranker 학습 가중치 JSON 없음")
    resolve_existing(glossary, "용어집 없음")
    if not (repo / "reranker" / "__main__.py").is_file():
        raise PipelineError("팀원 Reranker 저장소 위치가 올바르지 않습니다: " + str(repo))
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    w = json.loads(weights.read_text(encoding="utf-8-sig"))
    if "E_all" not in w or not isinstance(w["E_all"].get("weights"), dict):
        raise PipelineError("weights.json에 E_all 최종 재순위화 가중치가 없습니다.")
    adapters = {}
    for name in ("adapter", "backward_adapter"):
        val = cfg.get(name)
        if not val:
            raise PipelineError(f"config에서 {name} 모델 경로가 비었습니다.")
        d = Path(val).expanduser().resolve()
        if not (d / "adapter_config.json").is_file():
            raise PipelineError(f"{name}의 adapter_config.json이 없습니다: {d}\n"
                                "RunPod에서 팀원 setup_runpod.sh 또는 실제 모델 체크포인트 준비 후 실행하세요.")
        weights_file = d / "adapter_model.safetensors"
        if not weights_file.is_file() or weights_file.stat().st_size < 1_000_000:
            raise PipelineError(f"{name} 가중치가 없거나 Git LFS 포인터입니다: {weights_file}")
        adapters[name] = str(d)
    return {"config": str(config), "weights": str(weights), "glossary": str(glossary), "adapters": adapters}


def run_pipeline(args: argparse.Namespace) -> Path:
    source = resolve_existing(Path(args.input), "입력 PDF/ZIP/Markdown을 찾을 수 없음")
    if source.suffix.lower() not in {".pdf", ".zip", ".md"}:
        raise PipelineError("입력은 .pdf / MinerU .zip / .md만 지원합니다.")

    name = args.name or source.stem
    if not re.fullmatch(r"[a-zA-Z0-9가-힣_.-]+", name) or name in {".", ".."}:
        raise PipelineError("--name은 영문/한글/숫자/밑줄/하이픈으로만 지정하세요.")
    output_root = Path(args.out).expanduser().resolve() if args.out else (ROOT / "output" / name)
    doc_root = output_root / "document"
    doc_root.mkdir(parents=True, exist_ok=True)
    repo = Path(args.reranker_repo).expanduser().resolve() if args.reranker_repo else RERANKER
    if not (repo / "reranker" / "markdown_io.py").is_file():
        raise PipelineError("Reranker 소스 코드가 없습니다: " + str(repo))
    config = Path(args.config).expanduser().resolve() if args.config else (
        (repo / "rerank_config.json") if (repo / "rerank_config.json").exists() else (repo / "configs" / "final.json"))
    weights = Path(args.weights).expanduser().resolve() if args.weights else repo / "results" / "final" / "weights.json"
    glossary = Path(args.glossary).expanduser().resolve() if args.glossary else GLOSSARY
    md_python = str(Path(args.md_python).expanduser()) if args.md_python else sys.executable
    reranker_python = str(Path(args.reranker_python).expanduser()) if args.reranker_python else sys.executable
    pdf_python = str(Path(args.pdf_python).expanduser()) if args.pdf_python else sys.executable

    # Fail fast before expensive MinerU processing for real model mode.
    model_info = None
    if args.mode == "real":
        model_info = check_real_models(config, repo, weights, glossary)
    if args.renderer == "real":
        if not shutil.which("pandoc") or not shutil.which("xelatex"):
            raise PipelineError("Pandoc와 XeLaTeX가 필요합니다. --renderer skip으로 Markdown까지만 테스트할 수 있습니다.")

    # Never let old outputs masquerade as a successful new run.
    for stale in (doc_root / "paper.md", doc_root / "paper.translated_guarded.md",
                  doc_root / "paper.ko.md", output_root / (name + (".DEMO.pdf" if args.mode == "demo" else ".ko.pdf"))):
        if stale.is_file():
            stale.unlink()

    log("\n[1/4] 팀원 PDF→Markdown 변환기 실행")
    cmd = [md_python, str(CONVERTER), str(source), "--out", str(doc_root),
           "--table-mode", args.table_mode]
    if args.original_pdf:
        cmd.extend(["--original-pdf", str(Path(args.original_pdf).expanduser().resolve())])
    if args.no_previews:
        cmd.append("--no-previews")
    env = os.environ.copy()
    # On RunPod pdf and Reranker can have different Python environments.
    exe_folder = str(Path(md_python).parent)
    if exe_folder not in {"", "."}:
        env["PATH"] = exe_folder + os.pathsep + env.get("PATH", "")
    execute(cmd, env=env)

    paper = resolve_existing(doc_root / "paper.md", "PDF→Markdown 결과 없음")
    source_text = paper.read_text(encoding="utf-8-sig")
    guarded, replacements, stats = protect_markdown(source_text)
    guarded_path = doc_root / "paper.for_translation.md"
    guarded_path.write_text(guarded, encoding="utf-8")
    (doc_root / "protection_map.json").write_text(
        json.dumps(replacements, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n[2/4] 번역 Stage4 + Reranker E_all (" + args.mode + ")")
    translated_guarded = doc_root / "paper.translated_guarded.md"
    n_units = None
    if args.mode == "real":
        command = [reranker_python, "-m", "reranker", "translate", "--config", str(config),
                   "--glossary", str(glossary), "--weights", str(weights),
                   "--experiment", "E_all", "--input", str(guarded_path),
                   "--out", str(translated_guarded)]
        execute(command, cwd=repo)
        resolve_existing(translated_guarded, "Reranker 번역 결과 없음")
        guarded_ko = translated_guarded.read_text(encoding="utf-8-sig")
    else:
        guarded_ko, n_units = mark_mock_translation(guarded, repo)
        translated_guarded.write_text(guarded_ko, encoding="utf-8")
    if not guarded_ko.strip():
        raise PipelineError("번역 단계 결과가 비어 있습니다.")
    if args.mode == "real" and not re.search(r"[가-힣]", guarded_ko):
        raise PipelineError("Reranker 출력에 한국어가 없습니다. 실제 번역 결과를 확인하세요.")

    log("\n[3/4] 수식·그림·표 복원 및 검증")
    restored = restore_markdown(guarded_ko, replacements)
    check = verify_document(source_text, restored, doc_root)
    translated_path = doc_root / "paper.ko.md"
    translated_path.write_text(restored, encoding="utf-8")
    if args.mode == "demo":
        log("[주의] DEMO: 팀원 AI 모델은 호출하지 않았으며 실제 한국어 번역이 아닙니다.")

    pdf_path = output_root / (name + (".DEMO.pdf" if args.mode == "demo" else ".ko.pdf"))
    log("\n[4/4] 팀원 Markdown→PDF 출력기 (" + args.renderer + ")")
    if args.renderer == "real":
        cmd = [pdf_python, str(RENDERER), str(translated_path), str(pdf_path)]
        if args.main_font:
            cmd += ["--main-font", args.main_font]
        if args.sans_font:
            cmd += ["--sans-font", args.sans_font]
        execute(cmd)
        resolve_existing(pdf_path, "최종 PDF 파일 없음")
        if pdf_path.stat().st_size < 500:
            raise PipelineError("생성된 PDF 파일이 비정상적으로 작습니다.")
    else:
        log("PDF 생성 생략 (--renderer skip)")

    report = {
        "pipeline": "cs-paper-final-integration",
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode, "actually_ran_neural_translation": args.mode == "real",
        "renderer": args.renderer, "input": str(source),
        "paper_md": str(paper), "translated_md": str(translated_path),
        "output_pdf": str(pdf_path) if args.renderer == "real" else None,
        "units_in_demo": n_units, "protected": stats,
        "model_info": model_info, "verification": check,
        "limitations": [
            "원본 2단 지면과 동일한 PDF 레이아웃으로 출력하지는 않습니다.",
            "Markdown 표는 구조 보존되며 표 내부 영어는 팀원 Reranker에서 번역하지 않습니다.",
            "PDF 수식의 수학적 OCR 정확도는 사람 검증이 필요합니다.",
        ]
    }
    (output_root / "pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n========== 파이프라인 완료 ==========")
    log("원본 Markdown: " + str(paper))
    log("최종 Markdown: " + str(translated_path))
    log("최종 PDF: " + (str(pdf_path) if args.renderer == "real" else "생략"))
    log("보고서: " + str(output_root / "pipeline_report.json"))
    return pdf_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="팀 전체 PDF→Stage4+E_all Reranker→PDF 통합")
    p.add_argument("--input", required=True, help="입력 .pdf / MinerU .zip / .md")
    p.add_argument("--original-pdf", help="ZIP을 입력하는 경우 원본 PDF 경로")
    p.add_argument("--name", help="결과 폴더명")
    p.add_argument("--out", help="결과 폴더(기본 output/<입력파일명>)")
    p.add_argument("--mode", choices=["real", "demo"], default="real",
                   help="real: 실제 Stage4+E_all, demo: GPU 없이 연결 테스트 (실제 번역 아님)")
    p.add_argument("--renderer", choices=["real", "skip"], default="real")
    p.add_argument("--reranker-repo", help="팀원 reranker 폴더(기본: 포함된 소스)")
    p.add_argument("--config", help="실제 모델 경로가 들어 있는 rerank_config.json")
    p.add_argument("--glossary", help="용어집 JSON(기본: 포함된 팀원 용어집)")
    p.add_argument("--weights", help="최종 E_all weights.json")
    p.add_argument("--md-python", help="MinerU 환경의 Python 경로")
    p.add_argument("--reranker-python", help="GPU/모델 환경의 Python 경로")
    p.add_argument("--pdf-python", help="Pandoc/XeLaTeX 실행 환경의 Python 경로")
    p.add_argument("--table-mode", choices=["auto", "markdown", "image"], default="auto")
    p.add_argument("--no-previews", action="store_true")
    p.add_argument("--main-font", help="XeLaTeX 한글 글꼴: Linux=Noto Serif CJK KR")
    p.add_argument("--sans-font", help="XeLaTeX 한글 글꼴: Linux=Noto Sans CJK KR")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        run_pipeline(a)
    except (PipelineError, OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
        log("\n[실행 실패] " + str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
