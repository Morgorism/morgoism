#!/usr/bin/env python3
"""MinerU PDF/ZIP/MD -> translation-safe Markdown + math/table/column QA.

Standard library for ZIP/MD processing; optional PyMuPDF+Pillow for PDF layout inspection.
MinerU CLI is required only for PDF -> ZIP extraction.
No arbitrary OCR/math reconstruction. Uncertain tables use MinerU's original table crop.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from urllib.parse import unquote
import zipfile

# Include the tried-and-tested v1 safe ZIP handling and asset path helpers.
# ---- Standalone MinerU/ZIP helpers from tested v1 converter ----
IMG_MD = re.compile(r'!\[[^\]\n]*\]\(\s*(<[^>]+>|[^)]+?)\s*\)')

IMG_HTML = re.compile(r'<img\b[^>]*\bsrc\s*=\s*[\"\']([^\"\']+)[\"\'][^>]*>', re.I)

HEAD = re.compile(r'^\s{0,3}#{1,6}\s+')

RULE = re.compile(r'^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$')

FENCE = re.compile(r'^\s*(`{3,}|~{3,})')

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def unpack_safely(archive_path: Path, dest: Path):
    """경로 탈출, 링크, 특이 파일을 거부한다."""
    root = dest.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            fn = info.filename.replace('\\', '/')
            rel = Path(fn)
            if rel.is_absolute() or '..' in rel.parts or ':' in fn:
                raise ValueError('ZIP에 안전하지 않은 경로가 있음: ' + fn)
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise ValueError('ZIP 심볼릭 링크는 허용하지 않음: ' + fn)
            if not (root / rel).resolve().is_relative_to(root):
                raise ValueError('ZIP 경로가 대상 폴더 바깥을 가리킴: ' + fn)
        zf.extractall(dest)

def find_markdown(root: Path) -> Path:
    preferred = sorted(root.rglob('markdown.md'))
    candidates = preferred or sorted(
        p for p in root.rglob('*.md') if p.name not in {'paper.md', 'format_report.md'}
    )
    if not candidates:
        raise FileNotFoundError('MinerU 압축/폴더 안에서 markdown.md를 찾지 못함')
    if len(candidates) > 1 and not preferred:
        raise ValueError('Markdown 파일이 여러 개임. 원본 markdown.md 경로를 직접 지정해 줘.')
    return candidates[0]

def original_image_refs(text: str) -> list[str]:
    refs = [m.group(1).strip() for m in IMG_MD.finditer(text)]
    refs += [m.group(1).strip() for m in IMG_HTML.finditer(text)]
    cleaned = []
    for ref in refs:
        # ![](file.jpg "title") 형태의 타이틀만 제거
        if ref.startswith('<') and ref.endswith('>'):
            ref = ref[1:-1]
        else:
            ref = re.sub(r'\s+[\"\'][^\"\']*[\"\']$', '', ref)
        cleaned.append(ref.strip())
    return cleaned

def normalize_markdown(raw: str) -> str:
    """안전한 표기 통일만 한다: 줄바꿈, 독립 수식 구분자. 내용은 번역/추측하지 않는다."""
    text = raw.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff')
    lines = text.split('\n')
    out = []
    active_fence = None
    for line in lines:
        match = FENCE.match(line)
        if match:
            token = match.group(1)
            if active_fence is None:
                active_fence = (token[0], len(token))
            elif token[0] == active_fence[0] and len(token) >= active_fence[1]:
                active_fence = None
            out.append(line)
            continue
        if active_fence is not None:
            out.append(line)
            continue
        stripped = line.strip()
        # 마크다운 수식 렌더러가 인식하도록 LaTeX 구분자만 통일 (수식 본문 미변경)
        if stripped in (r'\[', r'\]'):
            out.append('$$')
        elif stripped.startswith(r'\[') and stripped.endswith(r'\]') and len(stripped) > 4:
            out.extend(['$$', stripped[2:-2].strip(), '$$'])
        else:
            # 코드 스팬(`...`) 안의 문자는 변경하지 않음
            parts = re.split(r'(`+[^`\n]*`+)', line)
            parts = [part if part.startswith('`') else re.sub(r'\\\((.*?)\\\)', r'$\1$', part)
                     for part in parts]
            out.append(''.join(parts))
    return '\n'.join(out).rstrip() + '\n'

def table_width(row: str) -> int:
    """구분자 개수의 대략적인 QA용. 수식 내부 파이프는 오탐 가능."""
    row = row.strip()
    if row.startswith('|'):
        row = row[1:]
    if row.endswith('|'):
        row = row[:-1]
    return len(re.split(r'(?<!\\)\|', row))

def analyze(text: str, source_folder: Path) -> dict:
    lines = text.splitlines()
    warnings = []
    images = []
    missing = []
    for ref in original_image_refs(text):
        if ref.startswith(('http://', 'https://', 'data:')):
            warnings.append('외부 이미지 주소는 로컬 이미지로 복사하지 않음: ' + ref)
            continue
        decoded = unquote(ref)
        path = Path(decoded.replace('\\', '/'))
        if path.is_absolute() or '..' in path.parts or ':' in decoded:
            raise ValueError('안전하지 않은 이미지 참조: ' + ref)
        target = (source_folder / path).resolve()
        if not target.is_relative_to(source_folder.resolve()):
            raise ValueError('폴더 바깥 이미지 참조: ' + ref)
        images.append(ref)
        if not target.is_file():
            missing.append(ref)

    # 표가 Markdown의 파이프 테이블로 추출되었는지 간단히 검사
    table_rows = []
    tables = 0
    old_table = False
    in_fence = None
    display_math_open = False
    for n, line in enumerate(lines, start=1):
        s = line.strip()
        fm = FENCE.match(line)
        if fm:
            if in_fence is None:
                in_fence = (fm.group(1)[0], len(fm.group(1)))
            elif fm.group(1)[0] == in_fence[0] and len(fm.group(1)) >= in_fence[1]:
                in_fence = None
            continue
        if in_fence is not None:
            continue
        if s == '$$':
            display_math_open = not display_math_open
            continue
        if display_math_open:
            continue
        if s.startswith('|') and s.endswith('|') and s.count('|') >= 2:
            table_rows.append((n, table_width(s)))
            cells = re.split(r'(?<!\\)\|', s.strip('|'))
            # OCR가 여러 열을 한 셀에 붙였을 때의 의심 신호 (정확한 판정은 아님)
            if cells and max(len(c.strip()) for c in cells) > 100:
                warnings.append(f'긴 표 셀 발견: {n}행 (OCR 열 합쳐짐 여부 확인)')
            if not old_table:
                tables += 1
            old_table = True
        else:
            old_table = False
    # 인접 테이블 행 중 열 수가 다른 구간을 경고
    for (prev_n, prev_w), (n, w) in zip(table_rows, table_rows[1:]):
        if n == prev_n + 1 and w != prev_w:
            warnings.append(f'표 열 개수가 달라 보임: {prev_n}행({prev_w}열) / {n}행({w}열)')
    if display_math_open:
        warnings.append('닫히지 않은 독립 수식 $$가 의심됨')
    if in_fence is not None:
        warnings.append('닫히지 않은 코드 블록이 의심됨')
    for ref in sorted(set(missing)):
        warnings.append('이미지 파일 누락: ' + ref)
    if tables:
        warnings.append('표 셀의 텍스트는 현재 팀원 Reranker의 markdown_io.py에서 번역하지 않음 (별도 처리 필요)')
    if re.search(r'(?m)^\s*<table\b', text):
        warnings.append('HTML 표가 존재함: 현재 Reranker/출력기와의 호환성 확인 필요')

    return {
        'lines': len(lines),
        'headings': sum(bool(HEAD.match(ln)) for ln in lines),
        'image_references': len(images),
        'local_image_refs': images,
        'missing_images': sorted(set(missing)),
        'table_blocks_estimate': tables,
        'table_rows_estimate': len(table_rows),
        'display_math_markers': len(re.findall(r'(?m)^\s*\$\$\s*$', text)),
        'inline_math_pairs_estimate': len(re.findall(r'(?<!\\)(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)', text)),
        'warnings': list(dict.fromkeys(warnings)),
    }

def copy_assets(source_folder: Path, out: Path, refs: list[str]) -> int:
    copied = set()
    # 일반 MinerU ZIP은 images/ 안에 원본 그림과 테이블 대체 이미지 등을 포함.
    # 참조되지 않은 테이블 이미지도 QA/수동 교정에 필요하므로 보존.
    img_dir = source_folder / 'images'
    if img_dir.is_dir():
        for source in img_dir.rglob('*'):
            if not source.is_file():
                continue
            relative = source.relative_to(source_folder)
            dest = out / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            copied.add(relative.as_posix())
    for ref in refs:
        if ref.startswith(('http://', 'https://', 'data:')):
            continue
        relative = Path(unquote(ref).replace('\\', '/'))
        if relative.is_absolute() or '..' in relative.parts or ':' in ref:
            raise ValueError('이미지 경로 문제: ' + ref)
        source = source_folder / relative
        if not source.is_file():
            continue
        dest = out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        copied.add(relative.as_posix())
    return len(copied)

basic_analyze = analyze
# ---- End v1 helpers ----

TABLE_ROW = re.compile(r'^\s*\|.+\|\s*$')
TABLE_IMAGE = re.compile(r'(?i)(?:^|/)page_(\d+)_table_(\d+)(?:_duplicate_\d+)?\.(?:jpe?g|png|webp)$')
FENCE = re.compile(r'^\s*(`{3,}|~{3,})')
MATH_INLINE = re.compile(r'(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$)')
MATH_SINGLE_DISPLAY = re.compile(r'^\s*\$\$(.+?)\$\$\s*$')
MATH_ENV = re.compile(r'\\(begin|end)\s*\{([^{}]+)\}')
MATH_GLYPHS = {'−': '-', '≤': r'\leq ', '≥': r'\geq ', '≠': r'\neq ', '×': r'\times ', '÷': r'\div ', '∞': r'\infty '}


def load_pdf_lib():
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        from PIL import Image, ImageDraw
        return fitz, Image, ImageDraw
    except ImportError:
        return None, None, None


def issues_in_latex(body: str) -> list[str]:
    issues = []
    # Count only non-escaped structural braces.
    depth = 0
    for m in re.finditer(r'(?<!\\)[{}]', body):
        depth += 1 if m.group() == '{' else -1
        if depth < 0:
            issues.append('닫는 중괄호가 더 많음')
            depth = 0
    if depth:
        issues.append('중괄호 개수가 맞지 않음')
    left = len(re.findall(r'\\left\b', body))
    right = len(re.findall(r'\\right\b', body))
    if left != right:
        issues.append(r'\left / \right 개수가 다름')
    stack = []
    for match in MATH_ENV.finditer(body):
        kind, env = match.groups()
        if kind == 'begin':
            stack.append(env)
        elif not stack or stack.pop() != env:
            issues.append(r'\begin / \end 환경이 일치하지 않음')
    if stack:
        issues.append(r'\begin / \end 환경이 닫히지 않음')
    if re.search(r'\\(?:frac|dfrac|tfrac)\s*\{[^{}]*\}\s*$', body):
        issues.append(r'분수 \frac의 분모가 없는 것으로 보임')
    if '�' in body:
        issues.append('OCR 대체 문자(�) 발견')
    if re.search(r'(?<!\d)\d(?:\s+\d){1,}\b', body):
        issues.append('숫자가 분리되어 추출됐을 가능성: 원본 PDF 확인 필요')
    return list(dict.fromkeys(issues))


def correct_safe_latex(body: str) -> str:
    # Only deterministic encoding fixes. Never synthesize missing variables or operators.
    for before, after in MATH_GLYPHS.items():
        body = body.replace(before, after)
    return body


def process_math(text: str, overrides: dict | None = None) -> tuple[str, list[dict], list[str]]:
    overrides = overrides or {}
    lines = text.splitlines()
    result = []
    records = []
    warnings = []
    active_fence = None
    in_display = False
    block = []
    block_line = None
    n = 0

    def record_math(body: str, line: int, kind: str) -> str:
        nonlocal n
        n += 1
        math_id = f'M{n:04d}'
        before = body
        corrected = correct_safe_latex(body)
        proposed = overrides.get(math_id)
        if proposed is not None:
            if not isinstance(proposed, str):
                raise ValueError(f'{math_id} 교정 내용은 문자열이어야 함')
            corrected = proposed
        findings = issues_in_latex(corrected)
        if before != corrected:
            reason = '사용자 수식 교정' if proposed is not None else '안전한 기호 표기 통일'
        else:
            reason = '변경 없음'
        records.append(dict(id=math_id, kind=kind, line=line, before=before,
                            after=corrected, change=reason, issues=findings))
        for issue in findings:
            warnings.append(f'{math_id} ({line}행): {issue}')
        return corrected

    for line_n, line in enumerate(lines, 1):
        fm = FENCE.match(line)
        if fm:
            if active_fence is None:
                active_fence = (fm.group(1)[0], len(fm.group(1)))
            elif fm.group(1)[0] == active_fence[0] and len(fm.group(1)) >= active_fence[1]:
                active_fence = None
            result.append(line)
            continue
        if active_fence is not None:
            result.append(line)
            continue
        trimmed = line.strip()
        if in_display:
            # Accept either $$ alone or inline closing delimiter. Keep original body intact.
            if trimmed == '$$':
                corrected = record_math('\n'.join(block), block_line, 'display')
                result.extend(corrected.split('\n'))
                result.append('$$')
                in_display = False
                block = []
            else:
                block.append(line)
            continue
        if trimmed == '$$':
            in_display = True
            block = []
            block_line = line_n
            result.append('$$')
            continue
        m = MATH_SINGLE_DISPLAY.match(line)
        if m:
            fixed = record_math(m.group(1), line_n, 'display')
            result.extend(['$$', fixed, '$$'])
            continue
        # Only act on math regions, never on code spans.
        spans = re.split(r'(`+[^`\n]*`+)', line)
        for i, span in enumerate(spans):
            if span.startswith('`'):
                continue
            spans[i] = MATH_INLINE.sub(
                lambda m: '$' + record_math(m.group(1), line_n, 'inline') + '$', span
            )
        result.append(''.join(spans))
    if in_display:
        # Don't change the text of an unterminated display equation.
        result.extend(block)
        warnings.append(f'{block_line}행: 닫히지 않은 독립 수식 $$')
    return '\n'.join(result).rstrip() + '\n', records, warnings


def find_table_blocks(text: str) -> list[dict]:
    lines = text.splitlines()
    blocks = []
    active_fence = None
    in_math = False
    i = 0
    while i < len(lines):
        line = lines[i]
        fm = FENCE.match(line)
        if fm:
            if active_fence is None:
                active_fence = (fm.group(1)[0], len(fm.group(1)))
            elif fm.group(1)[0] == active_fence[0] and len(fm.group(1)) >= active_fence[1]:
                active_fence = None
            i += 1
            continue
        if active_fence:
            i += 1
            continue
        if line.strip() == '$$':
            in_math = not in_math
            i += 1
            continue
        if in_math:
            i += 1
            continue
        if TABLE_ROW.match(line):
            a = i
            while i < len(lines) and TABLE_ROW.match(lines[i]):
                i += 1
            raw = lines[a:i]
            if len(raw) >= 2:
                widths = [table_width(x) for x in raw]
                reasons = []
                if len(set(widths)) > 1:
                    reasons.append('행마다 열 수 다름')
                if any(max(len(cell.strip()) for cell in re.split(r'(?<!\\)\|', x.strip().strip('|'))) > 90 for x in raw):
                    reasons.append('셀에 비정상적으로 긴 텍스트')
                if not re.match(r'^\s*\|(?:\s*:?-{3,}:?\s*\|){2,}\s*$', raw[1]):
                    # GFM separator only following header. Some MinerU tables may have more than one header in block.
                    reasons.append('두 번째 줄의 GFM 표 구분선 없음')
                if '�' in '\n'.join(raw):
                    reasons.append('OCR 대체 문자 포함')
                blocks.append(dict(start=a, end=i, lines=raw, widths=widths,
                                   reasons=reasons, table_no=len(blocks)+1))
            continue
        i += 1
    return blocks


def table_assets(root: Path) -> list[dict]:
    items = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        m = TABLE_IMAGE.search(p.relative_to(root).as_posix())
        if m:
            items.append(dict(page=int(m.group(1)), index=int(m.group(2)),
                              relative=p.relative_to(root).as_posix()))
    return sorted(items, key=lambda x: (x['page'], x['index']))


def process_tables(text: str, output: Path, mode: str) -> tuple[str, list[dict], list[str]]:
    blocks = find_table_blocks(text)
    assets = table_assets(output)
    available_assets = len(assets)
    warnings = []
    lines = text.splitlines()
    changed = []
    table_dir = output / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    associations = {}
    # Global sequential matching only if each extracted block has exactly one source table crop.
    if len(blocks) == len(assets):
        associations = {i: a for i, a in enumerate(assets)}
    elif len(blocks) == 1 and len(assets) == 1:
        associations = {0: assets[0]}

    for index, block in enumerate(blocks):
        number = index + 1
        raw = '\n'.join(block['lines']) + '\n'
        raw_name = f'table_{number:03d}.raw.md'
        (table_dir / raw_name).write_text(raw, encoding='utf-8')
        image = associations.get(index)
        would_replace = (mode == 'image' or (mode == 'auto' and bool(block['reasons'])))
        if would_replace and image is not None:
            replacement = [f'![]({image["relative"]})']
            action = 'table-image-fallback'
            warnings.append(f'표 {number}: 원본 표 이미지로 대체 (표 텍스트는 tables/{raw_name}에 보관)')
        elif would_replace:
            replacement = block['lines']
            action = 'unresolved-table'
            warnings.append(f'표 {number}: 표 구조가 의심스럽지만 안전한 원본 표 이미지 매칭 실패, 원문 유지')
        else:
            replacement = block['lines']
            action = 'markdown-preserved'
        changed.append(dict(number=number, start_line=block['start'] + 1,
                            rows=len(block['lines']), widths=block['widths'],
                            reasons=block['reasons'], action=action,
                            image_ref=image['relative'] if image else None,
                            source_raw=f'tables/{raw_name}'))
        block['replacement'] = replacement
    for block in sorted(blocks, key=lambda x:x['start'], reverse=True):
        lines[block['start']:block['end']] = block['replacement']
    if available_assets != len(blocks) and available_assets:
        warnings.append(f'표 크롭 이미지 {available_assets}개 / 표 블록 {len(blocks)}개: 자동 연결은 숫자가 일치할 때만 허용')
    return '\n'.join(lines).rstrip() + '\n', changed, warnings


def _norm_words(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('-\n', '').replace('\n', ' ')
    return ' '.join(re.findall(r'[a-z0-9]+', s.lower()))


def column_layout(pdf_path: Path, output: Path, make_previews: bool = True) -> dict:
    fitz, PILImage, ImageDraw = load_pdf_lib()
    if fitz is None:
        return {'available': False, 'warning': '2단 분석에는 pymupdf 및 pillow가 필요: python -m pip install pymupdf pillow'}
    pages = []
    order_parts = []
    previews_dir = output / 'layout_previews'
    if make_previews:
        previews_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(pdf_path)) as doc:
        for pn, page in enumerate(doc, start=1):
            w, h = float(page.rect.width), float(page.rect.height)
            mid = w / 2
            blocks = []
            for b in page.get_text('blocks'):
                if len(b) < 7 or b[6] != 0:
                    continue
                x0,y0,x1,y1,t = b[:5]
                text = t.strip()
                if len(text) < 8 or x1 <= x0 or y1 <= y0:
                    continue
                # Exclude side-margin watermarks (e.g. arXiv identifier).
                if x0 < 0.055*w or x1 > 0.945*w:
                    continue
                if x1 < mid - 3:
                    col = 'left'
                elif x0 > mid + 3:
                    col = 'right'
                else:
                    col = 'wide'
                blocks.append(dict(bbox=[round(float(q),1) for q in (x0,y0,x1,y1)],
                                   column=col, text=text))
            left = [b for b in blocks if b['column']=='left']
            right = [b for b in blocks if b['column']=='right']
            left_height = sum(b['bbox'][3]-b['bbox'][1] for b in left)
            right_height = sum(b['bbox'][3]-b['bbox'][1] for b in right)
            twocol = len(left) >= 2 and len(right) >= 2 and left_height > 0.08*h and right_height > 0.08*h
            if twocol:
                body_y = min([b['bbox'][1] for b in left+right])
                headers = sorted([b for b in blocks if b['column']=='wide' and b['bbox'][3] < body_y + 8], key=lambda b:b['bbox'][1])
                ordered = headers + sorted(left,key=lambda b:(b['bbox'][1],b['bbox'][0])) + sorted(right,key=lambda b:(b['bbox'][1],b['bbox'][0]))
                ordered += sorted([b for b in blocks if b['column']=='wide' and b not in headers], key=lambda b:b['bbox'][1])
            else:
                ordered = sorted(blocks,key=lambda b:(b['bbox'][1],b['bbox'][0]))
            for b in ordered:
                order_parts.append(f'[PAGE {pn} / {b["column"]} / bbox {b["bbox"]}]\n{b["text"]}\n')
            page_rec = dict(page=pn, page_size=[round(w,1),round(h,1)],
                            two_column=twocol, left_blocks=len(left),right_blocks=len(right),
                            reading_order=[dict(column=b['column'], bbox=b['bbox'],
                                                text_preview=b['text'][:160]) for b in ordered])
            pages.append(page_rec)
            if make_previews and PILImage is not None:
                scale = 1.25
                pix = page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
                im = PILImage.frombytes('RGB',(pix.width,pix.height),pix.samples)
                if twocol:
                    draw = ImageDraw.Draw(im)
                    # Split marker is for QA only; never inject page design into Markdown.
                    for y in range(0,im.height,18):
                        draw.line([(int(mid*scale),y),(int(mid*scale),min(y+9,im.height))],fill=(230,70,60),width=2)
                im.save(previews_dir / f'page_{pn:03d}.png')
    (output / 'reading_order_reference.txt').write_text('\n'.join(order_parts),encoding='utf-8')
    return {'available':True,'source':str(pdf_path.resolve()),'pages':pages,
            'two_column_pages':[p['page'] for p in pages if p['two_column']],
            'notes':'Markdown은 1차원 읽기 순서로 저장. 원본 PDF의 2단 시각 디자인 복원은 Markdown→PDF 담당 영역.'}


def check_reading_order(layout: dict, md: str) -> list[str]:
    if not layout.get('available'):
        return []
    flat = _norm_words(md)
    anchors = []
    for page in layout['pages']:
        if not page['two_column']:
            continue
        for b in page['reading_order']:
            if b['column'] not in ('left','right'):
                continue
            words = _norm_words(b['text_preview']).split()
            if len(words) < 8:
                continue
            candidate = ' '.join(words[:8])
            pos = flat.find(candidate)
            if pos >= 0 and flat.find(candidate,pos+1) < 0:
                anchors.append((page['page'], b['column'],pos))
    # only compare within a page to avoid cross-page mismatches after paragraphs flow.
    warnings=[]
    for page in layout['two_column_pages']:
        vals = [v for v in anchors if v[0]==page]
        if len(vals) < 3:
            warnings.append(f'{page}페이지: PDF↔Markdown 순서 비교 앵커 부족, 시각 확인 권장')
            continue
        inv = sum(vals[i][2] > vals[i+1][2] for i in range(len(vals)-1))
        if inv:
            warnings.append(f'{page}페이지: 추출 순서 불일치 의심 ({inv}회). layout_previews와 reading_order_reference.txt 확인')
    return warnings


def markdown_qa_report(report: dict) -> str:
    t=report['tables']
    m=report['math']
    layout=report['layout']
    qa=report['qa']
    lines=[
        '# PDF → Markdown 고급 구조 검사', '',
        f'- 원본: `{report["source"]}`',
        f'- 출력: `paper.md`',
        f'- 이미지 참조: {qa["image_references"]}개, 이미지 누락: {len(qa["missing_images"])}개',
        f'- 수식: {len(m)}개, 구문 주의: {sum(bool(x["issues"]) for x in m)}개',
        f'- 표: {len(t)}개, 이미지 대체: {sum(x["action"]=="table-image-fallback" for x in t)}개',
        f'- 2단 페이지: {layout.get("two_column_pages", "원본 PDF 미지정")}','',
        '## 수식 교정', '',
        '수학적 동치성은 자동으로 보증할 수 없음. 확실한 Unicode 기호만 LaTeX 표기로 바꾸고 구문 문제는 검수 목록에 남김.', '',
    ]
    if m:
        for x in m:
            issue=', '.join(x['issues']) or '구조 문제 없음'
            lines.append(f'- `{x["id"]}` ({x["line"]}행, {x["kind"]}): {x["change"]}; {issue}')
    else:
        lines.append('- 발견된 수식 없음')
    lines+=['','## 표 처리', '']
    if t:
        for x in t:
            reason=', '.join(x['reasons']) or '정상 후보'
            lines.append(f'- 표 {x["number"]} ({x["start_line"]}행): **{x["action"]}**; {reason}; 원문 `tables/table_{x["number"]:03d}.raw.md`')
    else: lines.append('- 인식된 파이프 표 없음')
    lines+=['','## 2단 레이아웃', '']
    if layout.get('available'):
        lines.append('- 2단 PDF는 왼쪽 열 → 오른쪽 열 읽기 순서를 참고자료로 저장함. Markdown 자체는 다시 2단으로 조판하지 않음.')
        lines.append('- 페이지 미리보기: `layout_previews/`, 텍스트 참고: `reading_order_reference.txt`, 좌표: `layout.json`')
    else:
        lines.append('- 원본 PDF를 제공하면 2단 판독 검사 및 페이지 미리보기 생성 가능 (`--original-pdf`)')
    lines+=['','## 확인해야 할 문제', '']
    lines+=['- '+s for s in report['warnings']] or ['- 자동 구조 검사 경고 없음 (정확도 보증은 아님)']
    lines+=['','## 최종 합의 규격', '',
            '- 제목/본문: UTF-8 Markdown, 2단 입력은 읽기 순서로 풀어 저장',
            '- 수식: 인라인 `$...$`, 독립 `$$` 블록, 원칙적으로 LaTeX 유지',
            '- 표: 정상은 파이프 Markdown, 파손은 표 크롭 이미지 (원본 추출 텍스트 별도 보관)',
            '- 그림/표 이미지: 상대경로 images/, 반드시 폴더와 함께 전달',
            '- 병합 셀/수학적으로 잘못 읽힌 글자/정확한 2단 지면은 사람의 검증 또는 출력단의 별도 구현 필요', '']
    return '\n'.join(lines)


def process(source: Path, out: Path, pdf_ref: Path | None, tier: str, pages: str | None,
            mode: str, overrides_path: Path | None, preview: bool) -> dict:
    if not source.is_file(): raise FileNotFoundError(source)
    if pdf_ref and not pdf_ref.is_file(): raise FileNotFoundError(pdf_ref)
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='md_upgrade_') as temp:
        temp_dir=Path(temp)
        ext=source.suffix.lower()
        if ext=='.pdf':
            pdf_ref=source
            executable=shutil.which('mineru-kit')
            if not executable:
                raise RuntimeError('mineru-kit을 찾을 수 없음 (conda activate mineru-env 확인)')
            zip_path=temp_dir/'parsed.zip'
            cmd=[executable,'parse',str(source.resolve()),'-o',str(zip_path),
                 '--format','zip','--tier',tier]
            if pages: cmd+=['--pages',pages]
            print('MinerU Standard 실행 중...', flush=True)
            subprocess.run(cmd,check=True)
            if not zip_path.exists(): raise FileNotFoundError(zip_path)
            extracted=temp_dir/'input';extracted.mkdir()
            unpack_safely(zip_path,extracted)
            md=find_markdown(extracted)
        elif ext=='.zip':
            extracted=temp_dir/'input'; extracted.mkdir()
            unpack_safely(source,extracted)
            md=find_markdown(extracted)
        elif ext=='.md':
            md=source
        else:
            raise ValueError('PDF/ZIP/MD 파일만 지원')
        print('기존 MinerU 추출물 및 그림 복사 중...', flush=True)
        # Read and normalize *before* any other transformations.
        raw=md.read_text(encoding='utf-8-sig')
        text=normalize_markdown(raw)
        copied=copy_assets(md.parent,out,original_image_refs(text))
        overrides={}
        if overrides_path:
            overrides=json.loads(overrides_path.read_text(encoding='utf-8-sig'))
            if not isinstance(overrides,dict): raise ValueError('수식 overrides는 JSON 객체여야 함')
        text,math,math_warnings=process_math(text,overrides)
        text,tables,table_warnings=process_tables(text,out,mode)
        # Report final paper.md lines, not intermediate MinerU line numbers.
        math_pos = 0
        for rec in math:
            rec['mineru_line'] = rec['line']
            needle = ('$' + rec['after'] + '$') if rec['kind'] == 'inline' else rec['after']
            offset = text.find(needle, math_pos)
            if offset < 0:
                offset = text.find(needle)
            if offset >= 0:
                rec['line'] = text.count('\n', 0, offset) + 1
                math_pos = offset + len(needle)
            else:
                rec['line'] = None
                math_warnings.append(f"{rec['id']}: 최종 Markdown에서 수식 위치 추적 실패")
        # Match warning line numbers to the final paper.md positions.
        for rec in math:
            if rec['line'] is None:
                continue
            old_tag = f"{rec['id']} ({rec['mineru_line']}행)"
            new_tag = f"{rec['id']} ({rec['line']}행)"
            math_warnings = [w.replace(old_tag, new_tag, 1) for w in math_warnings]
        # Warn when a comparison threshold seems to have escaped an inline equation.
        if re.search(r'\$[^$\n]*?\{\s*>\s*\}[^$\n]*\$\s+\d+', text):
            math_warnings.append('비교 연산자의 임계값 숫자가 수식 밖에 분리됐을 가능성: 원본 대조 필요')
        (out/'paper.md').write_text(text,encoding='utf-8')
        qa=basic_analyze(text,out)
        layout=column_layout(pdf_ref,out,preview) if pdf_ref else {
            'available':False,
            'warning':'원본 PDF 경로 미지정. --original-pdf 로 2단 분석 및 페이지 미리보기를 활성화 가능',
        }
        reading_warnings=check_reading_order(layout,text)
        warnings=list(dict.fromkeys(math_warnings+table_warnings+reading_warnings+qa['warnings']))
        if not layout.get('available'): warnings.append(layout['warning'])
        report=dict(version='md-contract-v2', source=str(source.resolve()),
                    output=str(out.resolve()), source_sha256=sha256_bytes(source.read_bytes()),
                    copied_assets=copied, table_mode=mode,qa=qa,math=math,
                    tables=tables,layout=layout,warnings=warnings)
        (out/'math_review.json').write_text(json.dumps(math,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'format_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'format_report.md').write_text(markdown_qa_report(report),encoding='utf-8')
        print('완료:',out/'paper.md')
        print('수식:',len(math),'| 표:',len(tables),'| 표 이미지 대체:',sum(t['action']=='table-image-fallback' for t in tables))
        print('2단 페이지:',layout.get('two_column_pages','PDF 미제공'))
        print('검사 보고서:',out/'format_report.md')
        return report


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description='PDF→Markdown: 수식/표/2단 검수')
    ap.add_argument('source',type=Path,help='PDF, MinerU ZIP, markdown.md')
    ap.add_argument('--name',type=str,help='출력 폴더명')
    ap.add_argument('--out',type=Path,help='출력 폴더')
    ap.add_argument('--original-pdf',type=Path,help='기존 ZIP에 해당하는 원본 PDF (2단 레이아웃 검사)')
    ap.add_argument('--tier',default='standard',choices=['standard'])
    ap.add_argument('--pages',type=str,help='MinerU 분석할 페이지 범위 (예: 1-2)')
    ap.add_argument('--table-mode',choices=['auto','markdown','image'],default='auto',help='표 처리 정책')
    ap.add_argument('--math-overrides',type=Path,help='수식 ID별 직접 교정 JSON 파일(선택)')
    ap.add_argument('--no-previews',action='store_true',help='PDF 페이지 미리보기 생성 안 함')
    a=ap.parse_args(argv)
    out=a.out or (Path('output_md_advanced')/(a.name or a.source.stem))
    try:
        process(a.source,out,a.original_pdf,a.tier,a.pages,a.table_mode,a.math_overrides,not a.no_previews)
    except (OSError,ValueError,RuntimeError,zipfile.BadZipFile,subprocess.CalledProcessError) as e:
        print('[오류]',e,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
