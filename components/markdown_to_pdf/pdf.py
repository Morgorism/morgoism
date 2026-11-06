"""Markdown -> PDF renderer (Pandoc + XeLaTeX).

Requirements
------------
- `pandoc` and `xelatex` available on PATH
- A Korean font installed (Windows default: Malgun Gothic)

Usage
-----
    python pdf_renderer.py input.md output.pdf

Or from Python:
    PDFRenderer().render(Path('input.md'), Path('output.pdf'))
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

class PDFRenderer:
    """Render Markdown files as PDFs with Korean text and HTML tables."""

    # Protect math inside HTML tables while Pandoc converts HTML to LaTeX.
    _MATH_PATTERN = re.compile(
        r"(?<!\\)\$\$.*?(?<!\\)\$\$"
        r"|\\\[.*?\\\]"
        r"|(?<!\\)\$(?!\$)(?:\\.|[^$\\\n])+\$(?!\$)"
        r"|\\\(.*?\\\)",
        flags=re.DOTALL,
    )
    _TABLE_PATTERN = re.compile(
        r"<table\b[^>]*>.*?</table>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    _FENCE_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")

    def __init__(self, main_font: str | None = None, sans_font: str | None = None):
        # Malgun Gothic is included in standard Korean-language Windows installations.
        # On Linux/macOS, pass fonts that are installed on your system if necessary.
        windows = platform.system() == "Windows"
        self.main_font = main_font or (
            "Malgun Gothic" if windows else "Noto Serif CJK KR"
        )
        self.sans_font = sans_font or (
            "Malgun Gothic" if windows else "Noto Sans CJK KR"
        )

    def render(self, input_path: Path, output_path: Path) -> Path:
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Markdown 파일을 찾을 수 없습니다: {input_path}")

        self._check_dependencies()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        markdown = input_path.read_text(encoding="utf-8-sig")
        markdown = self._remove_control_characters(markdown)
        markdown = self._convert_html_tables(markdown)

        # Raw LaTeX emitted for HTML tables commonly uses longtable/booktabs.
        # Load these even when Pandoc did not find any native Markdown tables.
        preamble = r"""\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{calc}
\usepackage{multirow}
% Pandoc >= 3.8 may emit \LTcaptype=none for uncaptioned longtables.
% Define the dummy counter required by some TeX installations.
\newcounter{none}
\hyphenpenalty=10000
\exhyphenpenalty=10000
\emergencystretch=3em
"""

        with tempfile.TemporaryDirectory(prefix="pdf_renderer_") as temp_dir:
            header_path = Path(temp_dir) / "header.tex"
            header_path.write_text(preamble, encoding="utf-8")

            command = [
                "pandoc",
                "-",
                "-f", "markdown+raw_attribute+raw_tex",
                "-o", str(output_path),
                "--pdf-engine=xelatex",
                "--standalone",
                "-V", "documentclass=article",
                "-V", f"mainfont={self.main_font}",
                "-V", f"sansfont={self.sans_font}",
                "-V", "geometry:margin=1in",
                "-H", str(header_path),
            ]

            try:
                subprocess.run(
                    command,
                    check=True,
                    cwd=input_path.parent,
                    input=markdown,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                details = (exc.stderr or exc.stdout or str(exc)).strip()
                raise RuntimeError(
                    "PDF 변환 중 오류가 발생했습니다. "
                    "MiKTeX 패키지 설치 상태와 폰트 이름을 확인하세요.\n"
                    f"{details}"
                ) from exc

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"PDF 파일이 정상적으로 생성되지 않았습니다: {output_path}")

        return output_path

    @staticmethod
    def _check_dependencies() -> None:
        missing = [name for name in ("pandoc", "xelatex") if shutil.which(name) is None]
        if missing:
            raise RuntimeError(
                "필수 실행 파일을 찾을 수 없습니다: " + ", ".join(missing)
                + ". 설치한 뒤 터미널/VS Code를 재시작하고 PATH를 확인하세요."
            )

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        # Preserve newlines/tabs. Remove non-printing Unicode control chars.
        return "".join(
            ch for ch in text
            if ch in "\n\r\t" or unicodedata.category(ch) != "Cc"
        )

    @classmethod
    def _protect_table_math(cls, html_table: str) -> tuple[str, dict[str, str]]:
        protected: dict[str, str] = {}

        def replace_math(match: re.Match[str]) -> str:
            raw_math = match.group(0)
            # Display math cannot be placed directly inside a LaTeX table cell.
            # Render it in display style, but keep it in inline math mode.
            if raw_math.startswith("$$"):
                raw_math = "$\\displaystyle " + raw_math[2:-2].strip() + "$"
            elif raw_math.startswith(r"\["):
                raw_math = "$\\displaystyle " + raw_math[2:-2].strip() + "$"

            placeholder = f"MATHPLACEHOLDER{len(protected):06d}TOKEN"
            protected[placeholder] = raw_math
            return placeholder

        return cls._MATH_PATTERN.sub(replace_math, html_table), protected

    def _convert_html_tables(self, text: str) -> str:
        def replace_table(match: re.Match[str]) -> str:
            html_table = match.group(0)
            protected_table, protected_math = self._protect_table_math(html_table)

            try:
                result = subprocess.run(
                    ["pandoc", "-f", "html", "-t", "latex"],
                    input=protected_table,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "HTML 표를 LaTeX로 변환하지 못했습니다:\n"
                    + (exc.stderr or str(exc))
                ) from exc

            latex = result.stdout.strip()
            for placeholder, original_math in protected_math.items():
                latex = latex.replace(placeholder, original_math)

            column_count = self._count_table_columns(html_table)
            if column_count >= 7:
                latex = (
                    "\\begingroup\n"
                    "\\scriptsize\n"
                    "\\setlength{\\tabcolsep}{2pt}\n"
                    "\\renewcommand{\\arraystretch}{0.95}\n"
                    + latex
                    + "\n\\endgroup"
                )
            elif column_count >= 5:
                latex = (
                    "\\begingroup\n"
                    "\\footnotesize\n"
                    "\\setlength{\\tabcolsep}{3pt}\n"
                    + latex
                    + "\n\\endgroup"
                )

            return "\n\n```{=latex}\n" + latex + "\n```\n\n"

        converted: list[str] = []
        outside_fence: list[str] = []
        active_fence: tuple[str, int] | None = None

        def flush_outside() -> None:
            if outside_fence:
                converted.append(
                    self._TABLE_PATTERN.sub(replace_table, "".join(outside_fence))
                )
                outside_fence.clear()

        for line in text.splitlines(keepends=True):
            fence_match = self._FENCE_PATTERN.match(line)

            if active_fence is None:
                if fence_match:
                    flush_outside()
                    fence = fence_match.group(1)
                    active_fence = (fence[0], len(fence))
                    converted.append(line)
                else:
                    outside_fence.append(line)
            else:
                if fence_match:
                    fence = fence_match.group(1)
                    # Closing fences may contain trailing whitespace, but no info string.
                    if (
                        fence[0] == active_fence[0]
                        and len(fence) >= active_fence[1]
                        and not line[fence_match.end():].strip()
                    ):
                        active_fence = None
                converted.append(line)

        flush_outside()
        return "".join(converted)

    @staticmethod
    def _count_table_columns(html_table: str) -> int:
        rows = re.findall(
            r"<tr\b[^>]*>(.*?)</tr>",
            html_table,
            flags=re.DOTALL | re.IGNORECASE,
        )
        max_columns = 0
        for row in rows:
            attributes_list = re.findall(
                r"<(?:td|th)\b([^>]*)>", row, flags=re.IGNORECASE
            )
            column_count = 0
            for attributes in attributes_list:
                colspan_match = re.search(
                    r"\bcolspan\s*=\s*(?:\"(\d+)\"|'(\d+)'|(\d+))",
                    attributes,
                    flags=re.IGNORECASE,
                )
                if colspan_match:
                    column_count += max(1, int(next(g for g in colspan_match.groups() if g)))
                else:
                    column_count += 1
            max_columns = max(max_columns, column_count)
        return max_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown 파일을 XeLaTeX PDF로 변환")
    parser.add_argument("input", type=Path, help="입력 Markdown 파일")
    parser.add_argument("output", type=Path, nargs="?", help="출력 PDF 파일 (기본: 입력과 같은 이름)")
    parser.add_argument("--main-font", default=None, help="본문 폰트 (예: Malgun Gothic)")
    parser.add_argument("--sans-font", default=None, help="산세리프 폰트")
    args = parser.parse_args()

    output_path = args.output or args.input.with_suffix(".pdf")
    result = PDFRenderer(args.main_font, args.sans_font).render(args.input, output_path)
    print(f"PDF 생성 완료: {result}")


if __name__ == "__main__":
    main()
