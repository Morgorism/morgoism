"""GPU-free tests that exercise the actual source team's Markdown parser.
These tests do NOT claim Stage4 neural inference has run.
"""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_pipeline import (PipelineError, protect_markdown, restore_markdown,
                          mark_mock_translation, verify_document, check_real_models)
RERANKER = ROOT / "components" / "horizon-reranker"


def sample_text():
    return """# Introduction

The CNN model uses $E=mc^2$ and [12]. See [paper](https://example.org/a).

![](images/figure.png)

| Name | Value |
| --- | --- |
| CNN | 10 |

$$
a^2+b^2=c^2
$$

```python
print('do not translate')
```

<table><tr><td>$x+1$</td><td>24</td></tr></table>

The proposed method is tested in 2026.
"""


def test_protection_restore_ignores_structural_components(tmp_path):
    base = sample_text()
    image = tmp_path / "images" / "figure.png"
    image.parent.mkdir()
    image.write_bytes(b"not-real-image-but-existence-check")
    ready, pieces, types = protect_markdown(base)
    assert "![](images/figure.png)" not in ready
    assert "$$\na^2+b^2=c^2\n$$" not in ready
    assert "<table>" not in ready
    assert "⟦PIPE" in ready
    assert types["RAW"] >= 1 and types["MATH"] >= 1 and types["INLINE"] >= 3
    translated, n = mark_mock_translation(ready, RERANKER)
    assert n > 0
    restored = restore_markdown(translated, pieces)
    assert "![](images/figure.png)" in restored
    assert "$$\na^2+b^2=c^2\n$$" in restored
    assert "<table><tr><td>$x+1$</td><td>24</td></tr></table>" in restored
    assert "print('do not translate')" in restored
    info = verify_document(base, restored, tmp_path)
    assert info["image_missing"] == []
    assert info["pipe_table_rows"] == 3


def test_missing_inline_token_fails():
    protected, mapping, _ = protect_markdown("The value is $x+1$.\n")
    with pytest.raises(PipelineError, match="훼손"):
        restore_markdown(protected.replace("⟦PIPE", "⟦BROKEN"), mapping)


def test_mutated_image_reference_fails(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"a")
    with pytest.raises(PipelineError, match="이미지 참조"):
        verify_document("![](images/a.png)\n", "![](images/b.png)\n", tmp_path)


def test_missing_image_fails(tmp_path):
    with pytest.raises(PipelineError, match="이미지 누락"):
        verify_document("![](images/lost.png)\n", "![](images/lost.png)\n", tmp_path)


def test_html_table_and_tilde_code_preserved():
    source = """~~~js
alert(1)
~~~
<table>
<tr><td>Data</td></tr>
</table>

Translation begins now.
"""
    protected, mapping, types = protect_markdown(source)
    assert types["CODE"] == 1
    assert types["HTMLTABLE"] == 1
    translated, n = mark_mock_translation(protected, RERANKER)
    assert n > 0
    assert "<table>\n<tr>" in restore_markdown(translated, mapping)


def test_bad_model_weights_explicitly_rejected(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"adapter": "/nonexistent/model", "backward_adapter": "/nonexistent/bwd"}))
    with pytest.raises(PipelineError, match="adapter_config.json"):
        check_real_models(cfg, RERANKER, RERANKER / "results/final/weights.json",
                          ROOT / "components/glossary_curated.json")


def test_team_reranker_weights_are_actual_e_all():
    weights = json.loads((RERANKER / "results/final/weights.json").read_text())
    assert weights["E_all"]["weights"]["glossary"] > 0
    assert weights["E_all"]["weights"]["bwd"] > 0
    cfg = json.loads((RERANKER / "configs/final.json").read_text())
    assert cfg["prompt_style"] == "jjha_stage4"
    assert cfg["n_samples"] == 6


def test_original_pdf_components_are_packaged():
    assert (ROOT / "components/pdf_to_markdown/make_paper_md.py").is_file()
    assert (ROOT / "components/markdown_to_pdf/pdf.py").is_file()
    assert (ROOT / "examples/mineru_result.zip").is_file()
    assert (ROOT / "examples/test.pdf").is_file()


def test_real_invocation_wiring_with_fake_model(tmp_path, monkeypatch):
    """Verify real CLI command, but substitute output: NEVER loads a real model."""
    import run_pipeline as runner
    a = tmp_path / 'stage4'
    b = tmp_path / 'bomin'
    for p in (a, b):
        p.mkdir()
        (p / 'adapter_config.json').write_text('{}')
        (p / 'adapter_model.safetensors').write_bytes(b'x' * 1_000_000)
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps({'adapter': str(a), 'backward_adapter': str(b)}))
    out = tmp_path / 'out'
    cmd_calls = []
    original_execute = runner.execute

    def capture(command, *, cwd=None, env=None):
        if '-m' in command and 'reranker' in command:
            cmd_calls.append(command)
            assert 'translate' in command
            assert command[command.index('--experiment') + 1] == 'E_all'
            src = Path(command[command.index('--input') + 1]).read_text()
            result, _ = runner.mark_mock_translation(src, RERANKER)
            Path(command[command.index('--out') + 1]).write_text(result)
        else:
            original_execute(command, cwd=cwd, env=env)

    monkeypatch.setattr(runner, 'execute', capture)
    args = runner.build_parser().parse_args([
        '--input', str(ROOT / 'examples/mineru_result.zip'),
        '--original-pdf', str(ROOT / 'examples/test.pdf'),
        '--out', str(out), '--name', 'fake_real',
        '--mode', 'real', '--renderer', 'skip',
        '--config', str(cfg), '--no-previews',
    ])
    runner.run_pipeline(args)
    assert len(cmd_calls) == 1
    assert (out / 'document/paper.ko.md').is_file()
    assert json.loads((out / 'pipeline_report.json').read_text())['verification']['image_missing'] == []
