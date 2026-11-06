"""Usage
  python -m evaluation run --pred PRED.jsonl --out results/NAME [--testset data/test.jsonl]
                           [--glossary glossary.jsonl] [--placeholder-regex REGEX] [--no-mt]
  python -m evaluation compare results/A results/B ... [--out compare.md]
  python -m evaluation significance --a A.jsonl --b B.jsonl [--glossary glossary_curated.json] [--n 1000]
"""
import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from . import mt, sanity, structure, term
from .io import ensure_dir, load_predictions, sha256_file

SUMMARY_KEYS = [
    "n", "bleu_ko_mecab", "chrf",
    "term_keep_f1_vs_ref", "term_keep_recall_vs_ref", "term_keep_precision_vs_ref",
    "team_term_rate", "team_keep_rate", "team_translate_rate", "team_keep_n", "team_translate_n",
    "glossary_compliance",
    "keep_preservation_rate", "overtranslation_rate", "translate_compliance_rate", "bilingual_accuracy",
    "glossary_hits",
    "structure_pass_rate", "invariant_preservation", "number_preservation",
    "clean_rate", "hit_max_tokens", "mostly_not_korean", "repetition",
]


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def evaluate(rows, glossary=None, placeholder_regex=structure.DEFAULT_PLACEHOLDER, with_mt=True):
    scored = [r for r in rows if r.get("status", "ok") == "ok"]
    detector = term._g.build_detector(glossary) if glossary else None
    per_ex, ref_per, glo_per, strict_per, str_per, san_per = [], [], [], [], [], []
    for r in rows:
        src, ref, pred = r["source_en"], r["reference_ko"], r["prediction_ko"]
        s = sanity.check(src, pred, r)
        san_per.append(s)
        row = {"id": r["id"], "domain": r.get("domain"), "status": r.get("status", "ok"),
               "len_ratio": s["len_ratio"], "issues": ";".join(s["issues"])}
        if r.get("status", "ok") == "ok":
            t = term.reference_based(src, ref, pred)
            ref_per.append(t)
            st = structure.check(src, pred, placeholder_regex)
            str_per.append(st)
            row.update({"ref_kept": t["ref_kept"], "pred_kept": t["pred_kept"], "agree": t["agree"],
                        "missed_terms": " ".join(t["missed_terms"]), "extra_terms": " ".join(t["extra_terms"]),
                        "structure_pass": st["structure_pass"], "structure_errors": " | ".join(st["errors"])})
            if glossary is not None:
                g = term.glossary_based(src, pred, glossary)
                glo_per.append(g)
                strict_per.append(term._g.team_check(src, pred, detector))
                row["glossary_violations"] = " | ".join(g["violations"])
                row["team_violations"] = " | ".join(strict_per[-1]["violations"])
        per_ex.append(row)

    summary = {"n": len(rows), "n_scored": len(scored)}
    if with_mt and scored:
        hyps = [r["prediction_ko"] for r in scored]
        refs = [r["reference_ko"] for r in scored]
        summary.update(mt.corpus_scores(hyps, refs))
        chrfs = dict(zip([r["id"] for r in scored], mt.sentence_chrf(hyps, refs)))
        for row in per_ex:
            row["chrf_sentence"] = chrfs.get(row["id"])
    summary.update(term.aggregate_reference(ref_per))
    if glossary is not None:
        summary.update(term.aggregate_glossary(glo_per, strict_per))
    summary.update(structure.aggregate(str_per))
    summary.update(sanity.aggregate(san_per))
    return summary, per_ex


def by_domain(rows, **kw):
    groups = defaultdict(list)
    for r in rows:
        if r.get("domain"):
            groups[r["domain"]].append(r)
    return {d: evaluate(g, **kw)[0] for d, g in sorted(groups.items())}


def write_outputs(out, summary, per_ex, meta, domains):
    out = ensure_dir(out)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if domains:
        (out / "by_domain.json").write_text(json.dumps(domains, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({k for row in per_ex for k in row}, key=lambda k: (k != "id", k))
    with open(out / "per_example.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_ex)
    lines = [f"# {meta['name']}", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {summary[k]} |" for k in SUMMARY_KEYS if k in summary]
    if domains:
        lines += ["", "## By domain", "", "| domain | n | chrf | term_keep_f1_vs_ref | structure_pass_rate |", "|---|---|---|---|---|"]
        lines += [f"| {d} | {s['n']} | {s.get('chrf')} | {s.get('term_keep_f1_vs_ref')} | {s.get('structure_pass_rate')} |"
                  for d, s in domains.items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_run(a):
    rows, missing = load_predictions(a.pred, a.testset)
    if missing:
        print(f"[warn] {len(missing)} test ids have no prediction; scores cover predictions only", file=sys.stderr)
    glossary = term.load_glossary(a.glossary) if a.glossary else None
    kw = dict(glossary=glossary, placeholder_regex=a.placeholder_regex, with_mt=not a.no_mt)
    summary, per_ex = evaluate(rows, **kw)
    summary["missing_predictions"] = len(missing)
    domains = by_domain(rows, **kw) if a.testset else {}
    meta = {
        "name": Path(a.out).name,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "pred_file": str(a.pred), "pred_sha256": sha256_file(a.pred),
        "testset": a.testset, "testset_sha256": sha256_file(a.testset) if a.testset else None,
        "glossary": a.glossary, "glossary_sha256": sha256_file(a.glossary) if a.glossary else None,
        "placeholder_regex": a.placeholder_regex,
        "adapter_path": sorted({r.get("adapter_path") for r in rows if r.get("adapter_path")}),
        "eval_code_commit": _git_commit(),
    }
    write_outputs(a.out, summary, per_ex, meta, domains)
    print(json.dumps({k: summary.get(k) for k in SUMMARY_KEYS if k in summary}, ensure_ascii=False, indent=2))
    print(f"saved: {a.out}")


def cmd_compare(a):
    runs = []
    for d in a.results:
        s = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
        runs.append((Path(d).name, s))
    keys = [k for k in SUMMARY_KEYS if any(k in s for _, s in runs)]
    base = runs[0][1]
    lines = ["| metric | " + " | ".join(n for n, _ in runs) + " |", "|---" * (len(runs) + 1) + "|"]
    for k in keys:
        cells = []
        for i, (_, s) in enumerate(runs):
            v = s.get(k)
            if i and isinstance(v, (int, float)) and isinstance(base.get(k), (int, float)) and not isinstance(v, bool):
                cells.append(f"{v} ({v - base[k]:+.4g})")
            else:
                cells.append(str(v))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + f"\n\n(괄호 안은 첫 번째 결과 {runs[0][0]} 대비 차이)\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


def cmd_significance(a):
    from . import glossary as G
    from .significance import paired_bootstrap
    rows_a, _ = load_predictions(a.a)
    rows_b, _ = load_predictions(a.b)
    det = G.build_detector(G.load(a.glossary)) if a.glossary else None
    res = paired_bootstrap(rows_a, rows_b, n=a.n, detector=det)
    res.update({"A_file": a.a, "B_file": a.b})
    text = json.dumps(res, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(prog="evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pred", required=True, help="predictions JSONL from test.py")
    r.add_argument("--out", required=True, help="results/<date>_<name>")
    r.add_argument("--testset", help="data/test.jsonl for alignment check and per-domain scores")
    r.add_argument("--glossary", help="team glossary_curated.json (or legacy JSONL)")
    r.add_argument("--placeholder-regex", default=structure.DEFAULT_PLACEHOLDER)
    r.add_argument("--no-mt", action="store_true", help="skip BLEU/chrF (no mecab installed)")
    c = sub.add_parser("compare")
    c.add_argument("results", nargs="+")
    c.add_argument("--out")
    sg = sub.add_parser("significance", help="paired bootstrap: is system B better than A?")
    sg.add_argument("--a", required=True); sg.add_argument("--b", required=True)
    sg.add_argument("--glossary"); sg.add_argument("--n", type=int, default=1000); sg.add_argument("--out")
    a = p.parse_args(argv)
    {"run": cmd_run, "compare": cmd_compare, "significance": cmd_significance}[a.cmd](a)


if __name__ == "__main__":
    main()
