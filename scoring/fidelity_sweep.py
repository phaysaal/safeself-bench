#!/usr/bin/env python3
"""
Phase B fidelity sweep — run the 3-judge panel (Opus + Codex + Gemini) on a
whole cohort of activity bundles. Mirrors the realism-panel architecture.

Usage:
  # Sweep one cohort with the 3-judge panel:
  python3 eval/fidelity_sweep.py \\
      --bundle-dir eval/activity_bundles/prod30_b1 \\
      --answer-keys-dir eval/answer_keys_phase_b \\
      --question-bank eval/question_bank_phase_b.json \\
      --judges opus,codex,gemini \\
      --out-dir eval/scores/fidelity_phase_b/prod30_b1

  # Single judge:
  python3 eval/fidelity_sweep.py \\
      --bundle-dir eval/activity_bundles/prod30_b1 \\
      --answer-keys-dir eval/answer_keys_phase_b \\
      --question-bank eval/question_bank_phase_b.json \\
      --judges opus \\
      --out-dir eval/scores/fidelity_phase_b/prod30_b1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import median
from typing import Optional

# Import evaluator pieces from the sibling script
sys.path.insert(0, str(Path(__file__).parent))
from fidelity_evaluate import evaluate as evaluate_one  # noqa: E402


# ── persona-name normalisation (shared with realism pipeline) ────────────────

def normalise(name: str) -> str:
    n = name.lower().strip().replace("-", "_")
    n = re.sub(r"[()\']", "", n)
    n = re.sub(r"[^a-z0-9_]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    for prefix in ("prod30_b1_textonly_", "prod30_b1_gptoss_textonly_",
                   "prod30_b2_textonly_", "prod30_b2_gptoss_textonly_",
                   "prod30_b1_gptoss_", "prod30_b2_gptoss_", "prod30_b2_",
                   "prod30_b1_", "ours_multimodal_deji_",
                   "codex30_b1_textonly_", "codex30_b1_",
                   "codex30_b2_textonly_", "codex30_b2_",
                   "opus_combined_textonly_",
                   "opus30_b1_textonly_", "opus30_b1_",
                   "opus30_b2_textonly_", "opus30_b2_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
    n = n.replace("_intake", "").replace("_target_1", "").replace("_target_2", "")
    n = n.replace("_target_3", "").replace("_target_4", "")
    if n.startswith("the_"):
        n = n[4:]
    return n


def find_answer_key(bundle_path: Path, ak_dir: Path) -> Optional[Path]:
    """Match a bundle to its answer key by normalised persona name + target index."""
    bname = normalise(bundle_path.stem)
    # extract target index from full bundle name (looks for _target_N)
    m = re.search(r"target_([1-4])", bundle_path.name)
    target = m.group(1) if m else None
    # Search the answer-key dir
    candidates = []
    for ak in ak_dir.glob("*.json"):
        ak_norm = normalise(ak.stem)
        ak_target_match = re.search(r"target_([1-4])", ak.name)
        ak_target = ak_target_match.group(1) if ak_target_match else None
        if ak_norm == bname and (target is None or ak_target == target):
            candidates.append(ak)
    if not candidates:
        # fallback: same persona name, prefer target_1
        for ak in ak_dir.glob("*.json"):
            if normalise(ak.stem) == bname:
                candidates.append(ak)
    # If multiple candidates and no target hint, prefer target_1 (default)
    if len(candidates) > 1 and target is None:
        for c in candidates:
            if "target_1" in c.name:
                return c
    return candidates[0] if candidates else None


# ── sweep driver ────────────────────────────────────────────────────────────

def sweep_judge(bundle_dir: Path, ak_dir: Path, qbank_path: Path,
                judge: str, out_dir: Path, retry: int = 2) -> dict:
    """Run one judge over all bundles in a cohort. Skip already-scored bundles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = sorted(bundle_dir.glob("*_intake.json"))
    if not bundles:
        bundles = sorted(bundle_dir.glob("*.json"))
    print(f"[sweep] judge={judge}  bundles={len(bundles)}  out={out_dir}", file=sys.stderr)
    summary = []
    for i, bp in enumerate(bundles, 1):
        out_path = out_dir / f"{bp.stem}__{judge}.json"
        if out_path.exists():
            try:
                rep = json.loads(out_path.read_text())
                if rep.get("scores", {}).get("overall_fidelity") is not None:
                    print(f"  [{i:2d}/{len(bundles)}] {bp.stem}  SKIP (already scored)", file=sys.stderr)
                    summary.append({"bundle": bp.name, "score": rep["scores"]["overall_fidelity"]})
                    continue
            except Exception:
                pass
        ak = find_answer_key(bp, ak_dir)
        if ak is None:
            print(f"  [{i:2d}/{len(bundles)}] {bp.stem}  NO ANSWER KEY — skip", file=sys.stderr)
            continue
        start = time.time()
        last_err = None
        for attempt in range(retry):
            try:
                rep = evaluate_one(bp, ak, qbank_path, judge, None, out_path)
                wall = time.time() - start
                score = rep["scores"]["overall_fidelity"]
                print(f"  [{i:2d}/{len(bundles)}] {bp.stem}  {judge}={score}  ({wall:.0f}s)",
                      file=sys.stderr)
                summary.append({"bundle": bp.name, "score": score, "n": rep["scores"]["n_scored"]})
                break
            except RuntimeError as e:
                last_err = str(e)
                if "GEMINI_QUOTA_EXHAUSTED" in last_err:
                    print(f"  [{i:2d}/{len(bundles)}] gemini quota exhausted — stopping sweep",
                          file=sys.stderr)
                    return {"judge": judge, "summary": summary, "stopped_on": bp.name,
                            "reason": "GEMINI_QUOTA_EXHAUSTED"}
                if attempt < retry - 1:
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"  [{i:2d}/{len(bundles)}] {bp.stem}  FAIL: {last_err[:120]}",
                          file=sys.stderr)
    return {"judge": judge, "summary": summary}


def median_panel(out_dir: Path, judges: list[str]) -> dict:
    """Walk all per-judge score files and compute the per-persona panel median."""
    by_persona: dict[str, dict] = {}
    for jf in out_dir.glob(f"*__*.json"):
        m = re.match(r"(.+)__([^_]+)\.json", jf.name)
        if not m:
            continue
        persona, judge = m.group(1), m.group(2)
        if judge not in judges:
            continue
        try:
            rep = json.loads(jf.read_text())
            sc = rep.get("scores", {}).get("overall_fidelity")
            if sc is not None:
                by_persona.setdefault(persona, {})[judge] = sc
        except Exception:
            pass
    rows = []
    for persona, jscores in by_persona.items():
        scores = list(jscores.values())
        rows.append({
            "persona": persona,
            "n_judges": len(scores),
            "judges": jscores,
            "median": round(median(scores), 3) if scores else None,
        })
    rows.sort(key=lambda r: r["median"] or 0, reverse=True)
    cohort_med = (round(median([r["median"] for r in rows if r["median"] is not None]), 3)
                  if rows else None)
    return {"cohort_median": cohort_med, "n_personas": len(rows), "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--answer-keys-dir", required=True)
    p.add_argument("--question-bank", required=True)
    p.add_argument("--judges", default="opus,codex,gemini",
                   help="Comma-separated judge names")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--retry", type=int, default=2)
    p.add_argument("--summary-only", action="store_true",
                   help="Just compute the panel median from existing score files")
    a = p.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judges = [j.strip() for j in a.judges.split(",") if j.strip()]

    if not a.summary_only:
        for j in judges:
            sweep_judge(Path(a.bundle_dir), Path(a.answer_keys_dir),
                        Path(a.question_bank), j, out_dir, a.retry)

    panel = median_panel(out_dir, judges)
    panel["judges"] = judges
    panel["bundle_dir"] = str(a.bundle_dir)
    summary_path = out_dir / "_panel_median.json"
    summary_path.write_text(json.dumps(panel, indent=2))
    print(f"\n[panel] cohort median = {panel['cohort_median']}  (n={panel['n_personas']})",
          file=sys.stderr)
    print(f"[panel] wrote {summary_path}", file=sys.stderr)
    # Print top + bottom 5 to stderr
    if panel["rows"]:
        print("\nTop 5:", file=sys.stderr)
        for r in panel["rows"][:5]:
            print(f"  {r['persona']:50s}  {r['median']:.2f}  judges={r['judges']}", file=sys.stderr)
        print("\nBottom 5:", file=sys.stderr)
        for r in panel["rows"][-5:]:
            print(f"  {r['persona']:50s}  {r['median']:.2f}  judges={r['judges']}", file=sys.stderr)


if __name__ == "__main__":
    main()
