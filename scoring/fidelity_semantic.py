#!/usr/bin/env python3
"""
Semantic-fidelity scorer (v2 methodology).

Replaces the rule-based truth + per-question scoring-rule pipeline with:

  1. Five type-dispatched semantic-distance operators (text / numeric /
     binary / ordinal / list), where the text operator uses a sentence-
     transformer embedding's cosine distance.

  2. Multi-judge consensus ground truth via popular(): for each (persona,
     question), the judge whose spec-read answer is most semantically
     central wins.

  3. L2 fidelity distance + a normalised [0,1] fidelity score.

Phase 1 (paid LLM): persona-spec reads + activity-bundle reads (already
exists from prior sweeps + GT validation).

Phase 2 (free): embeddings + popular() + distance — runs locally on
sentence-transformer model, deterministic, ~seconds per cohort.

Usage:
  python3 eval/fidelity_semantic.py \\
      --gt-val-dir eval/scores/gt_validation_phase_b \\
      --activity-scores-dir eval/scores/fidelity_phase_b/prod30_b1 \\
      --question-bank eval/question_bank_phase_b.json \\
      --out eval/scores/fidelity_phase_b_semantic/prod30_b1.json \\
      --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
# sentence_transformers imported lazily inside SemDist.model — only needed
# for the embedding backend, not for the ollama / claude LLM backend.


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_str(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, list):
        return " · ".join(str(i) for i in x)
    return str(x)


def _to_num(x: Any) -> float | None:
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x)
        if m: return float(m.group(0))
    return None


def _norm_yn(x: Any) -> str:
    s = str(x).strip().lower()
    if s in ("y","yes","true","1"): return "y"
    if s in ("n","no","false","0"): return "n"
    return s


# ── the five micro-`~` operators ─────────────────────────────────────────────

class SemDist:
    """Lazy-initialised semantic distance functions, all returning [0,1]
    where 0 = identical and 1 = completely unrelated/opposite."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 equivalence_threshold: float = 0.15,
                 distinction_threshold: float = 0.85,
                 llm_backend=None):
        """
        llm_backend : if provided, an LLMSemDist instance used for the .text()
                      method instead of embeddings. Other methods (numeric,
                      binary, ordinal) keep their principled formulas.
        """
        self.model_name = model_name
        self.equivalence_threshold = equivalence_threshold
        self.distinction_threshold = distinction_threshold
        self.llm_backend = llm_backend
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    # Module-level model cache keyed by name — prevents re-load across
    # multiple SemDist instantiations or wherever the lazy init is re-triggered.
    _GLOBAL_MODEL_CACHE: dict = {}

    @property
    def model(self):
        cache = SemDist._GLOBAL_MODEL_CACHE
        if self.model_name not in cache:
            import os
            hf_home = os.environ.get("HF_HOME")
            if hf_home and not Path(hf_home).is_dir():
                os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
                os.environ.pop("TRANSFORMERS_CACHE", None)
            print(f"[semantic] loading embedding model {self.model_name}...",
                  file=sys.stderr)
            from sentence_transformers import SentenceTransformer
            cache[self.model_name] = SentenceTransformer(self.model_name)
        self._model = cache[self.model_name]
        return self._model

    def embed(self, s: str) -> np.ndarray:
        if s not in self._cache:
            self._cache[s] = self.model.encode(s, normalize_embeddings=True)
        return self._cache[s]

    def text(self, a: Any, b: Any, context: dict | None = None) -> float:
        """Semantic distance for text. Delegates to LLM backend if set,
        otherwise uses embedding cosine with equivalence/distinction
        thresholds. The `context` dict (question metadata) is forwarded to
        the LLM backend if applicable; ignored by the embedding backend."""
        if self.llm_backend is not None:
            return self.llm_backend.text(a, b, context)
        sa = _to_str(a).strip().lower()
        sb = _to_str(b).strip().lower()
        if not sa and not sb: return 0.0
        if not sa or not sb: return 1.0
        if sa == sb: return 0.0
        ea, eb = self.embed(sa), self.embed(sb)
        cos = float(np.dot(ea, eb))
        raw = max(0.0, min(1.0, 1.0 - cos))
        if raw <= self.equivalence_threshold: return 0.0
        if raw >= self.distinction_threshold: return 1.0
        return raw

    def numeric(self, a: Any, b: Any, scale: float = 10.0) -> float:
        na, nb = _to_num(a), _to_num(b)
        if na is None or nb is None: return self.text(a, b)
        return min(1.0, abs(na - nb) / scale)

    def binary(self, a: Any, b: Any) -> float:
        return 0.0 if _norm_yn(a) == _norm_yn(b) else 1.0

    def ordinal(self, a: Any, b: Any, options: list) -> float:
        opts = [str(o).strip().lower() for o in options]
        sa, sb = str(a).strip().lower(), str(b).strip().lower()
        if sa not in opts or sb not in opts:
            return self.text(a, b)
        if len(opts) <= 1: return 0.0
        return abs(opts.index(sa) - opts.index(sb)) / (len(opts) - 1)

    def list_(self, a: Any, b: Any, context: dict | None = None) -> float:
        """Best-pairing average over list items."""
        la = a if isinstance(a, list) else [a] if a else []
        lb = b if isinstance(b, list) else [b] if b else []
        la = [str(x).strip() for x in la if x]
        lb = [str(x).strip() for x in lb if x]
        if not la and not lb: return 0.0
        if not la or not lb: return 1.0
        d = np.zeros((len(la), len(lb)))
        for i, x in enumerate(la):
            for j, y in enumerate(lb):
                d[i, j] = self.text(x, y, context)
        # Greedy best-match (good enough; Hungarian for n>4 if needed)
        used_a, used_b = set(), set()
        dists = []
        for _ in range(min(len(la), len(lb))):
            best, best_v = None, 2.0
            for i in range(len(la)):
                if i in used_a: continue
                for j in range(len(lb)):
                    if j in used_b: continue
                    if d[i, j] < best_v:
                        best_v, best = d[i, j], (i, j)
            if best is None: break
            used_a.add(best[0]); used_b.add(best[1])
            dists.append(best_v)
        # Penalty for unmatched items
        unmatched = (len(la) - len(used_a)) + (len(lb) - len(used_b))
        if unmatched:
            dists.extend([1.0] * unmatched)
        return float(mean(dists)) if dists else 1.0

    def mcq(self, a: Any, b: Any, options: list,
            aliases: dict | None = None) -> float:
        """Canonicalise each side to its option, then compare.
        Uses (1) exact match (2) substring match (3) alias table (4) embedding.
        """
        opts = [str(o) for o in options]
        opts_lc = [o.lower() for o in opts]
        # Build reverse alias map: alias_lc -> canonical_option
        alias_map: dict[str, str] = {}
        if aliases:
            for opt, alts in aliases.items():
                for alt in alts:
                    alias_map[str(alt).strip().lower()] = opt

        def canonicalise(x):
            s = str(x).strip().lower()
            if not s: return None
            if s in opts_lc: return opts[opts_lc.index(s)]
            if s in alias_map: return alias_map[s]
            # token-by-token: any token matches an option or alias
            for t in re.findall(r"[a-z]+", s):
                if t in opts_lc: return opts[opts_lc.index(t)]
                if t in alias_map: return alias_map[t]
            # substring word-boundary match of option in answer
            for o, ol in zip(opts, opts_lc):
                if re.search(rf'\b{re.escape(ol)}\b', s): return o
            # semantic fallback against options + alias labels
            try:
                e_x = self.embed(s)
                labels = list(opts) + list(alias_map.keys())
                canon = list(opts) + list(alias_map.values())
                sims = [float(np.dot(e_x, self.embed(str(lbl).lower())))
                        for lbl in labels]
                best = max(range(len(sims)), key=lambda i: sims[i])
                if sims[best] > 0.55: return canon[best]
            except Exception:
                pass
            return None

        ca, cb = canonicalise(a), canonicalise(b)
        if ca is not None and cb is not None:
            return 0.0 if ca == cb else 1.0
        return self.text(a, b)

    # Dispatcher
    def dist(self, a: Any, b: Any, question: dict) -> float:
        atype = question.get("answer_type", "string")
        if atype == "integer":
            tol = question.get("tolerance", 1)
            return self.numeric(a, b, scale=max(1.0, 2 * tol))
        if atype == "binary":         return self.binary(a, b)
        if atype == "ordinal":
            opts = question.get("options", [])
            return self.ordinal(a, b, opts)
        if atype in ("list_of_strings", "json_array"):
            return self.list_(a, b, question)
        if atype == "mcq":
            opts = question.get("options", [])
            aliases = question.get("option_aliases")
            return self.mcq(a, b, opts, aliases)
        # string, categorical, ternary all → semantic
        return self.text(a, b, question)


# ── popular() : multi-judge consensus truth ──────────────────────────────────

def popular(answers: dict[str, Any], question: dict, sd: SemDist,
            consensus_threshold: float = 0.5
           ) -> tuple[Any, dict[str, float], bool]:
    """Return (consensus_answer, per_judge_total_dist, no_consensus_flag).

    Answer with the minimum total distance to the others wins.
    no_consensus = True if min total distance > threshold.
    """
    judges = [j for j, v in answers.items() if v is not None]
    if not judges:
        return None, {}, True
    if len(judges) == 1:
        return answers[judges[0]], {judges[0]: 0.0}, False
    total = {}
    for j in judges:
        total[j] = sum(sd.dist(answers[j], answers[k], question)
                       for k in judges if k != j)
    best = min(total, key=lambda k: total[k])
    no_consensus = total[best] / (len(judges) - 1) > consensus_threshold
    return answers[best], total, no_consensus


# ── per-persona score ────────────────────────────────────────────────────────

def fidelity_score(truth: dict[str, Any], judge_ans: dict[str, Any],
                   qbank_idx: dict[str, dict], sd: SemDist) -> dict:
    """Compute D = sqrt(sum (truth[i] ~ judge_ans[i])^2) and the
    normalised F = 1 - D / sqrt(N) over questions present in both."""
    per_q = []
    sq_sum = 0.0
    n = 0
    for qid, t in truth.items():
        if t is None: continue
        if qid not in qbank_idx: continue
        a = judge_ans.get(qid)
        if a is None: continue
        d = sd.dist(a, t, qbank_idx[qid])
        per_q.append({"id": qid, "truth": t, "answer": a, "dist": round(d, 4)})
        sq_sum += d * d
        n += 1
    if n == 0:
        return {"n": 0, "L2": None, "F_norm": None, "per_question": []}
    L2 = math.sqrt(sq_sum)
    F = 1.0 - L2 / math.sqrt(n)
    return {"n": n, "L2": round(L2, 4),
            "F_norm": round(F, 4),
            "per_question": per_q}


# ── load helpers ─────────────────────────────────────────────────────────────

def normalise_pname(name: str) -> str:
    n = name.lower().strip()
    n = n.replace(".json", "")
    n = re.sub(r"__\w+$", "", n)           # strip __opus / __codex / __gemini
    n = re.sub(r"_intake$", "", n)          # strip activity-bundle suffix
    n = re.sub(r"_target_[1-4]$", "", n)    # strip target index
    # Strip cohort prefix (e.g. opus_combined_textonly_) so bundle and spec
    # filenames map to the same key. Order matters: longest-prefix first to
    # avoid e.g. "prod30_b1_" eating "prod30_b1_gptoss_textonly_".
    for prefix in ("prod30_b1_gptoss_textonly_", "prod30_b1_gptoss_dejittered_",
                   "prod30_b1_gptoss_", "prod30_b1_textonly_",
                   "prod30_b1_dejittered_", "prod30_b1_",
                   "prod30_b2_textonly_", "prod30_b2_dejittered_", "prod30_b2_",
                   "codex30_b1_textonly_", "codex30_b1_dejittered_", "codex30_b1_",
                   "codex30_b2_dejittered_", "codex30_b2_",
                   "opus30_b1_dejittered_", "opus30_b1_",
                   "opus_combined_textonly_", "opus_combined_",
                   "opus_exec_gemma_plans_dejittered_", "opus_exec_gemma_plans_",
                   "ours_multimodal_deji_", "ours_textonly_", "ours_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    if n.startswith("the_"): n = n[4:]
    return n


def load_judge_specs(gt_val_dir: Path, judges: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return d[persona_key][judge][question_id] = judge's spec answer."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for j in judges:
        for f in sorted(gt_val_dir.glob(f"*__{j}.json")):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            p = normalise_pname(f.stem.replace(f"__{j}", ""))
            for row in data.get("per_question", []):
                qid = row.get("id")
                ans = row.get("judge_answer")
                if qid and ans is not None:
                    out.setdefault(p, {}).setdefault(j, {})[qid] = ans
    return out


def load_activity_answers(scores_dir: Path, judges: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return d[persona_key][judge][question_id] = judge's answer from activity bundle."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for j in judges:
        for f in sorted(scores_dir.glob(f"*__{j}.json")):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            p = normalise_pname(f.stem.replace(f"__{j}", ""))
            for row in data.get("per_question", []):
                qid = row.get("id")
                ans = row.get("answer")
                if qid and ans is not None:
                    out.setdefault(p, {}).setdefault(j, {})[qid] = ans
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-val-dir", required=True,
                   help="Directory with persona-spec judge reads")
    p.add_argument("--activity-scores-dir", required=True,
                   help="Directory with activity-bundle judge reads")
    p.add_argument("--question-bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judges", default="opus,codex,gemini")
    p.add_argument("--truth-judges", default=None,
                   help="Which judges' spec reads to use for popular() truth (defaults to whatever is available)")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--consensus-threshold", type=float, default=0.5)
    p.add_argument("--backend", default="embedding",
                   choices=["embedding", "llm", "claude", "ollama"],
                   help="Text-distance backend. embedding=fast/free; llm/claude=Claude CLI; ollama=local Ollama HTTP")
    p.add_argument("--llm-model", default="haiku",
                   help="Which Claude model for the LLM backend")
    p.add_argument("--llm-cache", default="eval/cache/semdist_llm.json")
    p.add_argument("--tiered", action="store_true",
                   help="Apply cheap syntactic pre-filter before LLM ~ (substring, "
                        "token-Jaccard). Cuts LLM calls ~60-75% for open-text questions.")
    a = p.parse_args()

    qbank = json.loads(Path(a.question_bank).read_text())
    qbank_idx = {q["id"]: q for q in qbank["questions"]}

    judges = [j.strip() for j in a.judges.split(",") if j.strip()]
    truth_judges = ([j.strip() for j in a.truth_judges.split(",")]
                     if a.truth_judges else judges)

    print(f"[semantic] activity judges = {judges}", file=sys.stderr)
    print(f"[semantic] truth judges     = {truth_judges}", file=sys.stderr)
    print(f"[semantic] consensus threshold = {a.consensus_threshold}", file=sys.stderr)

    llm_backend = None
    if a.backend in ("llm", "claude", "ollama"):
        from llm_semdist import LLMSemDist  # local import
        # llm = claude CLI (default); ollama = local HTTP API
        llm_subbackend = "ollama" if a.backend == "ollama" else "claude"
        # Heuristic: if the model name contains ":" treat as ollama tag
        if ":" in a.llm_model and a.backend != "claude":
            llm_subbackend = "ollama"
        llm_backend = LLMSemDist(model=a.llm_model, cache_path=a.llm_cache,
                                 backend=llm_subbackend,
                                 use_tiered=a.tiered)
        print(f"[semantic] backend={llm_subbackend} ({a.llm_model})  cache={a.llm_cache}",
              file=sys.stderr)
    else:
        print(f"[semantic] backend=embedding ({a.model})", file=sys.stderr)

    sd = SemDist(a.model, llm_backend=llm_backend)
    spec_reads = load_judge_specs(Path(a.gt_val_dir), truth_judges)
    activity_reads = load_activity_answers(Path(a.activity_scores_dir), judges)

    # If LLM backend, walk all pairs we'll need and precompute distances
    if llm_backend is not None:
        print("[semantic] collecting needed pairs for LLM precompute...",
              file=sys.stderr)
        TEXT_TYPES = {"string", "categorical", "ternary", "mcq"}
        LIST_TYPES = {"list_of_strings", "json_array"}
        needed: list = []

        def build_ctx(qid: str, q: dict) -> dict:
            return {
                "id": qid,
                "question": q.get("question", ""),
                "answer_type": q.get("answer_type", ""),
                "options": q.get("options"),
                "tolerance": q.get("tolerance"),
            }

        for pkey, judges_for_persona in activity_reads.items():
            spec = spec_reads.get(pkey, {})
            if not spec: continue
            for qid, q in qbank_idx.items():
                atype = q.get("answer_type", "")
                # Only LLM-using types are worth precomputing
                if atype not in TEXT_TYPES and atype not in LIST_TYPES:
                    continue
                ctx = build_ctx(qid, q)
                truths = [spec[j].get(qid) for j in spec
                            if qid in spec[j] and spec[j].get(qid) is not None]
                # Build raw pairs (popular() needs pairwise on truth, plus
                # truth-vs-each-activity)
                raw_pairs = []
                for i, t1 in enumerate(truths):
                    for t2 in truths[i+1:]:
                        raw_pairs.append((t1, t2))
                for j, jd in judges_for_persona.items():
                    ans = jd.get(qid)
                    if ans is None: continue
                    for t in truths:
                        raw_pairs.append((t, ans))
                # Expand list-typed answers to item-level pairs
                if atype in LIST_TYPES:
                    for pa, pb in raw_pairs:
                        la = pa if isinstance(pa, list) else [pa] if pa else []
                        lb = pb if isinstance(pb, list) else [pb] if pb else []
                        for x in la:
                            for y in lb:
                                if x and y:
                                    needed.append((x, y, ctx))
                else:
                    for pa, pb in raw_pairs:
                        needed.append((pa, pb, ctx))
        print(f"[semantic] {len(needed)} total pair-requests (will dedup)",
              file=sys.stderr)
        llm_backend.precompute(needed)
    print(f"[semantic] spec reads     : {len(spec_reads)} personas across truth judges",
          file=sys.stderr)
    print(f"[semantic] activity reads : {len(activity_reads)} personas across activity judges",
          file=sys.stderr)

    # For each persona, compute truth via popular() across truth judges,
    # then compute distance for each activity judge.
    out_rows = []
    for pkey in sorted(activity_reads.keys()):
        spec = spec_reads.get(pkey, {})
        if not spec:
            print(f"  skip {pkey}: no spec reads available", file=sys.stderr)
            continue
        # Build truth: for each question, popular() across the judges that answered it
        truth_per_q: dict[str, Any] = {}
        no_consensus_qids: list[str] = []
        all_qs = set()
        for j_dict in spec.values(): all_qs.update(j_dict.keys())
        for qid in all_qs:
            if qid not in qbank_idx: continue
            j_answers = {j: spec[j].get(qid) for j in spec if qid in spec[j]}
            cons, totals, no_cons = popular(j_answers, qbank_idx[qid], sd,
                                              a.consensus_threshold)
            if cons is not None:
                truth_per_q[qid] = cons
            if no_cons:
                no_consensus_qids.append(qid)

        # Score each activity judge's answers against truth
        per_judge_scores: dict[str, dict] = {}
        for j in judges:
            ja = activity_reads[pkey].get(j, {})
            if not ja: continue
            res = fidelity_score(truth_per_q, ja, qbank_idx, sd)
            per_judge_scores[j] = res

        # Per-persona panel median over F_norm
        fs = [v["F_norm"] for v in per_judge_scores.values()
              if v.get("F_norm") is not None]
        panel_median = round(median(fs), 4) if fs else None

        out_rows.append({
            "persona": pkey,
            "n_truth_judges": len(spec),
            "n_questions_with_truth": len(truth_per_q),
            "no_consensus_qids": no_consensus_qids,
            "panel_median": panel_median,
            "per_judge": {j: {"L2": v["L2"], "F_norm": v["F_norm"], "n": v["n"]}
                            for j, v in per_judge_scores.items()},
        })

    # Cohort
    panel_meds = [r["panel_median"] for r in out_rows
                  if r.get("panel_median") is not None]
    cohort = {
        "n_personas": len(out_rows),
        "n_with_panel": len(panel_meds),
        "cohort_panel_median": round(median(panel_meds), 4) if panel_meds else None,
        "cohort_panel_mean": round(mean(panel_meds), 4) if panel_meds else None,
        "judges": judges,
        "truth_judges": truth_judges,
        "embedding_model": a.model,
        "consensus_threshold": a.consensus_threshold,
        "rows": out_rows,
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cohort, indent=2))

    print(f"\n[cohort] panel median F_norm = {cohort['cohort_panel_median']}",
          file=sys.stderr)
    print(f"[cohort] panel mean   F_norm = {cohort['cohort_panel_mean']}",
          file=sys.stderr)
    print(f"[cohort] n personas with panel = {len(panel_meds)} / {len(out_rows)}",
          file=sys.stderr)
    print(f"[wrote] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
