#!/usr/bin/env python3
"""
Realism judge.

Asks an LLM "On a 0–1 scale, does this corpus look like a real person's
digital footprint?" — given a fixed-budget intake bundle.

Usage:
  python3 scoring/realism_judge.py --bundle-dir cohorts/series_a
  python3 scoring/realism_judge.py --bundle-dir cohorts/series_a --out-dir scores/

Output: realism scores → <out-dir>/realism_<dir-name>.json
"""
import argparse, json, subprocess
from pathlib import Path

# Default score output: a `scores/` subdir of the current working dir.
SCORES = Path("scores")
SCORES.mkdir(parents=True, exist_ok=True)

REALISM_SYSTEM = """You are evaluating whether a synthetic digital-footprint
intake bundle looks plausibly like a real person's digital trail or whether
it has obvious tampering. Score 0–1 (continuous):

  1.0 = entirely plausible, no obvious markers of synthesis or tampering
  0.7 = mostly plausible with minor uneven patches
  0.5 = visibly synthetic but internally consistent
  0.3 = significant red flags (uniform style, missing temporal variation,
        artifacts inconsistent with the apparent character)
  0.0 = obvious tampering — duplicated patterns, anachronistic items,
        cross-modal inconsistency

Output ONLY a JSON object: {"realism": <float 0-1>, "rationale": "<2-3 sentences>"}.
No prose, no preamble."""

REALISM_USER_TMPL = """Below is a sampled slice of a person's digital footprint
(messages, searches, locations, videos watched, mail subjects, calendar events,
contacts manifest, photo metadata).

Evaluate whether this looks plausibly like a real person's digital life or
whether it has obvious tampering markers (synthetic-looking patterns,
uniformity, contradictions, anachronisms, demographic inconsistency).

Output JSON only.

INTAKE BUNDLE:
{bundle}"""

def _extract_json(text):
    i = text.find('{'); j = text.rfind('}')
    if i < 0 or j < 0:
        raise ValueError(f"No JSON in output: {text[:200]}")
    try:
        return json.loads(text[i:j+1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}; text={text[i:j+1][:200]}")


def _call_gemini(combined, timeout):
    cmd = ["gemini", "--model", "gemini-3.1-pro-preview", "--prompt", "-"]
    for _ in range(4):
        proc = subprocess.run(cmd, input=combined, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        import time; time.sleep(5)
    raise RuntimeError(f"gemini failed: rc={proc.returncode} stderr={proc.stderr[:300]!r}")


def _call_claude_code(combined, model, timeout):
    """Claude Code CLI (uses subscription credit, not API key)."""
    cmd = ["claude", "--print", "--model", model, "--tools", "",
           "--output-format", "json", "--no-session-persistence"]
    for _ in range(4):
        proc = subprocess.run(cmd, input=combined, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                env = json.loads(proc.stdout)
                return env.get("result", "").strip()
            except json.JSONDecodeError:
                pass
        import time; time.sleep(5)
    raise RuntimeError(f"claude failed: rc={proc.returncode} stderr={proc.stderr[:300]!r}")


def _ollama_post(prompt, model, timeout, host, force_json=False):
    """Single Ollama /api/generate call."""
    import urllib.request, urllib.error
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1, "num_ctx": 98304}}
    if force_json:
        payload["format"] = "json"
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                env = json.loads(resp.read().decode())
                # Some models (qwen3 reasoning) split chain-of-thought into a
                # separate `thinking` field. If the final `response` is empty
                # we still want to recover whatever content the model produced.
                response = (env.get("response") or "").strip()
                thinking = (env.get("thinking") or "").strip()
                if response:
                    return response
                return thinking
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            last = str(e)
            import time; time.sleep(5)
    raise RuntimeError(f"ollama failed: {last}")


EXTRACT_TMPL = """From the analysis below, extract a single JSON object with exactly two keys:
  "realism": a float in [0,1] representing how plausible the analysed bundle is as a real person's digital footprint
  "rationale": a one-sentence summary (max 200 chars) of why the analyst gave that score
Output the JSON object only — no markdown, no preamble, no commentary.

ANALYSIS:
{markdown}"""


def _call_ollama(combined, model, timeout, host="http://localhost:11434"):
    """Two-pass Ollama: (1) free-form markdown analysis, (2) structured JSON
    extraction from that markdown using the same model. Free-form pass 1 is
    much faster + higher-quality than format=json constraint for big MoE models.

    Returns the JSON string from pass 2.
    """
    # Pass 1: natural-mode reasoning
    markdown = _ollama_post(combined, model, timeout, host, force_json=False)

    # If pass 1 happened to emit clean JSON already, short-circuit
    try:
        _extract_json(markdown)
        return markdown
    except ValueError:
        pass

    # Pass 2: strict extraction. Same model + host (keeps real-data on the
    # rented pod; no extra cloud egress).
    extract_prompt = EXTRACT_TMPL.format(markdown=markdown[:12000])
    return _ollama_post(extract_prompt, model, timeout, host, force_json=True)


def call_realism_judge(bundle_json, backend="gemini", timeout=1200, ollama_host="http://localhost:11434"):
    """Score a bundle's realism. backend ∈ {gemini, opus, sonnet, haiku, ollama-MODEL}."""
    user = REALISM_USER_TMPL.format(bundle=json.dumps(bundle_json, ensure_ascii=False, indent=1))
    combined = f"{REALISM_SYSTEM}\n\n{user}"

    if backend == "gemini":
        text = _call_gemini(combined, timeout)
    elif backend in ("opus", "sonnet", "haiku") or backend.startswith("claude-"):
        model = backend if backend.startswith("claude-") else backend  # claude CLI accepts alias
        text = _call_claude_code(combined, model, timeout)
    elif backend.startswith("ollama-"):
        model = backend[len("ollama-"):]
        text = _call_ollama(combined, model, timeout, host=ollama_host)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return _extract_json(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle-dir', required=True,
                    help="Directory containing *_intake.json files (e.g. cohorts/series_a)")
    ap.add_argument('--out-dir', default='scores',
                    help="Where to write realism_<cohort>.json (default: ./scores)")
    ap.add_argument('--bundle-file', default=None,
                    help="Single bundle JSON file (overrides --bundle-dir + --persona)")
    ap.add_argument('--persona', default='all', help="P##, comma separated list, or 'all'")
    ap.add_argument('--backend', default='gemini',
                    help="gemini | opus | sonnet | haiku | ollama-MODEL (e.g. ollama-llama3.3:70b)")
    ap.add_argument('--ollama-host', default='http://localhost:11434',
                    help="Ollama server (use http://VAST_IP:11434 for rented GPU)")
    ap.add_argument('--out', default=None,
                    help="Output JSON path (default: scores/realism_<dir-or-file>_<backend>.json)")
    args = ap.parse_args()

    if args.bundle_file:
        files = [Path(args.bundle_file)]
        report_tag = Path(args.bundle_file).stem
    else:
        bundle_dir = Path(args.bundle_dir)
        if args.persona == 'all':
            files = sorted(bundle_dir.glob("*_intake.json"))
        else:
            ids = set(args.persona.split(','))
            files = sorted([p for p in bundle_dir.glob("*_intake.json") if any(p.name.startswith(i+'_') for i in ids)])
        report_tag = bundle_dir.name
    print(f"Judging realism of {len(files)} bundles  ·  backend={args.backend}")

    results = []
    for f in files:
        bundle = json.load(open(f, encoding='utf-8'))
        try:
            r = call_realism_judge(bundle.get('samples', {}), backend=args.backend,
                                   ollama_host=args.ollama_host)
            results.append({
                'persona': bundle.get('persona_slug'),
                'realism': r.get('realism'),
                'rationale': r.get('rationale'),
            })
            v = r.get('realism', 0)
            vstr = f"{v:.2f}" if isinstance(v, (int, float)) else "?"
            print(f"  {bundle['persona_slug']}: realism={vstr}  -- {r.get('rationale', '')[:100]}")
        except Exception as e:
            print(f"  {bundle['persona_slug']}: ERROR {e}")
            results.append({'persona': bundle.get('persona_slug'), 'error': str(e)})

    # Write report
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        safe_backend = args.backend.replace(":", "_").replace("/", "_")
        out = SCORES / f"realism_{report_tag}_{safe_backend}.json"
    json.dump({
        'bundle_dir': str(args.bundle_file or args.bundle_dir),
        'mean_realism': (sum(r['realism'] for r in results if isinstance(r.get('realism'), (int, float))) /
                         max(1, len([r for r in results if isinstance(r.get('realism'), (int, float))]))),
        'per_persona': results,
    }, open(out, 'w', encoding='utf-8'), indent=2)
    print(f"\nReport → {out}")

if __name__ == '__main__':
    main()
