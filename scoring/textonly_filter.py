"""Extract a text-only view of a multimodal bundle.

Filters each modality so only the text-bearing fields survive — drops media URLs,
GPS coordinates, video thumbnails, etc. Useful for adversary models that only
consume text, and for ablation studies comparing multimodal vs text-only signal.

Usage:
    python3 textonly_filter.py <bundle_in.json> <bundle_out.json>
    python3 textonly_filter.py --cohort path/to/cohorts/series_a/ --out path/to/series_a_textonly/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


# Fields to keep per modality. Everything else is dropped.
TEXT_FIELDS = {
    "searches":  ["day", "time", "query"],
    "messages":  ["day", "time", "with", "text"],
    "calendar":  ["day", "time", "title", "notes"],
    "videos":    ["day", "time", "title", "channel"],
    "media":     ["day", "time", "caption", "alt"],
    "locations": ["day", "time", "place", "category"],
}


def textonly(bundle: dict) -> dict:
    out = {k: v for k, v in bundle.items() if k != "samples"}
    samples = bundle.get("samples", {})
    out_samples = {}
    for mod, items in samples.items():
        if not isinstance(items, list):
            out_samples[mod] = items
            continue
        keep_fields = TEXT_FIELDS.get(mod, ["day", "time", "text", "title", "name"])
        out_samples[mod] = [
            {k: v for k, v in it.items() if k in keep_fields}
            for it in items if isinstance(it, dict)
        ]
    out["samples"] = out_samples
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_in", nargs="?")
    ap.add_argument("bundle_out", nargs="?")
    ap.add_argument("--cohort", help="Convert every *_intake.json in a directory")
    ap.add_argument("--out", help="Output directory (used with --cohort)")
    a = ap.parse_args()

    if a.cohort:
        src = Path(a.cohort); dst = Path(a.out or (str(src) + "_textonly"))
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.glob("*_intake.json")):
            b = json.loads(f.read_text())
            (dst / f.name).write_text(json.dumps(textonly(b), indent=1, ensure_ascii=False))
            n += 1
        print(f"wrote {n} bundles to {dst}")
    else:
        if not a.bundle_in or not a.bundle_out:
            ap.error("provide bundle_in and bundle_out, or --cohort + --out")
        b = json.loads(Path(a.bundle_in).read_text())
        Path(a.bundle_out).write_text(json.dumps(textonly(b), indent=1, ensure_ascii=False))
        print(f"wrote {a.bundle_out}")


if __name__ == "__main__":
    main()
