"""Create an auditable combined manifest from approved real and AI image roots.

Example:
python scripts/build_manifest.py --real-root data/cifake/train/REAL \
  --ai-root data/cifake/train/FAKE --output manifests/cifake.csv --source cifake
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def stable_split(path: Path, val_fraction: float) -> str:
    value = int(hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_fraction else "train"


def read_exclusion_file(path: str | None) -> set[str]:
    if path is None:
        return set()
    with open(path, encoding="utf-8") as handle:
        return {line.strip().replace("\\", "/") for line in handle if line.strip() and not line.startswith("#")}


def is_forbidden(path: Path, forbidden_roots: list[Path], exclusion_entries: set[str]) -> bool:
    resolved = path.resolve()
    if any(resolved.is_relative_to(root) for root in forbidden_roots):
        return True
    canonical = str(resolved).replace("\\", "/")
    return canonical in exclusion_entries or resolved.name in exclusion_entries


def collect(root: Path, label: int, source: str, forbidden_roots: list[Path], exclusions: set[str], val_fraction: float):
    rows, excluded = [], []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if is_forbidden(path, forbidden_roots, exclusions):
            excluded.append({"image_path": str(path.resolve()), "reason": "forbidden_validation"})
            continue
        rows.append({
            "image_path": str(path.resolve()), "label": label, "source": source,
            "split": stable_split(path.resolve(), val_fraction),
        })
    return rows, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a labeled, leakage-audited image manifest.")
    parser.add_argument("--real-root", required=True, type=Path)
    parser.add_argument("--ai-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Dataset/source label stored in the manifest.")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--forbidden-root", type=Path, action="append", default=[],
                        help="Directory containing challenge validation data. May be repeated.")
    parser.add_argument("--exclude-file", help="Newline-delimited paths or filenames never permitted in this manifest.")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if not args.real_root.is_dir() or not args.ai_root.is_dir():
        parser.error("Both --real-root and --ai-root must be existing directories.")

    forbidden = [p.resolve() for p in args.forbidden_root]
    exclusions = read_exclusion_file(args.exclude_file)
    real_rows, real_excluded = collect(args.real_root, 0, args.source, forbidden, exclusions, args.val_fraction)
    ai_rows, ai_excluded = collect(args.ai_root, 1, args.source, forbidden, exclusions, args.val_fraction)
    manifest = pd.DataFrame(real_rows + ai_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    if manifest.empty or manifest.label.nunique() != 2:
        raise ValueError("The resulting manifest must contain both real and AI images.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    audit_path = args.output.with_suffix(".excluded.csv")
    pd.DataFrame(real_excluded + ai_excluded, columns=["image_path", "reason"]).to_csv(audit_path, index=False)
    print(f"Wrote {len(manifest)} rows to {args.output}")
    print(manifest.groupby(["split", "label"]).size().rename("count"))
    print(f"Wrote {len(real_excluded) + len(ai_excluded)} excluded records to {audit_path}")


if __name__ == "__main__":
    main()

