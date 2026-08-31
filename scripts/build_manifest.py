"""Create a group-safe, deduplicated manifest from approved image roots.

Example:
python scripts/build_manifest.py --real-root data/cifake/train/REAL \
  --ai-root data/cifake/train/FAKE --output manifests/cifake.csv \
  --source cifake --ai-generator stable_diffusion
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def stable_split(
    group_id: str,
    val_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> str:
    """Assign every member of a stable group to exactly one split."""
    key = f"{seed}:{group_id}".encode("utf-8")
    value = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < calibration_fraction:
        return "calibration"
    if value < calibration_fraction + val_fraction:
        return "val"
    return "train"


def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exclusion_file(path: str | None) -> set[str]:
    if path is None:
        return set()
    with open(path, encoding="utf-8") as handle:
        return {
            line.strip().replace("\\", "/")
            for line in handle
            if line.strip() and not line.startswith("#")
        }


def is_forbidden(path: Path, forbidden_roots: list[Path], exclusion_entries: set[str]) -> bool:
    resolved = path.resolve()
    if any(resolved.is_relative_to(root) for root in forbidden_roots):
        return True
    canonical = str(resolved).replace("\\", "/")
    return canonical in exclusion_entries or resolved.name in exclusion_entries


def derive_group_id(path: Path, root: Path, source: str, group_regex: str | None) -> str:
    relative_stem = path.relative_to(root).with_suffix("").as_posix()
    if group_regex is None:
        group = relative_stem
    else:
        match = re.search(group_regex, relative_stem)
        if match is None:
            raise ValueError(
                f"--group-regex did not match {relative_stem!r}; every image needs a stable group"
            )
        group = match.group(1) if match.lastindex else match.group(0)
    return f"{source}:{group}"


def collect(
    root: Path,
    label: int,
    source: str,
    generator: str,
    forbidden_roots: list[Path],
    exclusions: set[str],
    val_fraction: float,
    calibration_fraction: float,
    seed: int,
    group_regex: str | None,
    hash_content: bool,
) -> tuple[list[dict], list[dict]]:
    rows, excluded = [], []
    root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if is_forbidden(path, forbidden_roots, exclusions):
            excluded.append({"image_path": str(path.resolve()), "reason": "forbidden_validation"})
            continue
        group_id = derive_group_id(path, root, source, group_regex)
        rows.append(
            {
                "image_path": str(path.resolve()),
                "label": label,
                "source": source,
                "generator": generator,
                "group_id": group_id,
                "content_sha256": content_sha256(path) if hash_content else "",
                "split": stable_split(group_id, val_fraction, calibration_fraction, seed),
            }
        )
    return rows, excluded


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove exact duplicates and reject contradictory duplicate labels."""
    kept, excluded = [], []
    seen: dict[str, dict] = {}
    for row in rows:
        digest = row["content_sha256"]
        if not digest:
            kept.append(row)
            continue
        previous = seen.get(digest)
        if previous is None:
            seen[digest] = row
            kept.append(row)
            continue
        if previous["label"] != row["label"]:
            raise ValueError(
                "Identical image content has contradictory labels: "
                f"{previous['image_path']} and {row['image_path']}"
            )
        excluded.append(
            {
                "image_path": row["image_path"],
                "reason": f"duplicate_content:{previous['image_path']}",
            }
        )
    return kept, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a group-safe AIGC image manifest.")
    parser.add_argument("--real-root", required=True, type=Path)
    parser.add_argument("--ai-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Stable dataset/source name.")
    parser.add_argument("--ai-generator", default="unknown_ai", help="Generator name stored for AI rows.")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-regex",
        help="Optional regex applied to each root-relative path; the first capture is group_id.",
    )
    parser.add_argument("--forbidden-root", type=Path, action="append", default=[])
    parser.add_argument("--exclude-file", help="Newline-delimited paths or filenames to exclude.")
    parser.add_argument(
        "--skip-content-dedup",
        action="store_true",
        help="Skip SHA-256 exact-content deduplication to reduce manifest-build time.",
    )
    args = parser.parse_args()
    if args.val_fraction <= 0 or args.calibration_fraction <= 0:
        parser.error("--val-fraction and --calibration-fraction must be positive")
    if args.val_fraction + args.calibration_fraction >= 1:
        parser.error("validation and calibration fractions must sum to less than 1")
    if not args.real_root.is_dir() or not args.ai_root.is_dir():
        parser.error("Both --real-root and --ai-root must be existing directories.")

    forbidden = [path.resolve() for path in args.forbidden_root]
    exclusions = read_exclusion_file(args.exclude_file)
    common = {
        "source": args.source,
        "forbidden_roots": forbidden,
        "exclusions": exclusions,
        "val_fraction": args.val_fraction,
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
        "group_regex": args.group_regex,
        "hash_content": not args.skip_content_dedup,
    }
    real_rows, real_excluded = collect(
        root=args.real_root, label=0, generator="real", **common
    )
    ai_rows, ai_excluded = collect(
        root=args.ai_root, label=1, generator=args.ai_generator, **common
    )
    rows, duplicate_excluded = deduplicate(real_rows + ai_rows)
    manifest = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if manifest.empty or manifest.label.nunique() != 2:
        raise ValueError("The resulting manifest must contain both real and AI images.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    audit_path = args.output.with_suffix(".excluded.csv")
    excluded = real_excluded + ai_excluded + duplicate_excluded
    pd.DataFrame(excluded, columns=["image_path", "reason"]).to_csv(audit_path, index=False)
    print(f"Wrote {len(manifest)} rows to {args.output}")
    print(manifest.groupby(["split", "label"]).size().rename("count"))
    print(f"Groups: {manifest.group_id.nunique()}; excluded records: {len(excluded)}")
    print(f"Wrote exclusion audit to {audit_path}")


if __name__ == "__main__":
    main()
