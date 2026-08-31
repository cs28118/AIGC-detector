"""Export SID_Set from Hugging Face into the project's real/fake folders.

The SID_Set labels are converted to the binary labels used by this project:

    0 (real)       -> real/
    1 (synthetic)  -> fake/
    2 (tampered)   -> fake/   (unless --full-synthetic-only is supplied)

Examples (run with system Python, not a virtual environment)::

    python -m pip install --user datasets
    python scripts/download_sid_set.py --limit 1000 --flat
    python scripts/download_sid_set.py

By default only the publicly available ``train`` and ``val`` splits are
exported. The test split is intentionally not downloaded because its labels
are withheld by the dataset authors.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def safe_name(value: Any) -> str:
    """Return a filesystem-safe image identifier."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name or "image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SID_Set into data/<name>/<split>/{real,fake}."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/sid_set"),
        help="Destination directory (default: data/sid_set).",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write to data/sid_set/{real,fake}; otherwise retain split subdirectories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="SID_Set splits to export (default: train val).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Export at most this many rows per split; useful for a smoke test.",
    )
    parser.add_argument(
        "--full-synthetic-only",
        action="store_true",
        help="Exclude label 2 tampered images instead of treating them as fake.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/cache each split before iterating (uses substantially more disk).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files already present in the destination.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def rows_for_split(split: str, streaming: bool) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required. Install it with "
            "'python -m pip install --user datasets'."
        ) from exc

    return load_dataset("saberzl/SID_Set", split=split, streaming=streaming)


def export_split(
    split: str,
    output_root: Path,
    limit: int | None,
    full_synthetic_only: bool,
    streaming: bool,
    overwrite: bool,
    flat: bool,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    dataset = rows_for_split(split, streaming)

    for row_number, row in enumerate(dataset):
        if limit is not None and row_number >= limit:
            break

        label = int(row["label"])
        if label == 0:
            category = "real"
        elif label == 1:
            category = "fake"
        elif label == 2 and not full_synthetic_only:
            category = "fake"
        elif label == 2:
            counts["excluded_tampered"] += 1
            continue
        else:
            raise ValueError(f"Unexpected SID_Set label {label} at row {row_number}")

        image_id = safe_name(row.get("img_id", f"row_{row_number:07d}"))
        if flat:
            # Prefixing the split keeps IDs unique while still matching the
            # project's data/<dataset>/{real,fake} convention.
            filename = f"{split}_{image_id}.png"
            destination = output_root / category / filename
        else:
            destination = output_root / split / category / f"{image_id}.png"
        if destination.exists() and not overwrite:
            counts["already_present"] += 1
            continue

        image = row["image"]
        if image is None:
            counts["missing_image"] += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(destination, format="PNG")
        counts[category] += 1

    return counts


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        counts = export_split(
            split=split,
            output_root=args.output_root,
            limit=args.limit,
            full_synthetic_only=args.full_synthetic_only,
            streaming=not args.no_streaming,
            overwrite=args.overwrite,
            flat=args.flat,
        )
        print(f"{split}: {dict(counts)}")

    print(f"Images written below: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
