from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FONT_EXTS = {".ttf", ".otf", ".ttc"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LABEL_EXTS = {".txt", ".csv", ".json", ".jsonl"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz"}


def scan(base_dir: Path) -> dict:
    all_files = [p for p in base_dir.rglob("*") if p.is_file()]
    ext_counter = Counter(p.suffix.lower() for p in all_files)

    fonts = [str(p) for p in all_files if p.suffix.lower() in FONT_EXTS]
    images = [str(p) for p in all_files if p.suffix.lower() in IMAGE_EXTS]
    labels = [str(p) for p in all_files if p.suffix.lower() in LABEL_EXTS]
    archives = [str(p) for p in all_files if p.suffix.lower() in ARCHIVE_EXTS]

    return {
        "base_dir": str(base_dir.resolve()),
        "total_files": len(all_files),
        "extensions": dict(sorted(ext_counter.items(), key=lambda x: x[0])),
        "font_files": fonts,
        "image_files": images,
        "label_files": labels,
        "archive_files": archives,
        "ready_for_training": len(fonts) > 0 or len(images) > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Nankai dataset folder")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("Nankai-Chinese-Font-Style-Dataset-master")
        / "Nankai-Chinese-Font-Style-Dataset-master",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_scan.json"))
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")

    report = scan(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Scan saved to: {args.output.resolve()}")
    print(f"Total files: {report['total_files']}")
    print(f"Font files: {len(report['font_files'])}")
    print(f"Image files: {len(report['image_files'])}")
    print(f"Label files: {len(report['label_files'])}")
    if not report["ready_for_training"]:
        print("Dataset is not ready: no font/image files found.")
        print("Please add *.ttf/*.otf/*.ttc or image dataset first.")


if __name__ == "__main__":
    main()
