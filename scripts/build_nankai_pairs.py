from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 与推理一致：ControlNet 条件图固定为「微软雅黑」字形 + skeleton 细线骨架
DEFAULT_YAHEI_WINDOWS = Path(r"C:\Windows\Fonts\msyh.ttc")


def resolve_yahei_font(font_path: Path | None) -> Path:
    if font_path is not None and font_path.exists():
        return font_path
    if DEFAULT_YAHEI_WINDOWS.exists():
        return DEFAULT_YAHEI_WINDOWS
    raise FileNotFoundError(
        "未找到微软雅黑字体。请安装 Windows 微软雅黑，或用 --yahei_font 指定 .ttc/.ttf 路径。"
    )


def prune_skeleton_spurs(skel: np.ndarray, iterations: int = 12) -> np.ndarray:
    skel_u8 = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(iterations):
        neigh = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = (skel_u8 == 1) & (neigh == 11)
        if not endpoints.any():
            break
        skel_u8[endpoints] = 0
    return skel_u8.astype(bool)


def remove_small_components(bw: np.ndarray, min_area: int = 40) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cleaned = np.zeros_like(bw)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def render_yahei_then_skeleton(
    char: str,
    font_file: Path,
    size: int = 512,
    font_size: int = 400,
) -> Image.Image:
    """白底黑字微软雅黑渲染，再细化为 skeleton 线稿，供 ControlNet 训练/推理一致使用。"""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_file), font_size)
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=(0, 0, 0), font=font)

    gray = np.array(img.convert("L"))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    bw = remove_small_components(bw, min_area=40)
    skel = skeletonize(bw > 0)
    skel = prune_skeleton_spurs(skel, iterations=12)

    out = np.full_like(gray, 255, dtype=np.uint8)
    out[skel] = 0
    rgb = np.repeat(out[:, :, None], 3, axis=2)
    return Image.fromarray(rgb)


def collect_style_dirs(split_dir: Path) -> list[Path]:
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="南开多字体目标图 + 微软雅黑 skeleton 控制图，用于带 ControlNet 条件的 LoRA 训练"
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("Nankai-Chinese-Font-Style-Dataset-master")
        / "Nankai-Chinese-Font-Style-Dataset-master"
        / "FontData",
    )
    parser.add_argument(
        "--yahei_font",
        type=Path,
        default=None,
        help="微软雅黑字体路径，默认尝试 C:\\Windows\\Fonts\\msyh.ttc",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    yahei = resolve_yahei_font(args.yahei_font)

    train_dir = args.dataset_root / "train"
    val_dir = args.dataset_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/val under: {args.dataset_root}")

    out_images = args.output_dir / "images"
    out_control = args.output_dir / "control"
    out_target = args.output_dir / "target"
    for d in (out_images, out_control, out_target):
        d.mkdir(parents=True, exist_ok=True)

    def build_split(split_name: str) -> list[dict]:
        split_dir = args.dataset_root / split_name
        style_dirs = collect_style_dirs(split_dir)
        rows: list[dict] = []
        idx = 0

        all_tasks: list[tuple[Path, Path]] = []
        for style_dir in style_dirs:
            for target_file in sorted(style_dir.iterdir()):
                if not target_file.is_file() or target_file.suffix.lower() not in IMAGE_EXTS:
                    continue
                all_tasks.append((style_dir, target_file))

        for style_dir, target_file in tqdm(all_tasks, desc=f"build_{split_name}"):
            style_name = style_dir.name
            char_name = target_file.stem
            try:
                control_img = render_yahei_then_skeleton(char_name, yahei, size=args.image_size)
            except Exception:
                continue

            target_img = Image.open(target_file).convert("RGB")

            name = f"{split_name}_{idx:08d}.png"
            image_path = out_images / name
            target_path = out_target / name
            control_path = out_control / name
            target_img.save(image_path)
            target_img.save(target_path)
            control_img.save(control_path)

            rows.append(
                {
                    "image": str(image_path.as_posix()),
                    "conditioning_image": str(control_path.as_posix()),
                    "target_image": str(target_path.as_posix()),
                    "text": f"a chinese character {char_name}, Chinese {style_name} style",
                    "char": char_name,
                    "font_name": style_name,
                    "control_skeleton": "Microsoft YaHei skeleton",
                }
            )
            idx += 1
        return rows

    train_rows = build_split("train")
    val_rows = build_split("val")

    train_file = args.output_dir / "metadata_train.jsonl"
    val_file = args.output_dir / "metadata_val.jsonl"
    with train_file.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with val_file.open("w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"yahei_font={yahei.resolve()}")
    print(f"train samples={len(train_rows)}")
    print(f"val samples={len(val_rows)}")
    print(f"train metadata={train_file.resolve()}")
    print(f"val metadata={val_file.resolve()}")


if __name__ == "__main__":
    main()
