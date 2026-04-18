
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

ROOT = Path(".")
OUT = ROOT / "data" / "english_v2_stageA"
IMG_DIR = OUT / "images"
CTRL_MASK_DIR = OUT / "control_mask"
CTRL_BLANK_DIR = OUT / "control_blank"

for d in [IMG_DIR, CTRL_MASK_DIR, CTRL_BLANK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
FONT_PX = 340

TRAIN_LETTERS = list("abcdefghijklmnopqrstuvw")
VAL_LETTERS = list("xyz")

TARGET_FONTS = {
    "Arial": "assets/fonts/arial.ttf",
    "Courier New": "assets/fonts/cour.ttf",
}

STYLE_PROMPTS = {
    "Arial": "Arial style, clean sans-serif printed font, black glyph on white background",
    "Courier New": "Courier New style, monospace typewriter font, black glyph on white background",
}

SOURCE_FONT_NAME = "Arial"
SOURCE_FONT_PATH = TARGET_FONTS[SOURCE_FONT_NAME]


def slugify(x: str) -> str:
    return x.lower().replace(" ", "_").replace("-", "_")


def render_letter(letter: str, font_path: str, size: int = 512, font_px: int = 340) -> Image.Image:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_px)

    bbox = draw.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), letter, fill=(0, 0, 0), font=font)
    return img


def make_filled_mask(letter: str, font_path: str, size: int = 512) -> Image.Image:
    img = render_letter(letter, font_path, size=size, font_px=FONT_PX)
    gray = np.array(img.convert("L"))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rgb = np.stack([bw, bw, bw], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


def make_blank(size: int = 512) -> Image.Image:
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    return Image.fromarray(arr)


def build_rows(split_name: str, letters: list[str]):
    mask_rows = []
    mixed_rows = []

    blank_path = CTRL_BLANK_DIR / f"{split_name}_blank.png"
    if not blank_path.exists():
        make_blank(IMAGE_SIZE).save(blank_path)

    for font_name, font_path in TARGET_FONTS.items():
        font_slug = slugify(font_name)

        for letter in letters:
            name = f"{split_name}_{font_slug}_{letter}.png"

            image_path = IMG_DIR / name
            mask_path = CTRL_MASK_DIR / name

            target_img = render_letter(letter, font_path, size=IMAGE_SIZE, font_px=FONT_PX)
            mask_img = make_filled_mask(letter, SOURCE_FONT_PATH, size=IMAGE_SIZE)

            target_img.save(image_path)
            mask_img.save(mask_path)

            prompt = f"letter {letter}, {STYLE_PROMPTS[font_name]}"

            base_row = {
                "image": str(image_path.as_posix()),
                "text": prompt,
                "char": letter,
                "font_name": font_name,
                "control_source_font": SOURCE_FONT_NAME,
            }

            row_mask = dict(base_row)
            row_mask["conditioning_image"] = str(mask_path.as_posix())
            row_mask["control_type"] = "mask"
            mask_rows.append(row_mask)

            row_mixed_mask = dict(base_row)
            row_mixed_mask["conditioning_image"] = str(mask_path.as_posix())
            row_mixed_mask["control_type"] = "mask"
            mixed_rows.append(row_mixed_mask)

            row_mixed_blank = dict(base_row)
            row_mixed_blank["conditioning_image"] = str(blank_path.as_posix())
            row_mixed_blank["control_type"] = "blank"
            mixed_rows.append(row_mixed_blank)

    return mask_rows, mixed_rows


train_mask_rows, train_mixed_rows = build_rows("train", TRAIN_LETTERS)
val_mask_rows, val_mixed_rows = build_rows("val", VAL_LETTERS)


def dump_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


dump_jsonl(OUT / "metadata_train_stageA_mask.jsonl", train_mask_rows)
dump_jsonl(OUT / "metadata_val_stageA_mask.jsonl", val_mask_rows)
dump_jsonl(OUT / "metadata_train_stageA_mixed.jsonl", train_mixed_rows)
dump_jsonl(OUT / "metadata_val_stageA_mixed.jsonl", val_mixed_rows)

print("saved to:", OUT.resolve())
print("train mask:", len(train_mask_rows))
print("val mask:", len(val_mask_rows))
print("train mixed:", len(train_mixed_rows))
print("val mixed:", len(val_mixed_rows))
