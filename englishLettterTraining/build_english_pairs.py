from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

ROOT = Path(".")
OUT = ROOT / "data" / "english_processed"
IMG_DIR = OUT / "images"
CTRL_DIR = OUT / "control"

for d in [IMG_DIR, CTRL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
LETTERS_TRAIN = list("abcd")
LETTERS_VAL = list("e")

FONTS = {
    "Arial": "assets/fonts/arial.ttf",
    "Comic Sans MS": "assets/fonts/comic.ttf",
    "Courier New": "assets/fonts/cour.ttf",
}

SOURCE_FONT_NAME = "Arial"
SOURCE_FONT_PATH = FONTS[SOURCE_FONT_NAME]


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


def make_lineart_hint(letter: str, font_path: str, size: int = 512) -> Image.Image:
    img = render_letter(letter, font_path, size=size)
    gray = np.array(img.convert("L"))

    # binary glyph
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # contour edge from source font
    edges = cv2.Canny(bw, 100, 200)

    # make black lines on white background
    hint = 255 - edges
    hint_rgb = np.stack([hint, hint, hint], axis=-1)
    return Image.fromarray(hint_rgb.astype(np.uint8))


def build_split(split_name: str, letters: list[str]):
    rows = []
    idx = 0

    for font_name, font_path in FONTS.items():
        for letter in letters:
            target_img = render_letter(letter, font_path, size=IMAGE_SIZE)
            control_img = make_lineart_hint(letter, SOURCE_FONT_PATH, size=IMAGE_SIZE)

            name = f"{split_name}_{idx:06d}.png"
            image_path = IMG_DIR / name
            control_path = CTRL_DIR / name

            target_img.save(image_path)
            control_img.save(control_path)

            rows.append({
                "image": str(image_path.as_posix()),
                "conditioning_image": str(control_path.as_posix()),
                "text": f"letter {letter}, {font_name} style",
                "char": letter,
                "font_name": font_name,
                "control_source_font": SOURCE_FONT_NAME,
            })
            idx += 1
    return rows


train_rows = build_split("train", LETTERS_TRAIN)
val_rows = build_split("val", LETTERS_VAL)

with (OUT / "metadata_train.jsonl").open("w", encoding="utf-8") as f:
    for row in train_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with (OUT / "metadata_val.jsonl").open("w", encoding="utf-8") as f:
    for row in val_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("train:", len(train_rows))
print("val:", len(val_rows))
print("saved to:", OUT.resolve())
