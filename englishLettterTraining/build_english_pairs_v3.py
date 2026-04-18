
from pathlib import Path
import json
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

ROOT = Path(".")
OUT = ROOT / "data" / "english_v4_stageA"
IMG_DIR = OUT / "images"
CTRL_OUTLINE_DIR = OUT / "control_outline"
CTRL_BLANK_DIR = OUT / "control_blank"

for d in [IMG_DIR, CTRL_OUTLINE_DIR, CTRL_BLANK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
FONT_PX = 340
SEED = 42
random.seed(SEED)

TRAIN_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
VAL_LETTERS = list("xyz")

TARGET_FONTS = {
    "Arial": "assets/fonts/arial.ttf",
    "Courier New": "assets/fonts/cour.ttf",
}

SOURCE_FONT_NAME = "Arial"
SOURCE_FONT_PATH = TARGET_FONTS[SOURCE_FONT_NAME]

HARD_LETTERS = {"e", "a", "g", "r", "z", "m"}

BASE_REPEAT = 10
HARD_REPEAT = 25

FONT_STYLE_PROMPTS = {
    "Arial": (
        "Arial style, clean sans-serif printed font, smooth terminals, "
        "no decorative tail, no serif, flat 2d black glyph, plain white background"
    ),
    "Courier New": (
        "Courier New style, monospace typewriter font, small terminal tails, "
        "typewriter-like endings, flat 2d black glyph, plain white background"
    ),
}

LETTER_SPECIFIC_HINTS = {
    "a": {
        "Arial": "double-storey lowercase a without a small lower-right tail",
        "Courier New": "double-storey lowercase a with a small lower-right tail and typewriter-like ending",
    },
    "e": {
        "Arial": "lowercase e with a rounded bowl and a clean horizontal crossbar",
        "Courier New": "lowercase e with a rounded bowl, a clear horizontal crossbar, and small terminal details",
    },
    "m": {
        "Arial": "lowercase m with smooth rounded arches and clean sans-serif ends",
        "Courier New": "lowercase m with narrower arches and typewriter-style terminal endings",
    },
    "z": {
        "Arial": "clean sans-serif lowercase z without extra terminal tails",
        "Courier New": "monospace lowercase z with small terminal tails and typewriter-like ending details",
    },
    "g": {
        "Arial": "clean printed lowercase g in a sans-serif style",
        "Courier New": "monospace lowercase g with typewriter-like ending details",
    },
    "r": {
        "Arial": "clean sans-serif lowercase r with simple shoulder",
        "Courier New": "monospace lowercase r with typewriter-style terminal ending",
    },
}

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

def make_outline(letter: str, font_path: str, size: int = 512) -> Image.Image:
    img = render_letter(letter, font_path, size=size, font_px=FONT_PX)
    gray = np.array(img.convert("L"))
    _, bw_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(bw_inv, 100, 200)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    outline = 255 - edges
    rgb = np.stack([outline, outline, outline], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))

def make_blank(size: int = 512) -> Image.Image:
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    return Image.fromarray(arr)

def build_prompt(letter: str, font_name: str) -> str:
    base = f"letter {letter}, {FONT_STYLE_PROMPTS[font_name]}"
    extra = LETTER_SPECIFIC_HINTS.get(letter, {}).get(font_name, "")
    if extra:
        return f"{base}, {extra}"
    return base

def dump_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_split(split_name: str, letters: list[str]):
    train_blank_rows = []
    preview_rows = []

    blank_path = CTRL_BLANK_DIR / f"{split_name}_blank.png"
    if not blank_path.exists():
        make_blank(IMAGE_SIZE).save(blank_path)

    for font_name, font_path in TARGET_FONTS.items():
        font_slug = slugify(font_name)

        for letter in letters:
            name = f"{split_name}_{font_slug}_{letter}.png"
            image_path = IMG_DIR / name
            outline_path = CTRL_OUTLINE_DIR / name

            target_img = render_letter(letter, font_path, size=IMAGE_SIZE, font_px=FONT_PX)
            outline_img = make_outline(letter, SOURCE_FONT_PATH, size=IMAGE_SIZE)

            target_img.save(image_path)
            outline_img.save(outline_path)

            prompt = build_prompt(letter, font_name)

            base_row = {
                "image": str(image_path.as_posix()),
                "text": prompt,
                "char": letter,
                "font_name": font_name,
                "control_source_font": SOURCE_FONT_NAME,
            }

            # preview rows: both blank and outline, for inference/debug
            row_preview_blank = dict(base_row)
            row_preview_blank["conditioning_image"] = str(blank_path.as_posix())
            row_preview_blank["control_type"] = "blank"
            preview_rows.append(row_preview_blank)

            row_preview_outline = dict(base_row)
            row_preview_outline["conditioning_image"] = str(outline_path.as_posix())
            row_preview_outline["control_type"] = "outline"
            preview_rows.append(row_preview_outline)

            # training rows: blank-only, but hard letters repeated more
            if split_name == "train":
                repeat_n = HARD_REPEAT if letter in HARD_LETTERS else BASE_REPEAT
            else:
                repeat_n = 1

            for rep in range(repeat_n):
                row_train = dict(base_row)
                row_train["conditioning_image"] = str(blank_path.as_posix())
                row_train["control_type"] = "blank"
                row_train["repeat_id"] = rep
                row_train["is_hard_letter"] = letter in HARD_LETTERS
                train_blank_rows.append(row_train)

    random.shuffle(train_blank_rows)
    return train_blank_rows, preview_rows

train_rows, train_preview_rows = build_split("train", TRAIN_LETTERS)
val_rows, val_preview_rows = build_split("val", VAL_LETTERS)

dump_jsonl(OUT / "metadata_train_v4_blankonly.jsonl", train_rows)
dump_jsonl(OUT / "metadata_val_v4_blankonly.jsonl", val_rows)
dump_jsonl(OUT / "metadata_preview_train_v4.jsonl", train_preview_rows)
dump_jsonl(OUT / "metadata_preview_val_v4.jsonl", val_preview_rows)

hard_train_rows = [r for r in train_rows if r["is_hard_letter"]]
normal_train_rows = [r for r in train_rows if not r["is_hard_letter"]]

print("saved to:", OUT.resolve())
print("train blank-only total:", len(train_rows))
print("val blank-only total:", len(val_rows))
print("train hard-letter rows:", len(hard_train_rows))
print("train normal-letter rows:", len(normal_train_rows))
print("preview train:", len(train_preview_rows))
print("preview val:", len(val_preview_rows))
