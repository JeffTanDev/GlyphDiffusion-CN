
from pathlib import Path
import json
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

ROOT = Path(".")
OUT = ROOT / "data" / "english_v6_stageA"
IMG_DIR = OUT / "images"
CTRL_BLANK_DIR = OUT / "control_blank"
CTRL_OUTLINE_DIR = OUT / "control_outline"

for d in [IMG_DIR, CTRL_BLANK_DIR, CTRL_OUTLINE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

# 训练参数
FINAL_SIZE = 512
RENDER_SIZE = 1024
FONT_PX = 680

TRAIN_CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
VAL_CHARS = list("aemzAEMZ127")

TARGET_FONTS = {
    "Arial": "assets/fonts/arial.ttf",
    "Courier New": "assets/fonts/cour.ttf",
}

SOURCE_OUTLINE_FONT = "DejaVu Sans"
SOURCE_OUTLINE_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if not Path(SOURCE_OUTLINE_FONT_PATH).exists():
    # fallback
    SOURCE_OUTLINE_FONT = "Arial"
    SOURCE_OUTLINE_FONT_PATH = "assets/fonts/arial.ttf"

# 更能体现 Courier 风格或更难学的字符
HARD_CHARS = set(list("aegmrzAEMZ127IJLT"))
BASE_REPEAT = 8
HARD_REPEAT = 20

FONT_STYLE_PROMPTS = {
    "Arial": (
        "Arial style, clean sans-serif printed font, smooth terminals, "
        "no decorative tail, no serif, flat 2d black glyph, plain white background"
    ),
    "Courier New": (
        "Courier New style, monospace typewriter font, subtle typewriter characteristics, "
        "flat 2d black glyph, plain white background"
    ),
}

LETTER_SPECIFIC_HINTS = {
    "a": {
        "Arial": "double-storey lowercase a without a lower-right tail",
        "Courier New": "double-storey lowercase a with a subtle lower-right terminal",
    },
    "e": {
        "Arial": "lowercase e with rounded bowl and clear horizontal crossbar",
        "Courier New": "lowercase e with rounded bowl, clear horizontal crossbar, and subtle monospace proportions",
    },
    "m": {
        "Arial": "lowercase m with smooth rounded arches",
        "Courier New": "lowercase m with narrower arches and monospace proportions",
    },
    "z": {
        "Arial": "simple clean sans-serif lowercase z",
        "Courier New": "lowercase z with subtle monospace terminal structure",
    },
    "A": {
        "Arial": "uppercase A in a clean sans-serif style",
        "Courier New": "uppercase A in a monospace typewriter style",
    },
    "E": {
        "Arial": "uppercase E in a clean sans-serif style",
        "Courier New": "uppercase E in a monospace typewriter style",
    },
    "M": {
        "Arial": "uppercase M in a clean sans-serif style",
        "Courier New": "uppercase M in a monospace typewriter style",
    },
    "Z": {
        "Arial": "uppercase Z in a clean sans-serif style",
        "Courier New": "uppercase Z in a monospace typewriter style",
    },
    "1": {
        "Arial": "digit one in a clean sans-serif printed style",
        "Courier New": "digit one in a monospace typewriter style",
    },
    "2": {
        "Arial": "digit two in a clean sans-serif printed style",
        "Courier New": "digit two in a monospace typewriter style",
    },
    "7": {
        "Arial": "digit seven in a clean sans-serif printed style",
        "Courier New": "digit seven in a monospace typewriter style",
    },
    "I": {
        "Arial": "uppercase I in a clean sans-serif printed style",
        "Courier New": "uppercase I in a monospace typewriter style",
    },
    "J": {
        "Arial": "uppercase J in a clean sans-serif printed style",
        "Courier New": "uppercase J in a monospace typewriter style",
    },
    "L": {
        "Arial": "uppercase L in a clean sans-serif printed style",
        "Courier New": "uppercase L in a monospace typewriter style",
    },
    "T": {
        "Arial": "uppercase T in a clean sans-serif printed style",
        "Courier New": "uppercase T in a monospace typewriter style",
    },
}

def slugify(x: str) -> str:
    return x.lower().replace(" ", "_").replace("-", "_")

def render_letter_highres(char: str, font_path: str, size: int, font_px: int) -> Image.Image:
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_px)

    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]

    # 轻微随机平移，避免死记中心栅格
    jitter_x = random.randint(-6, 6)
    jitter_y = random.randint(-6, 6)
    draw.text((x + jitter_x, y + jitter_y), char, fill=0, font=font)
    return img

def downsample_clean(img_l: Image.Image, final_size: int = 512) -> Image.Image:
    img = img_l.resize((final_size, final_size), Image.Resampling.LANCZOS)
    return img.convert("RGB")

def make_blank(size: int = 512) -> Image.Image:
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    return Image.fromarray(arr)

def make_light_outline(char: str, font_path: str, render_size: int, font_px: int, final_size: int) -> Image.Image:
    img = render_letter_highres(char, font_path, size=render_size, font_px=font_px).resize((final_size, final_size), Image.Resampling.LANCZOS)
    gray = np.array(img)
    _, bw_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(bw_inv, 80, 160)

    out = np.full_like(edges, 255)
    out[edges > 0] = 205   # 更浅
    rgb = np.stack([out, out, out], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))

def build_prompt(char: str, font_name: str) -> str:
    base = f"character {char}, {FONT_STYLE_PROMPTS[font_name]}"
    extra = LETTER_SPECIFIC_HINTS.get(char, {}).get(font_name, "")
    if extra:
        return f"{base}, {extra}"
    return base

def dump_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_split(split_name: str, chars: list[str]):
    train_rows = []
    preview_rows = []

    blank_path = CTRL_BLANK_DIR / f"{split_name}_blank.png"
    if not blank_path.exists():
        make_blank(FINAL_SIZE).save(blank_path)

    for font_name, font_path in TARGET_FONTS.items():
        font_slug = slugify(font_name)

        for ch in chars:
            name = f"{split_name}_{font_slug}_{ord(ch)}.png"
            image_path = IMG_DIR / name
            outline_path = CTRL_OUTLINE_DIR / f"{split_name}_{ord(ch)}.png"

            # target image
            target_l = render_letter_highres(ch, font_path, size=RENDER_SIZE, font_px=FONT_PX)
            target_img = downsample_clean(target_l, FINAL_SIZE)
            target_img.save(image_path)

            # preview outline only，用于推理辅助，不进入训练
            if not outline_path.exists():
                outline_img = make_light_outline(ch, SOURCE_OUTLINE_FONT_PATH, RENDER_SIZE, FONT_PX, FINAL_SIZE)
                outline_img.save(outline_path)

            prompt = build_prompt(ch, font_name)

            base_row = {
                "image": str(image_path.as_posix()),
                "text": prompt,
                "char": ch,
                "font_name": font_name,
            }

            # preview rows: blank + outline
            row_preview_blank = dict(base_row)
            row_preview_blank["conditioning_image"] = str(blank_path.as_posix())
            row_preview_blank["control_type"] = "blank"
            preview_rows.append(row_preview_blank)

            row_preview_outline = dict(base_row)
            row_preview_outline["conditioning_image"] = str(outline_path.as_posix())
            row_preview_outline["control_type"] = "outline"
            preview_rows.append(row_preview_outline)

            # training rows: blank-only
            if split_name == "train":
                repeat_n = HARD_REPEAT if ch in HARD_CHARS else BASE_REPEAT
            else:
                repeat_n = 1

            for rep in range(repeat_n):
                row_train = dict(base_row)
                row_train["conditioning_image"] = str(blank_path.as_posix())
                row_train["control_type"] = "blank"
                row_train["repeat_id"] = rep
                row_train["is_hard_char"] = ch in HARD_CHARS
                train_rows.append(row_train)

    random.shuffle(train_rows)
    return train_rows, preview_rows

train_rows, train_preview_rows = build_split("train", TRAIN_CHARS)
val_rows, val_preview_rows = build_split("val", VAL_CHARS)

dump_jsonl(OUT / "metadata_train_v6_blankonly_expanded.jsonl", train_rows)
dump_jsonl(OUT / "metadata_val_v6_blankonly_expanded.jsonl", val_rows)
dump_jsonl(OUT / "metadata_preview_train_v6.jsonl", train_preview_rows)
dump_jsonl(OUT / "metadata_preview_val_v6.jsonl", val_preview_rows)

hard_rows = [r for r in train_rows if r["is_hard_char"]]
normal_rows = [r for r in train_rows if not r["is_hard_char"]]

print("saved to:", OUT.resolve())
print("source outline font:", SOURCE_OUTLINE_FONT)
print("train total:", len(train_rows))
print("train hard-char rows:", len(hard_rows))
print("train normal-char rows:", len(normal_rows))
print("val total:", len(val_rows))
print("preview train:", len(train_preview_rows))
print("preview val:", len(val_preview_rows))
