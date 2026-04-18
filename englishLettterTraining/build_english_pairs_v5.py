
from pathlib import Path
import json
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2

ROOT = Path(".")
OUT = ROOT / "data" / "english_v5_stageA"
IMG_DIR = OUT / "images"
CTRL_WEAK_DIR = OUT / "control_weak"
CTRL_BLANK_DIR = OUT / "control_blank"

for d in [IMG_DIR, CTRL_WEAK_DIR, CTRL_BLANK_DIR]:
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

# source control font pool: 自动过滤不存在的字体
SOURCE_FONT_CANDIDATES = [
    ("Arial", "assets/fonts/arial.ttf"),
    ("Nimbus Sans", "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"),
    ("DejaVu Sans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ("Liberation Sans", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
]

SOURCE_FONT_POOL = [(n, p) for (n, p) in SOURCE_FONT_CANDIDATES if Path(p).exists()]
if len(SOURCE_FONT_POOL) < 2:
    raise RuntimeError(f"Need at least 2 source control fonts, found: {SOURCE_FONT_POOL}")

HARD_LETTERS = {"e", "a", "g", "r", "z", "m"}
PATCH_LETTERS = {"a", "e", "m", "z"}

BASE_REPEAT = 10
HARD_REPEAT = 25
WEAKCONTROL_RATIO = 0.15   # 15% weak-control, 85% blank

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

def make_blank(size: int = 512) -> Image.Image:
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    return Image.fromarray(arr)

def make_outline_mask(letter: str, font_path: str, size: int = 512, font_px: int = 340) -> np.ndarray:
    img = render_letter(letter, font_path, size=size, font_px=font_px)
    gray = np.array(img.convert("L"))
    _, bw_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(bw_inv, 100, 200)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges

def mask_to_light_rgb(mask: np.ndarray, line_gray: int = 185) -> Image.Image:
    out = np.full(mask.shape, 255, dtype=np.uint8)
    out[mask > 0] = line_gray
    rgb = np.stack([out, out, out], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))

def get_glyph_bbox(letter: str, font_path: str, size: int = 512, font_px: int = 340):
    img = render_letter(letter, font_path, size=size, font_px=font_px)
    gray = np.array(img.convert("L"))
    ys, xs = np.where(gray < 250)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, size, size)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (x0, y0, x1 + 1, y1 + 1)

def crop_relative(img_arr: np.ndarray, bbox, rx0, ry0, rx1, ry1):
    x0, y0, x1, y1 = bbox
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    cx0 = x0 + int(rx0 * w)
    cy0 = y0 + int(ry0 * h)
    cx1 = x0 + int(rx1 * w)
    cy1 = y0 + int(ry1 * h)
    cx0 = max(0, min(img_arr.shape[1] - 1, cx0))
    cy0 = max(0, min(img_arr.shape[0] - 1, cy0))
    cx1 = max(cx0 + 1, min(img_arr.shape[1], cx1))
    cy1 = max(cy0 + 1, min(img_arr.shape[0], cy1))
    return img_arr[cy0:cy1, cx0:cx1]

PATCH_ROIS = {
    "a": [(0.50, 0.45, 1.00, 1.00), (0.30, 0.00, 1.00, 0.50)],
    "e": [(0.20, 0.20, 0.95, 0.75), (0.35, 0.35, 1.00, 0.95)],
    "m": [(0.00, 0.00, 1.00, 0.45), (0.55, 0.45, 1.00, 1.00)],
    "z": [(0.00, 0.00, 1.00, 0.35), (0.00, 0.65, 1.00, 1.00)],
}

def compose_weak_control(letter: str, source_font_path: str, size: int = 512) -> Image.Image:
    main_mask = make_outline_mask(letter, source_font_path, size=size, font_px=FONT_PX)
    main_img = mask_to_light_rgb(main_mask, line_gray=188)

    # 稍微模糊一下，让它更像弱提示
    main_img = main_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(main_img, (0, 0))

    # 对 hard letters 添加局部 patch insets
    if letter in PATCH_LETTERS:
        bbox = get_glyph_bbox(letter, source_font_path, size=size, font_px=FONT_PX)
        patch_size = 78
        y = size - patch_size - 14
        xs = [16, size - patch_size - 16]

        for i, roi in enumerate(PATCH_ROIS[letter]):
            patch_mask = crop_relative(main_mask, bbox, *roi)
            patch = mask_to_light_rgb(patch_mask, line_gray=176).resize((patch_size, patch_size), Image.Resampling.BILINEAR)
            patch = patch.filter(ImageFilter.GaussianBlur(radius=0.4))
            # 不加黑框，只给很淡的灰背景框
            patch_bg = Image.new("RGB", (patch_size, patch_size), (250, 250, 250))
            patch_bg.paste(patch, (0, 0))
            canvas.paste(patch_bg, (xs[i], y))

    return canvas

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

def get_or_create_blank(split_name: str):
    blank_path = CTRL_BLANK_DIR / f"{split_name}_blank.png"
    if not blank_path.exists():
        make_blank(IMAGE_SIZE).save(blank_path)
    return blank_path

def get_or_create_weak(split_name: str, source_name: str, source_path: str, letter: str):
    source_slug = slugify(source_name)
    weak_path = CTRL_WEAK_DIR / f"{split_name}_{source_slug}_{letter}.png"
    if not weak_path.exists():
        img = compose_weak_control(letter, source_path, size=IMAGE_SIZE)
        img.save(weak_path)
    return weak_path

def build_split(split_name: str, letters: list[str]):
    train_rows = []
    preview_rows = []
    blank_path = get_or_create_blank(split_name)

    # 先预生成所有 source-font × letter 的 weak control
    weak_lookup = {}
    for source_name, source_path in SOURCE_FONT_POOL:
        for letter in letters:
            weak_lookup[(source_name, letter)] = get_or_create_weak(split_name, source_name, source_path, letter)

    for font_name, font_path in TARGET_FONTS.items():
        font_slug = slugify(font_name)

        for letter in letters:
            name = f"{split_name}_{font_slug}_{letter}.png"
            image_path = IMG_DIR / name

            if not image_path.exists():
                target_img = render_letter(letter, font_path, size=IMAGE_SIZE, font_px=FONT_PX)
                target_img.save(image_path)

            prompt = build_prompt(letter, font_name)

            base_row = {
                "image": str(image_path.as_posix()),
                "text": prompt,
                "char": letter,
                "font_name": font_name,
            }

            # preview: 一个 blank + 每个 source font 一个 weak-control
            row_preview_blank = dict(base_row)
            row_preview_blank["conditioning_image"] = str(blank_path.as_posix())
            row_preview_blank["control_type"] = "blank"
            row_preview_blank["source_control_font"] = "blank"
            preview_rows.append(row_preview_blank)

            for source_name, source_path in SOURCE_FONT_POOL:
                row_preview_weak = dict(base_row)
                row_preview_weak["conditioning_image"] = str(weak_lookup[(source_name, letter)].as_posix())
                row_preview_weak["control_type"] = "weakcontrol"
                row_preview_weak["source_control_font"] = source_name
                preview_rows.append(row_preview_weak)

            # train: hard letters upweighted
            repeat_n = HARD_REPEAT if letter in HARD_LETTERS else BASE_REPEAT
            if split_name != "train":
                repeat_n = 1

            for rep in range(repeat_n):
                row_train = dict(base_row)
                row_train["repeat_id"] = rep
                row_train["is_hard_letter"] = letter in HARD_LETTERS

                use_weak = (random.random() < WEAKCONTROL_RATIO) if split_name == "train" else True
                if use_weak:
                    source_name, source_path = random.choice(SOURCE_FONT_POOL)
                    row_train["conditioning_image"] = str(weak_lookup[(source_name, letter)].as_posix())
                    row_train["control_type"] = "weakcontrol"
                    row_train["source_control_font"] = source_name
                else:
                    row_train["conditioning_image"] = str(blank_path.as_posix())
                    row_train["control_type"] = "blank"
                    row_train["source_control_font"] = "blank"

                train_rows.append(row_train)

    random.shuffle(train_rows)
    return train_rows, preview_rows

train_rows, train_preview_rows = build_split("train", TRAIN_LETTERS)
val_rows, val_preview_rows = build_split("val", VAL_LETTERS)

dump_jsonl(OUT / "metadata_train_v5_blankmain8515.jsonl", train_rows)
dump_jsonl(OUT / "metadata_val_v5_blankmain8515.jsonl", val_rows)
dump_jsonl(OUT / "metadata_preview_train_v5.jsonl", train_preview_rows)
dump_jsonl(OUT / "metadata_preview_val_v5.jsonl", val_preview_rows)

hard_train_rows = [r for r in train_rows if r["is_hard_letter"]]
normal_train_rows = [r for r in train_rows if not r["is_hard_letter"]]
weak_rows = [r for r in train_rows if r["control_type"] == "weakcontrol"]
blank_rows = [r for r in train_rows if r["control_type"] == "blank"]

print("saved to:", OUT.resolve())
print("source font pool:", [x[0] for x in SOURCE_FONT_POOL])
print("train total:", len(train_rows))
print("train hard-letter rows:", len(hard_train_rows))
print("train normal-letter rows:", len(normal_train_rows))
print("train weakcontrol rows:", len(weak_rows))
print("train blank rows:", len(blank_rows))
print("val total:", len(val_rows))
print("preview train:", len(train_preview_rows))
print("preview val:", len(val_preview_rows))
