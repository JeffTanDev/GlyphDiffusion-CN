from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from controlnet_aux import LineartDetector
from skimage.morphology import skeletonize

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


def pick_yahei_font_path() -> str | None:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def render_char_image(char: str, font_path: str, size: int = 512, font_size: int = 400) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=(0, 0, 0), font=font)
    return img


def to_lineart_hint(char_img: Image.Image, detector: LineartDetector, size: int = 512) -> Image.Image:
    hint = detector(char_img, detect_resolution=size, image_resolution=size)
    return hint.convert("RGB")


def to_skeleton_hint(char_img: Image.Image) -> Image.Image:
    gray = np.array(char_img.convert("L"))
    # OTSU binarization + morphology denoise
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    fg = bw > 0
    skel = skeletonize(fg)
    # Prune short spur branches by iterative endpoint removal
    skel_u8 = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(8):
        neigh = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = (skel_u8 == 1) & (neigh == 11)
        if not endpoints.any():
            break
        skel_u8[endpoints] = 0
    skel = skel_u8.astype(bool)
    # White background with black skeleton lines
    out = np.full_like(gray, 255, dtype=np.uint8)
    out[skel] = 0
    rgb = np.repeat(out[:, :, None], 3, axis=2)
    return Image.fromarray(rgb)


def make_contact_sheet(image_paths: list[Path], cols: int, out_path: Path) -> None:
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(imgs):
        x = (i % cols) * w
        y = (i // cols) * h
        canvas.paste(img, (x, y))
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: lineart hint vs skeleton hint")
    parser.add_argument("--char", type=str, default="龙")
    parser.add_argument("--style", type=str, default="行书")
    parser.add_argument("--lora_dir", type=Path, default=Path("checkpoints/lora-sd15-font"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/ablation_hint"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scales", type=str, default="0.55,0.70,0.85")
    parser.add_argument("--guidance_scale", type=float, default=9.0)
    parser.add_argument("--yahei_font", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]

    font_path = args.yahei_font or pick_yahei_font_path()
    if font_path is None:
        raise FileNotFoundError("未找到微软雅黑，请用 --yahei_font 指定字体路径")

    char_img = render_char_image(args.char, font_path)
    char_img.save(args.output_dir / "char_render_yahei.png")

    lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
    hint_lineart = to_lineart_hint(char_img, lineart_detector)
    hint_skeleton = to_skeleton_hint(char_img)
    hint_lineart.save(args.output_dir / "hint_lineart.png")
    hint_skeleton.save(args.output_dir / "hint_skeleton.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if not args.lora_dir.exists():
        raise FileNotFoundError(f"LoRA dir not found: {args.lora_dir}")
    pipe.load_lora_weights(str(args.lora_dir))

    prompt = f"a chinese character {args.char}, Chinese {args.style} style, clean background, high contrast"
    negative_prompt = "blurry, distorted, low quality, handwritten, artistic font"

    outputs: list[Path] = []
    for method, hint in [("lineart", hint_lineart), ("skeleton", hint_skeleton)]:
        for scale in scales:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            image = pipe(
                prompt=prompt,
                image=hint,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                controlnet_conditioning_scale=scale,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            out_name = f"out_{method}_scale_{scale:.2f}.png"
            out_path = args.output_dir / out_name
            image.save(out_path)
            outputs.append(out_path)
            print(f"saved: {out_path}")

    make_contact_sheet(outputs, cols=len(scales), out_path=args.output_dir / "compare_grid.png")
    print(f"done: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
