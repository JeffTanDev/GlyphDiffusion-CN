from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


def pick_yahei_font_path() -> str | None:
    """与数据构建一致：骨架使用微软雅黑（训练/推理同一套）。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def prune_skeleton_spurs(skel: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Iteratively remove endpoints to prune tiny spur branches."""
    skel_u8 = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(iterations):
        neigh = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        # Center=10 and exactly one neighbor => 11
        endpoints = (skel_u8 == 1) & (neigh == 11)
        if not endpoints.any():
            break
        skel_u8[endpoints] = 0
    return skel_u8.astype(bool)


def remove_small_components(bw: np.ndarray, min_area: int = 40) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cleaned = np.zeros_like(bw)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def build_clean_skeleton(
    gray: np.ndarray,
    morph_kernel: int = 3,
    spur_prune_iters: int = 12,
    min_component_area: int = 40,
) -> np.ndarray:
    # OTSU binarization: foreground text=True
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove tiny burrs and close tiny gaps
    morph_kernel = max(3, morph_kernel | 1)  # force odd and >=3
    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    bw = remove_small_components(bw, min_area=min_component_area)
    fg = bw > 0
    skel = skeletonize(fg)
    skel = prune_skeleton_spurs(skel, iterations=spur_prune_iters)
    return skel


def build_hint_image(
    char: str,
    font_file: str | None,
    size: int = 512,
    morph_kernel: int = 3,
    spur_prune_iters: int = 12,
    min_component_area: int = 40,
) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_path = font_file or pick_yahei_font_path()
    if font_path is None:
        raise FileNotFoundError("未找到微软雅黑 (msyh.ttc)。请用 --yahei_font 指定路径。")
    font = ImageFont.truetype(font_path, 400)

    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=(0, 0, 0), font=font)

    # Skeleton hint with denoise + spur pruning.
    gray = np.array(img.convert("L"))
    skel = build_clean_skeleton(
        gray,
        morph_kernel=morph_kernel,
        spur_prune_iters=spur_prune_iters,
        min_component_area=min_component_area,
    )
    out = np.full_like(gray, 255, dtype=np.uint8)
    out[skel] = 0
    rgb = np.repeat(out[:, :, None], 3, axis=2)
    return Image.fromarray(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with ControlNet (Lineart model) + Skeleton hint + LoRA")
    parser.add_argument("--char", type=str, default="龙")
    parser.add_argument("--style", type=str, default="Songti")
    parser.add_argument("--lora_dir", type=Path, default=Path("checkpoints/lora-sd15-font"))
    parser.add_argument("--output", type=Path, default=Path("outputs/infer_char.png"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--yahei_font", type=str, default=None, help="微软雅黑字体路径，默认 C:\\\\Windows\\\\Fonts\\\\msyh.ttc")
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.2,
        help="ControlNet 强度（已默认调高）。若你观察到骨架还不够稳，可试 1.2~1.6；若风格被压制，再回调到 0.8~1.1",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance，略提高(如9~12)有时能让 prompt 里的风格词更强",
    )
    parser.add_argument(
        "--spur_prune_iters",
        type=int,
        default=12,
        help="骨架端点剪枝迭代次数。更大=更少小分叉（建议 8~20）",
    )
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=40,
        help="去除小连通域面积阈值。更大=去噪更强（建议 20~80）",
    )
    parser.add_argument(
        "--morph_kernel",
        type=int,
        default=3,
        help="形态学核大小（奇数）。更大=更平滑（建议 3~5）",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    hint = build_hint_image(
        args.char,
        args.yahei_font,
        morph_kernel=args.morph_kernel,
        spur_prune_iters=args.spur_prune_iters,
        min_component_area=args.min_component_area,
    )
    hint.save(args.output.parent / "hint_infer.png")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(device)
    # Disable safety checker for local font-style experiments (avoid false positive black images).
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if args.lora_dir.exists():
        pipe.load_lora_weights(str(args.lora_dir))
    else:
        raise FileNotFoundError(f"LoRA dir not found: {args.lora_dir}")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt = f"a chinese character {args.char}, Chinese {args.style} style, clean background, high contrast"
    negative_prompt = "blurry, distorted, low quality, handwritten, artistic font"

    print(
        f"controlnet_conditioning_scale={args.controlnet_conditioning_scale}, "
        f"guidance_scale={args.guidance_scale}"
    )

    result = pipe(
        prompt=prompt,
        image=hint,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
