"""
训练 Stable Diffusion 1.5 的 UNet LoRA（带 ControlNet 条件图前向）。

- 仅更新 UNet 上的 LoRA 权重；VAE/TextEncoder/ControlNet 均冻结。
- 训练时显式使用 metadata 里的 conditioning_image（微软雅黑 skeleton）进入 ControlNet，
  以消除 train/infer mismatch。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict


class FontJsonlDataset(Dataset):
    def __init__(self, metadata_path: Path, tokenizer: CLIPTokenizer, resolution: int = 512) -> None:
        self.rows = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # keep in [0,1] for ControlNet condition image
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        conditioning_image = Image.open(row["conditioning_image"]).convert("RGB")
        pixel_values = self.transform(image)
        conditioning_pixel_values = self.cond_transform(conditioning_image)
        text = row["text"]
        input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "input_ids": input_ids,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SD1.5 UNet LoRA for font style (ControlNet not trained, use at inference only)"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default="lllyasviel/control_v11p_sd15_lineart")
    parser.add_argument("--train_metadata", type=Path, default=Path("data/processed/metadata_train.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/lora-sd15-font"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    vae.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    unet.to(device, dtype=torch.float32)
    controlnet.to(device, dtype=dtype)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable params (UNet LoRA only): {sum(p.numel() for p in trainable_params):,}")
    print(f"ControlNet condition model: {args.controlnet_model_name_or_path}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    dataset = FontJsonlDataset(args.train_metadata, tokenizer, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_steps = min(args.max_train_steps, args.num_train_epochs * updates_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_steps,
    )

    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="train")

    unet.train()
    while global_step < max_steps:
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            conditioning_pixel_values = batch["conditioning_pixel_values"].to(device=device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_pixel_values,
                    return_dict=False,
                )

            model_pred = unet(
                noisy_latents.to(dtype=torch.float32),
                timesteps,
                encoder_hidden_states.to(dtype=torch.float32),
                down_block_additional_residuals=[x.to(dtype=torch.float32) for x in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32),
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=float(loss.item() * args.gradient_accumulation_steps))

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(args.output_dir),
        unet_lora_layers=unet_lora_state_dict,
    )
    print(f"UNet LoRA saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
