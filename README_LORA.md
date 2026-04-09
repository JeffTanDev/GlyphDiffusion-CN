# 字体风格 LoRA（SD 1.5 UNet）+ ControlNet 条件图一致训练

## 设计说明

- **LoRA 只加在 SD1.5 的 UNet 上**，用于学习字体风格。
- **ControlNet 权重冻结**，但训练前向中会真实接入 `conditioning_image`。
- **训练与推理使用同分布骨架条件图**（微软雅黑 skeleton），减少 train/infer mismatch。

---

## 环境

```text
.\.venv\Scripts\activate
python -m pip install -U peft pyyaml
```

---

## 1) 生成训练 pair（南开字体目标 + 微软雅黑 skeleton 控制图）

```text
python scripts/build_nankai_pairs.py --dataset_root "Nankai-Chinese-Font-Style-Dataset-master/Nankai-Chinese-Font-Style-Dataset-master/FontData"
```

可选：

```text
python scripts/build_nankai_pairs.py --yahei_font "D:/path/to/msyh.ttc"
```

输出目录 `data/processed/`：

| 子目录/文件 | 含义 |
|-------------|------|
| `images/` | 目标字体图（训练主图） |
| `target/` | 目标图备份，便于核验 |
| `control/` | 微软雅黑 skeleton 控制图（训练/推理都使用） |
| `metadata_train.jsonl` / `metadata_val.jsonl` | 每行包含 `image`、`conditioning_image`、`text` 等字段 |

---

## 2) 训练 LoRA（带 ControlNet 条件图前向）

```text
python scripts/train_lora_sd15.py --train_metadata data/processed/metadata_train.jsonl --controlnet_model_name_or_path lllyasviel/control_v11p_sd15_lineart --output_dir checkpoints/lora-sd15-font --max_train_steps 4000 --train_batch_size 1 --gradient_accumulation_steps 4
```

建议先用 `--max_train_steps 1000` 冒烟，再拉长。

---

## 3) 推理（ControlNet + LoRA）

```text
python scripts/infer_controlnet_lora.py --char 人 --style Songti --lora_dir checkpoints/lora-sd15-font --output outputs/ren_songti.png
```

同目录会生成 `outputs/hint_infer.png`（微软雅黑 skeleton 条件图）。

---

## 文件作用速查

| 路径 | 作用 |
|------|------|
| `scripts/build_nankai_pairs.py` | 生成目标图 + skeleton 控制图 + metadata |
| `scripts/train_lora_sd15.py` | 带 ControlNet 条件图前向的 UNet LoRA 训练脚本 |
| `scripts/infer_controlnet_lora.py` | ControlNet + LoRA 推理脚本（骨架可调清理参数） |
| `configs/train_lora_sd15.yaml` | 训练参数参考 |
