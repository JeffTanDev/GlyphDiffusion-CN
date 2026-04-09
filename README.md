# 南开中文字体风格：SD 1.5 UNet LoRA + ControlNet（条件图一致训练）

本项目在 **Stable Diffusion 1.5** 上为 **UNet 训练 LoRA**，使生成结果带有指定中文字体风格；**ControlNet 权重全程冻结**，但训练与推理都会在 UNet 前向中接入同类型的 **条件图**，以减轻 train/infer 不一致。

---

## 设计要点

| 组件 | 说明 |
|------|------|
| **LoRA** | 仅加在 UNet（`to_q` / `to_k` / `to_v` / `to_out.0`），学习字体风格。 |
| **ControlNet** | 使用 `lllyasviel/control_v11p_sd15_lineart`，**不训练**；训练时把 `conditioning_image` 送入 ControlNet，残差注入 UNet。 |
| **条件图** | 与南开目标字形对应的 **微软雅黑单字 + 细化骨架（skeleton）**：OTSU、形态学去噪、小连通域过滤、细化、端点剪枝。训练集由 `build_nankai_pairs.py` 生成；推理由 `infer_controlnet_lora.py` 用同一思路现场生成。 |
| **目标图** | 南开数据集各风格文件夹中的 PNG，作为 VAE 编码的监督目标。 |

---

## 环境

在仓库根目录创建并激活虚拟环境，再安装依赖（详见 `requirements.txt` 文件头说明：建议先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装带 CUDA 的 `torch` / `torchvision`，再安装其余包）：

```text
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

需 **CUDA 版 PyTorch** 才能在本机 GPU 上高效训练/推理；仅 CPU 会较慢。

---

## 1. 生成训练数据（南开目标 + 雅黑 skeleton 控制图）

```text
python scripts/build_nankai_pairs.py --dataset_root "Nankai-Chinese-Font-Style-Dataset-master/Nankai-Chinese-Font-Style-Dataset-master/FontData"
```

可选：显式指定微软雅黑路径（默认尝试 `C:\Windows\Fonts\msyh.ttc`）：

```text
python scripts/build_nankai_pairs.py --yahei_font "D:\path\to\msyh.ttc"
```

输出目录 `data/processed/`：

| 路径 | 含义 |
|------|------|
| `images/` | 训练用目标图（与南开源图一致，重命名保存） |
| `target/` | 目标图副本，便于人工核对 |
| `control/` | **ControlNet 条件图**：雅黑渲染后的 skeleton 线稿 |
| `metadata_train.jsonl` / `metadata_val.jsonl` | 每行 JSON：`image`、`conditioning_image`、`text`、`char`、`font_name` 等 |

样本量以你当前数据集为准（此前全量构建约为训练约 1.4 万条、验证约 3500 条，以脚本打印为准）。

---

## 2. 训练 UNet LoRA（训练前向接入 ControlNet）

默认与配置文件一致：`runwayml/stable-diffusion-v1-5` + `lllyasviel/control_v11p_sd15_lineart`。

```text
python scripts/train_lora_sd15.py --train_metadata data/processed/metadata_train.jsonl --controlnet_model_name_or_path lllyasviel/control_v11p_sd15_lineart --output_dir checkpoints/lora-sd15-font --max_train_steps 4000 --train_batch_size 1 --gradient_accumulation_steps 4
```

说明：

- **损失**：在随机时间步对 latent 加噪，UNet（含 LoRA）在给定 **文本 + ControlNet 残差** 下预测噪声，与真实噪声做 MSE。
- **数值**：VAE / Text Encoder / ControlNet 可用 FP16；**带 LoRA 的 UNet 在 FP32 上优化**，减轻 NaN。
- **输出**：LoRA 权重写入 `--output_dir`（`save_lora_weights` 格式，供 `infer_controlnet_lora.py` 的 `load_lora_weights` 使用）。

建议先用较小 `--max_train_steps`（如 500～1000）做冒烟，再按 loss 曲线与生成效果加长。

可参考 `configs/train_lora_sd15.yaml` 中的参数说明（命令行会覆盖其中同名项；训练脚本当前以 CLI 为主）。

---

## 3. 推理（ControlNet Lineart + 雅黑 skeleton 条件图 + LoRA）

```text
python scripts/infer_controlnet_lora.py --char 人 --style Songti --lora_dir checkpoints/lora-sd15-font --output outputs/ren_songti.png
```

- 会在 `--output` 同目录生成 **`hint_infer.png`**（当前使用的 skeleton 条件图）。
- **已关闭** 内置 NSFW safety checker，避免字体实验被误判为敏感内容而输出黑图。
- **`--controlnet_conditioning_scale`**：默认 `1.2`。一般可在 **约 0～2+** 间调节（常见尝试 **0.8～1.6**）：偏大更贴骨架，偏小给风格更多空间；无绝对上限，过大可能导致过度约束或伪影。
- 骨架质量相关：`--morph_kernel`（默认 3）、`--spur_prune_iters`（默认 12）、`--min_component_area`（默认 40），与数据构建思路一致，可按需微调。

---

## 其他脚本（可选）

| 路径 | 作用 |
|------|------|
| `scripts/scan_nankai_dataset.py` | 扫描南开数据目录，统计图片等文件数量。 |
| `scripts/run_hint_ablation.py` | 对比不同 hint 与 `controlnet_conditioning_scale` 的消融实验。 |

---

## 文件索引

| 路径 | 作用 |
|------|------|
| `scripts/build_nankai_pairs.py` | 南开多风格目标图 + 雅黑 skeleton 控制图 + `metadata_*.jsonl` |
| `scripts/train_lora_sd15.py` | 带 ControlNet 条件前向的 UNet LoRA 训练 |
| `scripts/infer_controlnet_lora.py` | Lineart ControlNet + skeleton hint + LoRA 推理 |
| `configs/train_lora_sd15.yaml` | 训练超参与路径的 YAML 参考 |
| `ControlNet+_LoRA.ipynb` | 本地化后的交互示例（依赖已装于 `.venv`） |

---

## 预期与注意

- **风格**：模型通过 `text`（如 `Chinese {风格名} style`）区分南开中的不同字体；效果受数据量、训练步数、推理 `guidance_scale` 与 `controlnet_conditioning_scale` 共同影响。
- **一致性**：条件图务必与训练分布一致（雅黑 skeleton + 同一 ControlNet 型号），否则容易出现结构或风格异常。

若你之后需要「按步数自动抽样出图」做 early stopping，可在训练脚本外单独加验证脚本或回调。
