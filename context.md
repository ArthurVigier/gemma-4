# Context Transfer File (Updated April 2026)

This file is a full handoff context for continuing work on `gemma-4`.

## Repository Status

- Repo path: `/Users/robertbadinter/Documents/gemma-4`
- Git remote: `https://github.com/ArthurVigier/gemma-4.git`
- State: Infrastructure complete. Now focused on **Specialized Teacher Fine-tuning** and **Global Attention Student Architecture**.

## Project Evolution: From Generalist to Specialist

The project has transitioned from general distillation to a high-performance specialized pipeline:

1.  **Specialized Teacher:** Instead of distilling from a generalist Gemma, we now fine-tune the teacher (Gemma-4-E4B or 26B-A4B MoE) explicitly on the ARC-Drone-Bench using QLoRA.
2.  **Hybrid Multimodal SOTA:** The teacher now learns from **High-Fidelity PIL Images** of ARC grids + **Textual Serializations**. This provides holistic spatial intuition (Vision) and symbol-perfect precision (Text).
3.  **Global Attention Student:** The student encoder has been upgraded from a simple MLP to a `GridAttentionEncoder` with learned 2D positional embeddings and 4-head self-attention.

## Target Architecture (ARChitects/NVARC Inspired)

- **Teacher:** `google/gemma-4-e4b` (Dense/PLE) or `google/gemma-4-26b-a4b` (MoE).
- **Student:** `TRMReasoner` (Transformer Reasoning Module).
  - **Encoder:** `GridAttentionEncoder` (160 dim).
  - **Reasoning:** Recursive Self-Refinement via `GRUCell` (6 steps).
  - **Halting:** Adaptive halting based on task complexity.
- **Deployment:** `PyTorch -> ONNX -> TensorRT (INT8)` (7000+ FPS on A6000).

## Infrastructure & Hardware

### Remote Pods (RunPod / Vast.ai)
- **NVIDIA A100 (80GB) / H100:** Required for Gemma-4-26B MoE fine-tuning.
- **NVIDIA RTX A6000 (48GB):** Suitable for Gemma-4-E4B and Student distillation.
- **Python Compatibility:** Lowered to `>=3.11` in `pyproject.toml` for standard pod environments.
- **CUDA Configuration:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` used to manage fragmentation.

### MoE Optimization (Surgical QLoRA)
To fit the 26B MoE on a single 80GB card:
- **BFloat16 Casting:** Bypassed `prepare_model_for_kbit_training` (which forces float32) to manually cast all layers to `bfloat16`.
- **Flash Attention 2:** Explicitly enabled (`attn_implementation="flash_attention_2"`) for memory efficiency.
- **Gradient Accumulation:** Set to 16 for batch stability with batch size 1.

## Active Files & Pipelines

### 1. Teacher Fine-tuning (The "Specialist" Phase)
- `src/arc_drone/teacher_finetuning.py`: Standard Hybrid QLoRA (E4B).
- `src/arc_drone/teacher_finetuning_moe.py`: Memory-optimized Surgical MoE QLoRA (26B).
- `scripts/finetune_gemma_teacher_moe.py`: CLI for MoE specialization.

### 2. Distillation (The "Inheritance" Phase)
- `src/arc_drone/student_distillation.py`: Uses cached teacher internal features (Layer 17).
- `scripts/cache_teacher_targets.py`: Extracts and stores teacher "thought vectors" for 32k+ tasks.

### 3. Student Implementation
- `src/arc_drone/model.py`: Updated with `GridAttentionEncoder`.
- `src/arc_drone/config.py`: Default `hidden_size=160`.

## Current Benchmarks (Accuracy Cap)

- **Supervised Baseline:** ~0.51 (Limited by MLP encoder topology destruction).
- **Teacher Probe (Generalist):** ~0.82 (Signal exists but is noisy).
- **Target (Specialized):** Aiming for >0.90 after specialist teacher fine-tuning.

## Important CLI Commands

### Specialist Teacher Fine-tuning (A100/H100)
```bash
python3 scripts/finetune_gemma_teacher_moe.py \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --epochs 2
```

### Student Distillation (RTX A6000)
```bash
python3 scripts/distill_trm_student.py \
  --device cuda \
  --cache-dir artifacts/distillation_cache/specialist_e4b_l17 \
  --batch-size 32 \
  --epochs 100 \
  --hidden-size 160
```

## Known "Gotchas" Fixed
- **Gemma-4 Multimodal Tokenization:** Processor fails to auto-expand `<image>`. Solution: Manually inject 256 `<image>` tokens in the prompt.
- **Python Version:** Use `python3 -m pip install -e '.[training]'` (now works on 3.11).
- **CUDA Version:** If `torch.cuda.is_available()` is False, reinstall torch for the pod's specific driver (e.g., `whl/cu121`).

## One-Sentence Current State
The project has successfully moved to a **Multimodal Specialist Teacher** strategy to provide a crystalline reasoning signal to a new **Attention-based Student**, breaking the previous architectural plateaus.
