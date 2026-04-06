# Context Transfer File (Updated April 2026)

This file is a full handoff context for continuing work on `gemma-4`, specifically focusing on the **ARC-Drone-Bench** pipeline applied to the **AU-AIR** real-world drone dataset.

## Repository Status

- Repo path: `/Users/robertbadinter/Documents/gemma-4`
- Git remote: `https://github.com/ArthurVigier/gemma-4.git`
- State: Evaluation pipeline stabilized. Transitioning to **Unsloth-optimized High-Speed VLM Fine-Tuning**.

## Project Evolution: The AU-AIR Drone Pipeline

The project has transitioned from synthetic ARC grids to real-world drone telemetry processing using the AU-AIR dataset. The goal is to predict drone navigation actions from a temporal window of consecutive aerial frames.

1.  **Native Multimodal Inputs:** The model analyzes a sequence of $T=4$ frames.
2.  **The "Gray Image" Bug Resolved:** We discovered that aggressive downscaling or using extreme 5D tensor structures caused the `gemma-4-e4b-it` Vision Encoder (SigLIP/ViT) to collapse or hallucinate "solid gray images." We resolved this by using a strictly resized 224x224 2x2 mosaic image, completely preserving the visual information.
3.  **Unsloth VLM Optimization:** Standard HuggingFace `BitsAndBytes` 4-bit quantization destructively quantizes the Vision Encoder, rendering the model blind. We have migrated the training pipeline to **Unsloth FastVisionModel**, which safely quantizes the LLM backbone while maintaining the Vision Tower in full precision (`bfloat16`), resulting in a 2x training speedup and drastically reduced VRAM usage.

## Target Architecture

- **Teacher Model:** `unsloth/gemma-4-e4b-it` (4B Multimodal).
- **Student Model:** `TRMReasoner` (Transformer Reasoning Module).
- **Dataset:** `auair_sequences.jsonl` (Parsed telemetry and cropped frames).
- **Input Format:** 448x448 Mosaic (composed of four 224x224 frames).
- **Output Format:** Structured Text chunk (`Action_0: X Halt_0: Y ...`).

## Infrastructure & Hardware Constraints

### Remote Pod Environment (PyTorch 2.6.0 Challenges)
The project is currently running on a remote pod with a cutting-edge environment (PyTorch `2.6.0`). This introduced severe compatibility issues with `torchao` and `unsloth_zoo`:
- **Missing `torch._inductor.config`:** Handled via an early-import monkey patch (`import torch._dynamo`).
- **Missing sub-byte dtypes (`torch.int1` to `int7`):** `torchao` iterating over non-existent dtypes caused fatal import errors. Handled by injecting mock dtypes into the `torch` namespace before loading Unsloth.
- **`infer_schema` ValueError:** PyTorch 2.6 strictly forbids `torch.dtype` as a default parameter value in C++ custom ops, breaking `torchao` compilation globally.

### The Virtual Environment Solution
To bypass the PyTorch 2.6.0 kernel incompatibilities without breaking the pod's global environment, Unsloth training must be executed inside a dedicated virtual environment running **PyTorch 2.5.1**.

## Active Files & Pipelines

### 1. Data Preparation
- `scripts/parse_auair.py`: Extracts and formats AU-AIR telemetry into JSONL sequences.

### 2. Teacher Fine-Tuning (Unsloth)
- `src/arc_drone/teacher_finetuning_unsloth.py`: The highly optimized Unsloth QLoRA training loop. Preserves vision encoder precision while fitting on a single 24GB GPU.
- `scripts/finetune_unsloth.py`: The CLI entrypoint for the Unsloth trainer.

### 3. Evaluation & Benchmarking
- `src/arc_drone/auair_eval.py`: The core benchmarking logic. Evaluates Vanilla Gemma, LoRA adapters (TTA), and the TRM Student. Uses native `bfloat16` to prevent evaluation blindness.
- `scripts/benchmark_auair.py`: CLI for running the evaluation suites.

## Important CLI Commands

### 1. Setting up the Unsloth Environment (Once per pod)
```bash
python -m venv unsloth_env
source unsloth_env/bin/activate
pip install --upgrade pip
pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-cache-dir unsloth_zoo trl
```

### 2. Unsloth Teacher Fine-Tuning (Inside venv)
```bash
# Ensure HF_HOME is set if the root disk is small
mkdir -p /workspace/gemma-4/.hf_cache
nohup env HF_HOME=/workspace/gemma-4/.hf_cache ./unsloth_env/bin/python scripts/finetune_unsloth.py \
    --auair-path data/auair_sequences.jsonl \
    --auair-images-path /workspace/gemma-4/images \
    --epochs 3 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 > logs/finetune_unsloth.log 2>&1 &
```

### 3. Benchmarking (Global Environment)
```bash
nohup python scripts/benchmark_auair.py \
    --sequences data/auair_sequences.jsonl \
    --images-path /workspace/gemma-4/images \
    --model-id google/gemma-4-e4b-it \
    --n-eval 300 > logs/benchmark_vanilla.log 2>&1 &
```

## Known "Gotchas" Fixed
- **The "Blind VLM" Bug:** Using standard `BitsAndBytesConfig(load_in_4bit=True)` on Gemma-4 aggressively quantizes the Vision Encoder, destroying the image embeddings and causing the LLM to perceive random noise as a "solid gray color." **Solution:** Use Unsloth `FastVisionModel` (which protects the vision tower) or run standard HF evaluations in pure `bfloat16`.
- **Transformers 5.5.0 Submodule Bug:** Missing `set_submodule` in older `torch.nn.Module` prevents adapter loading. Fixed via monkey-patching in the evaluation script.
- **Disk Space Exhaustion (`OS Error 28`):** Downloading Gemma-4 weights or installing PyTorch can fill the pod's root `/tmp` or cache partition. **Solution:** Delete large zips, use `export TMPDIR=/workspace/...` for pip, and `export HF_HOME=/workspace/...` for HuggingFace caching.

## One-Sentence Current State
The pipeline has successfully solved the VLM "gray image" hallucination bug by preserving vision encoder precision and has been migrated to an isolated PyTorch 2.5.1 / Unsloth environment for robust, high-speed multimodal fine-tuning on the AU-AIR dataset.
