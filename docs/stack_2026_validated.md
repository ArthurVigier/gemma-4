# Validated 2026 Stack

Validation date: 2026-04-03

This document fixes the recommended `ARC-Drone-Bench` stack using official or primary sources checked on 2026-04-03.

## Executive Summary

- recommended foundation family: `Gemma 4`
- edge variants: `Gemma 4 E2B` and `Gemma 4 E4B`
- default simulation stack: `ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL`
- secondary backend only: `AirSim`
- training and distillation stack: `Transformers + PEFT/LoRA + BitsAndBytes + Unsloth`
- deployment path: `PyTorch -> ONNX -> TensorRT 10+ / TensorRT-LLM`
- recommended quantization: `INT8` or `INT4` with `NNCF` or data-aware PyTorch quantization
- recommended cloud environment for fine-tuning, distillation, and final export:
  `A100 40/80GB` minimum, or `H100 80GB`

## Recommended Choices

### 1. Models

- reference base: `Gemma 4`
- edge variants: `E2B`, `E4B`
- final student target:
  - preferred target `< 50M` parameters
  - acceptable ceiling `<= 100M` after `INT4/INT8` quantization
- architectural constraints to keep:
  - recursive self-refinement is required
  - adaptive halting is required
  - ONNX export followed by TensorRT inference is required

Rationale:

- Google announced `Gemma 4` on `2026-04-02`, with `E2B`, `E4B`, `26B MoE`, and `31B Dense`, plus day-one support for `Hugging Face`, `Transformers.js`, `Unsloth`, `vLLM`, `llama.cpp`, `Ollama`, `NVIDIA NIM`, and more.

Sources:

- [Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Gemma docs landing page](https://ai.google.dev/gemma/docs)
- [Gemma releases](https://ai.google.dev/gemma/docs/releases)

### 2. Simulation and Middleware

Default stack:

- `Ubuntu 24.04`
- `ROS 2 Jazzy`
- `Gazebo Harmonic`
- `PX4 SITL`
- `ros_gz`

Tolerated but non-default stack:

- `ROS 2 Humble + Gazebo Harmonic`

Stack to avoid for a new project:

- `ROS 2 Iron`

Rationale:

- Gazebo documentation explicitly recommends `ROS 2 Jazzy + Gazebo Harmonic` for new users
- `ROS 2 Iron` is listed as `EOL`
- `ROS 2 Humble + Gazebo Harmonic` is still possible, but marked `use with caution`

Sources:

- [Gazebo ROS installation guidance](https://gazebosim.org/docs/harmonic/ros_installation)
- [Gazebo Harmonic docs](https://gazebosim.org/docs/harmonic/install)
- [ROS 2 end-of-life list](https://docs.ros.org/en/jazzy/Releases/End-of-Life.html)

### 3. AirSim

Project rule:

- `AirSim` remains supported as a secondary backend or compatibility path
- `Gazebo Harmonic + PX4 SITL` is the primary backend

Rationale:

- the Microsoft repository indicates that the original AirSim is no longer actively updated
- PX4 documentation treats AirSim as community-supported and warns that support may not work with current PX4 versions

Sources:

- [AirSim repository](https://github.com/microsoft/AirSim)
- [PX4 AirSim documentation](https://docs.px4.io/main/en/sim_airsim/index.html)

### 4. Training, Distillation, and Compression

Recommended stack:

- `PyTorch 2.5+`
- `CUDA 12.4+`
- `Transformers`
- `PEFT`
- `BitsAndBytes`
- `Unsloth`
- `ONNX`
- `TensorRT 10+`
- `TensorRT-LLM`
- `NNCF`

Notes:

- `Unsloth` is explicitly listed in the Gemma 4 announcement as day-one supported
- `NNCF` remains actively maintained for `INT4/INT8`, PTQ, and QAT with LoRA
- `ONNX` remains the right intermediate format

Sources:

- [Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [TensorRT-LLM docs](https://docs.nvidia.com/tensorrt-llm/)
- [NNCF documentation](https://openvinotoolkit.github.io/nncf/)
- [NNCF releases](https://github.com/openvinotoolkit/nncf/releases)
- [ONNX intro](https://onnx.ai/onnx/intro/)
- [ONNX releases](https://github.com/onnx/onnx/releases)
- [Transformers installation](https://huggingface.co/docs/transformers/en/installation)
- [PEFT quicktour](https://huggingface.co/docs/peft/main/quicktour)
- [bitsandbytes quantization docs](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)
- [Unsloth docs](https://docs.unsloth.ai/)

### 5. ARC Prize Reference Architectures

Research components confirmed as relevant:

- `TRM` for compact recursive reasoning
- `HRM` as a recurrent / hierarchical reference
- `ARChitects` for:
  - autoregressive reasoning
  - recursive masked diffusion
  - 2D positional encoding inspired by `Golden Gate RoPE`
  - recursive latent refinement

Sources:

- [TRM paper](https://arxiv.org/abs/2510.04871)
- [HRM paper](https://arxiv.org/abs/2506.21734)
- [ARChitects technical report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)
- [ARC Prize 2026 competition](https://arcprize.org/competitions/2026)

## Cloud GPU Recommendation

For this stage, I recommend renting a GPU with at least 40 GB of VRAM, for example an A100 40/80GB on RunPod or Vast.ai.

Why:

- Gemma 4 fine-tuning
- teacher -> student TRM-like distillation
- data-aware quantization calibration
- reliable ONNX / TensorRT export
- realistic latency benchmarking before the Jetson Orin Nano target

Useful platforms:

- [Runpod Cloud GPUs](https://www.runpod.io/product/cloud-gpus/)
- [Runpod GPU models](https://www.runpod.io/gpu-models)
- [Vast.ai](https://vast.ai/)
- [Vast.ai docs](https://docs.vast.ai/)

## Project Decisions to Enforce in Code

1. `ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL` becomes the default stack.
2. `AirSim` stays a secondary integration.
3. Any documentation that still suggests `Iron` as a current option must be removed or corrected.
4. Local hardware limitations must never be used to shrink the target architecture.
5. Heavy training and distillation must be explicitly marked as a `cloud GPU` workflow.
