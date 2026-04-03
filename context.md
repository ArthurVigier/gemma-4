# Context Transfer File

This file is a full handoff context for continuing work on `gemma-4` in another chat or with another LLM.

## Repository

- Repo path: `/Users/robertbadinter/Documents/gemma-4`
- Git remote: `https://github.com/ArthurVigier/gemma-4.git`
- Main branch is actively used
- Current work has been pushed multiple times to `main`

## Project Goal

The repository is an `ARC-Drone-Bench` scaffold:

- symbolic ARC-like grid reasoning for drone control
- target teacher family: `Gemma-4-E2B/E4B`
- target student: compact TRM-like recursive reasoner
- target deployment path: `PyTorch -> ONNX -> TensorRT`
- target runtime stack: `Gazebo Harmonic + PX4 SITL + ROS 2 Jazzy`

Current focus is not simulator integration anymore. The active problem is:

- how to distill useful teacher signal from Gemma into the compact exportable student

## Important High-Level Conclusion

Infrastructure is mostly solved.

What already works:

- GPU smoke
- ONNX export
- TensorRT engine build
- TensorRT engine validation
- supervised student training CLI
- Gemma hidden-layer sweep CLI
- Gemma-guided distillation CLI
- teacher-target cache pipeline for heavier distillation

Current bottleneck:

- model quality / distillation signal quality

## Environments In Use

### Local workstation

- macOS
- repo path: `/Users/robertbadinter/Documents/gemma-4`
- Python 3.12 locally
- tests run locally

### Remote pod

Observed in the session:

- host GPU: `NVIDIA RTX A6000`
- remote repo path: `/workspace/gemma-4`
- remote shell prompt looked like `root@6deca5e0faf1:/workspace/gemma-4#`
- remote Python version was effectively `3.11.10`

Important remote implication:

- `python3 -m pip install -e '.[training]'` failed because `pyproject.toml` requires `>=3.12`
- workaround used:
  - install training dependencies manually instead of editable install

Recommended manual install on pod:

```bash
python3 -m pip install \
  "transformers>=4.50" \
  "peft>=0.14" \
  "bitsandbytes>=0.45" \
  "onnx>=1.20"
```

## Remote SSH / SCP Details

The working SSH connection information used in the session was:

```bash
ssh root@38.147.83.15 -p 28909 -i ~/.ssh/id_ed25519
```

Example SCP command from local machine to download remote checkpoints:

```bash
scp -P 28909 -i ~/.ssh/id_ed25519 -r \
  root@38.147.83.15:/workspace/gemma-4/artifacts/checkpoints/trm_student_a6000_run1 \
  /Users/robertbadinter/Documents/
```

Important:

- do not run this from the pod
- run it from the local machine

## TensorRT / CUDA Debugging History

This was already solved, but is important context if TensorRT fails again.

Initial issue on pod:

- TensorRT package was built for `cuda13.2`
- host driver/runtime only supported `CUDA 12.4`
- symptom:
  - `CUDA driver version is insufficient for CUDA runtime version`

The fix was:

1. remove incompatible TensorRT packages
2. install CUDA 12.4 compatible TensorRT packages

Installed good package set:

- `libnvinfer-bin=10.1.0.27-1+cuda12.4`
- `libnvinfer-dispatch10=10.1.0.27-1+cuda12.4`
- `libnvinfer-lean10=10.1.0.27-1+cuda12.4`
- `libnvinfer-plugin10=10.1.0.27-1+cuda12.4`
- `libnvinfer-vc-plugin10=10.1.0.27-1+cuda12.4`
- `libnvinfer10=10.1.0.27-1+cuda12.4`
- `libnvonnxparsers10=10.1.0.27-1+cuda12.4`

Note:

- `trtexec` lives at `/usr/src/tensorrt/bin/trtexec` on the pod

## Simulation / Deployment Validation Already Completed

These phases were already validated successfully:

### 1. Gazebo mission world validation

The simulator/world side was validated earlier in the project.

### 2. GPU smoke + ONNX export

The following path works:

- GPU smoke on RTX A6000
- ONNX export to `artifacts/onnx/trm_reasoner.onnx`

### 3. Baseline TensorRT build and inference

The baseline model exported and built successfully.

Representative earlier baseline perf on RTX A6000:

- mean host latency around `0.198 ms`
- mean GPU compute around `0.183 ms`
- throughput around `5384 qps`

### 4. Trained student TensorRT validation

A trained student ONNX was built and validated in TensorRT.

Representative trained student perf:

- mean host latency around `0.140 ms`
- mean GPU compute around `0.126 ms`
- throughput around `7740 qps`

So deployment is not the blocker.

## Core Student Model Changes Already Made

These are important code-level changes already integrated into the repo.

### Halting / ONNX stabilization

File:

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/model.py`

Changes:

- removed Python-side early `break` in the recursive loop
- removed ONNX tracer warning caused by `if bool(torch.all(has_halted))`
- model now always emits a fixed-length halting trace
- `TRMReasonerOutput` now includes:
  - `action_logits`
  - `halt_logits`
  - `halt_probabilities`
  - `hidden_states`
  - `halted_at_step`

Effect:

- no more problematic trace collapse
- ONNX output `halt_probabilities` now stays full length, e.g. `[batch, 6]`

### Benchmark action alignment

File:

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/arc_drone_bench.py`

The synthetic benchmark target actions were aligned to the exact deployed action vocabulary.

This was done because action labels were previously only approximately aligned, which capped action learning.

### Student training objective improvement

File:

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/student_training.py`

Changes:

- halting supervision switched from monotonic BCE-like targets to direct halt-step classification
- auxiliary continuous action-control regression added
- action loss weight increased
- training/export path validated after these changes

## Supervised Student Runs Already Done

### Supervised baseline runs

Relevant commands already executed on the pod:

```bash
python3 scripts/train_trm_student.py \
  --device cuda \
  --task-count 16384 \
  --eval-task-count 2048 \
  --batch-size 64 \
  --epochs 30 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --hidden-size 96 \
  --refinement-steps 6 \
  --halting-threshold 0.82 \
  --output-dir artifacts/checkpoints/trm_student_a6000_run1 \
  --export-onnx
```

Result:

- `best_eval_action_accuracy=0.5000`
- `best_eval_halt_step_mae=0.2939`

After student training improvements:

```bash
python3 scripts/train_trm_student.py \
  --device cuda \
  --task-count 16384 \
  --eval-task-count 2048 \
  --batch-size 64 \
  --epochs 30 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --hidden-size 96 \
  --refinement-steps 6 \
  --halting-threshold 0.82 \
  --output-dir artifacts/checkpoints/trm_student_a6000_run2 \
  --export-onnx
```

Result:

- `best_eval_action_accuracy=0.5146`
- `best_eval_halt_step_mae=0.2837`

Another run:

```bash
python3 scripts/train_trm_student.py \
  --device cuda \
  --task-count 16384 \
  --eval-task-count 2048 \
  --batch-size 64 \
  --epochs 30 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --hidden-size 96 \
  --refinement-steps 6 \
  --halting-threshold 0.82 \
  --action-loss-weight 2.0 \
  --action-regression-weight 0.5 \
  --halt-loss-weight 0.5 \
  --output-dir artifacts/checkpoints/trm_student_a6000_run3 \
  --export-onnx
```

Result:

- `best_eval_action_accuracy=0.5142`
- `best_eval_halt_step_mae=0.2837`

Interpretation:

- supervised-only student plateaued around `0.51`
- infrastructure and export path were working
- model quality bottleneck remained

## Gemma Hidden-Layer Sweep Work

### Why it was added

Before doing expensive full distillation, a lightweight hidden-layer sweep was implemented to identify where the teacher signal is strongest.

New files:

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/gemma_layer_sweep.py`
- `/Users/robertbadinter/Documents/gemma-4/scripts/sweep_gemma_layers.py`

Important implementation detail:

- `google/gemma-4-e2b` must be treated as a multimodal HF model
- the sweep ended up loading Gemma 4 directly and extracting hidden states from the model outputs in text-only mode
- dtype mismatch on probe features was fixed by casting features to `float32`

### Coarse sweep run

Executed:

```bash
python3 scripts/sweep_gemma_layers.py \
  --device cuda \
  --foundation-model-id google/gemma-4-e2b \
  --task-count 512 \
  --eval-task-count 128 \
  --probe-epochs 3 \
  --probe-batch-size 16 \
  --probe-learning-rate 1e-3 \
  --layer-fractions 0.25 0.5 0.75 0.9 \
  --output-dir artifacts/layer_sweeps/gemma_e2b_sweep1
```

Output:

- `hidden_layer_count=35`
- selected layers: `8,17,26,31`
- best layer: `17`
- `best_eval_action_accuracy=0.7656`

Per-layer:

- `layer 8`: `eval_action_accuracy=0.5938`, `eval_halt_step_mae=0.3047`
- `layer 17`: `eval_action_accuracy=0.7656`, `eval_halt_step_mae=0.2422`
- `layer 26`: `eval_action_accuracy=0.5547`, `eval_halt_step_mae=0.2891`
- `layer 31`: `eval_action_accuracy=0.5938`, `eval_halt_step_mae=0.2891`

### Fine sweep run

Executed:

```bash
python3 scripts/sweep_gemma_layers.py \
  --device cuda \
  --foundation-model-id google/gemma-4-e2b \
  --task-count 1024 \
  --eval-task-count 256 \
  --probe-epochs 4 \
  --probe-batch-size 16 \
  --probe-learning-rate 1e-3 \
  --layers 14 15 16 17 18 19 20 \
  --output-dir artifacts/layer_sweeps/gemma_e2b_sweep2
```

Output:

- `hidden_layer_count=35`
- selected layers: `14,15,16,17,18,19,20`
- best layer: `17`
- `best_eval_action_accuracy=0.8203`

Per-layer:

- `14`: `eval_action_accuracy=0.6719`, `eval_halt_step_mae=0.1992`
- `15`: `eval_action_accuracy=0.8008`, `eval_halt_step_mae=0.1875`
- `16`: `eval_action_accuracy=0.8086`, `eval_halt_step_mae=0.1641`
- `17`: `eval_action_accuracy=0.8203`, `eval_halt_step_mae=0.1641`
- `18`: `eval_action_accuracy=0.8047`, `eval_halt_step_mae=0.2422`
- `19`: `eval_action_accuracy=0.6953`, `eval_halt_step_mae=0.2344`
- `20`: `eval_action_accuracy=0.6875`, `eval_halt_step_mae=0.2344`

Main conclusion:

- best teacher layer found so far is `17`
- backup ablation candidate is `16`
- strongest zone is roughly `15-18`

## Why Vision/Audio Were Not Included In The Sweep

This was discussed explicitly.

Current teacher prompt for the sweep is text/symbolic only:

- ARC-like grid serialized as text
- task family metadata
- text prompt

Therefore, the current sweep is intentionally language-path only.

Reason:

- including visual/audio towers without real multimodal inputs would add noise rather than meaningful multimodal evaluation

Future direction:

- once real image/audio teacher inputs are wired in, it may make sense to compare:
  - text-only teacher
  - vision+text teacher
  - maybe audio+text if relevant

## Distillation Work Already Attempted

### Heavy cache-based pipeline added

The repo now contains a heavier two-stage distillation path:

1. build a reusable teacher cache
2. train the student from that cache

Relevant files:

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/student_distillation.py`
- `/Users/robertbadinter/Documents/gemma-4/scripts/cache_teacher_targets.py`
- `/Users/robertbadinter/Documents/gemma-4/scripts/distill_trm_student.py`

Capabilities added:

- cache train/eval teacher targets to disk
- support multiple teacher layers
- support teacher feature pooling:
  - `mean`
  - `concat`
- auto-build cache when none is provided to `distill_trm_student.py`
- store:
  - `grid`
  - `action_index`
  - `action_target_vector`
  - `halt_step`
  - `teacher_features`
  - `teacher_action_logits`
  - `teacher_halt_logits`

### Distillation CLI introduced

Files:

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/student_distillation.py`
- `/Users/robertbadinter/Documents/gemma-4/scripts/distill_trm_student.py`

Goal:

- distill student from Gemma `layer 17`
- keep final student exportable to ONNX/TensorRT

### Distilled run 1

Executed on pod:

```bash
python3 scripts/distill_trm_student.py \
  --device cuda \
  --foundation-model-id google/gemma-4-e2b \
  --teacher-layer-index 17 \
  --task-count 4096 \
  --eval-task-count 512 \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --hidden-size 96 \
  --refinement-steps 6 \
  --halting-threshold 0.82 \
  --action-loss-weight 2.0 \
  --action-regression-weight 0.5 \
  --halt-loss-weight 0.5 \
  --teacher-representation-weight 1.0 \
  --output-dir artifacts/checkpoints/trm_student_distilled_run1 \
  --export-onnx
```

Result:

- `best_eval_action_accuracy=0.5410`
- `best_eval_halt_step_mae=0.2812`

Interpretation:

- a small gain over the supervised plateau
- but nowhere near the teacher probe quality (`0.8203`)

### Distilled run 3

After softening the distillation weights:

```bash
python3 scripts/distill_trm_student.py \
  --device cuda \
  --foundation-model-id google/gemma-4-e2b \
  --teacher-layer-index 17 \
  --task-count 4096 \
  --eval-task-count 512 \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --hidden-size 96 \
  --refinement-steps 6 \
  --halting-threshold 0.82 \
  --action-loss-weight 2.0 \
  --action-regression-weight 0.5 \
  --halt-loss-weight 0.5 \
  --teacher-representation-weight 0.0 \
  --teacher-kl-weight 0.25 \
  --teacher-probe-epochs 5 \
  --teacher-probe-learning-rate 1e-3 \
  --teacher-temperature 3.0 \
  --output-dir artifacts/checkpoints/trm_student_distilled_run3 \
  --export-onnx
```

Result at `10` epochs:

- `best_eval_action_accuracy=0.5059`
- `best_eval_halt_step_mae=0.2812`

Same run family retried at `30` epochs:

- `best_eval_action_accuracy=0.5254`
- `best_eval_halt_step_mae=0.2812`

Interpretation:

- softening the distillation avoided the collapse seen in `run2`
- but gains remained very small
- halting still did not improve
- this strongly suggests the current live distillation design is still not the right transfer mechanism

### Distilled run 2

After enhancing distillation with teacher-probe KL:

```bash
python3 scripts/distill_trm_student.py \
  --device cuda \
  --foundation-model-id google/gemma-4-e2b \
  --teacher-layer-index 17 \
  --task-count 4096 \
  --eval-task-count 512 \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --hidden-size 96 \
  --refinement-steps 6 \
  --halting-threshold 0.82 \
  --action-loss-weight 2.0 \
  --action-regression-weight 0.5 \
  --halt-loss-weight 0.5 \
  --teacher-representation-weight 1.0 \
  --teacher-kl-weight 1.0 \
  --teacher-probe-epochs 5 \
  --teacher-probe-learning-rate 1e-3 \
  --teacher-temperature 2.0 \
  --output-dir artifacts/checkpoints/trm_student_distilled_run2 \
  --export-onnx
```

Result:

- `best_eval_action_accuracy=0.4961`
- `best_eval_halt_step_mae=0.2812`

Interpretation:

- this change made things worse
- current distillation signal is still misaligned
- current branch of distillation is not yet trustworthy

## Current Best Understanding

These statements are the most important ones to preserve:

1. The student architecture/export path is valid.
2. Supervised student training alone plateaus around `0.51`.
3. Gemma `layer 17` contains a much better policy signal, around `0.82` in the probe setting.
4. Current distillation implementation is not yet effectively transferring that signal.
5. Therefore, the main remaining problem is distillation design, not infra.
6. The heavier cache-based pipeline is now the preferred way to continue experiments.

## Current Files That Matter Most

### Student / export

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/model.py`
- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/student_training.py`
- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/export_tensorrt.py`
- `/Users/robertbadinter/Documents/gemma-4/scripts/train_trm_student.py`

### Gemma teacher probing

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/gemma_layer_sweep.py`
- `/Users/robertbadinter/Documents/gemma-4/scripts/sweep_gemma_layers.py`

### Distillation

- `/Users/robertbadinter/Documents/gemma-4/src/arc_drone/student_distillation.py`
- `/Users/robertbadinter/Documents/gemma-4/scripts/distill_trm_student.py`
- `/Users/robertbadinter/Documents/gemma-4/scripts/cache_teacher_targets.py`

### Tests

- `/Users/robertbadinter/Documents/gemma-4/tests/test_model.py`
- `/Users/robertbadinter/Documents/gemma-4/tests/test_student_training.py`
- `/Users/robertbadinter/Documents/gemma-4/tests/test_gemma_layer_sweep.py`
- `/Users/robertbadinter/Documents/gemma-4/tests/test_student_distillation.py`

### Notes / docs

- `/Users/robertbadinter/Documents/gemma-4/README.md`
- `/Users/robertbadinter/Documents/gemma-4/docs/distillation_reasoning_trace.md`

Note:

- `docs/distillation_reasoning_trace.md` was created locally but not deliberately committed in the last distillation commit sequence
- check git status before relying on it being pushed

## Commits Of Interest

These commit messages matter when reconstructing evolution:

- `5a8ad80` `Add TRM student training CLI`
- `bd46671` `Fix student training halt loss under autocast`
- `f4c4121` `Stabilize halting trace and align student supervision`
- `03c895f` `Add continuous action supervision for student training`
- `0b1fdc0` `Add Gemma hidden-layer sweep CLI`
- `b34984e` `Load Gemma 4 language backbone for layer sweep`
- `ae33c86` `Call Gemma 4 directly for hidden-state sweep`
- `dcfc9d3` `Cast Gemma sweep features to float32`
- `49ea0f0` `Add Gemma-guided student distillation CLI`
- `45359f9` `Distill student against teacher probe logits`

## Tests / Validation Status Locally

Local test suite status reached:

- `47 passed`

This means the repo is not obviously broken at the unit level.

However:

- passing tests do not mean the current distillation design is effective

## Best Immediate Next Steps

If another model/chat resumes from here, the most useful next work is probably one of these:

### Option A: Use the cache-based heavy pipeline

Preferred next workflow:

1. build a large teacher cache
2. reuse that cache across many student runs
3. compare teacher layer combinations and student sizes without rerunning Gemma every time

Recommended cache command:

```bash
python3 scripts/cache_teacher_targets.py \
  --device cuda \
  --foundation-model-id google/gemma-4-e2b \
  --teacher-layer-indices 16 17 18 \
  --teacher-feature-pooling mean \
  --task-count 32768 \
  --eval-task-count 4096 \
  --teacher-probe-epochs 5 \
  --teacher-probe-learning-rate 1e-3 \
  --teacher-probe-batch-size 64 \
  --output-dir artifacts/distillation_cache/gemma_e2b_l16_17_18_mean
```

Recommended training-from-cache command:

```bash
python3 scripts/distill_trm_student.py \
  --device cuda \
  --foundation-model-id google/gemma-4-e2b \
  --cache-dir artifacts/distillation_cache/gemma_e2b_l16_17_18_mean \
  --batch-size 32 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --hidden-size 128 \
  --refinement-steps 6 \
  --halting-threshold 0.82 \
  --action-loss-weight 2.0 \
  --action-regression-weight 0.5 \
  --halt-loss-weight 0.5 \
  --teacher-representation-weight 0.0 \
  --teacher-kl-weight 0.25 \
  --teacher-temperature 3.0 \
  --output-dir artifacts/checkpoints/trm_student_distilled_cache_run1 \
  --export-onnx
```

### Option B: Distillation design changes still worth considering

If the heavy cache pipeline still stalls, likely next ideas are:

- action-only teacher KL, while leaving halting to direct supervision
- layer `16` only as an ablation
- `17` only vs `16+17+18 mean`
- larger student (`hidden_size=160`)
- better grid encoder before the recursive core
- curriculum:
  - hard labels first
  - teacher soft labels second
  - final deployment fine-tune last

### Option C: Build multimodal teacher later

Not the current immediate priority, but future work may involve:

- real image-based teacher inputs
- compare text-only vs vision+text teacher

## Important Things Not To Waste Time Re-Debugging

These are already solved:

- TensorRT CUDA mismatch on pod
- ONNX tracer warning from Python boolean halting
- collapsed `halt_probabilities` ONNX shape
- remote SCP/SSH basics for this pod
- hidden-layer sweep loading of Gemma 4
- bfloat16 vs float32 mismatch in sweep probe
- repeated live Gemma extraction for every student run (now avoidable with cache)

## One-Sentence Current State

The repo now has a stable student/export stack, a verified Gemma `layer 17` teacher signal, and a reusable teacher-cache pipeline, but the current distilled student still does not transfer enough of that teacher quality into the compact exportable model.
