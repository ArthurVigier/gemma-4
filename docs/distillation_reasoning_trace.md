# Distillation Reasoning Trace

## Goal

Find a teacher-guided path that improves the TRM-like student beyond the supervised-only plateau on `ARC-Drone-Bench`, while keeping the final model exportable to `ONNX -> TensorRT`.

## What Was Stabilized First

Before touching teacher distillation, the student/export path was cleaned up:

- removed the ONNX tracer warning caused by Python-side early halting
- fixed the halting trace to remain full-length across refinement steps
- aligned benchmark target actions with the deployed student action vocabulary
- changed halting supervision from monotonic BCE-style targets to direct halt-step classification
- added an auxiliary continuous action-control loss

These changes improved training stability and export correctness, but the student still plateaued near:

- `eval_action_accuracy ~= 0.51`
- `eval_halt_step_mae ~= 0.28`

Conclusion: the bottleneck was no longer infrastructure or export. The bottleneck was teacher signal quality.

## Why A Gemma Layer Sweep Was Introduced

Rather than launching full expensive distillation immediately, a lightweight hidden-layer sweep was added for `google/gemma-4-e2b`.

The idea:

1. freeze Gemma
2. extract hidden states from selected layers
3. train a lightweight probe on top of each layer
4. compare action and halting signal quality
5. choose the best teacher layer for real distillation

This was intentionally cheaper than running a full student training for every candidate layer.

## First Sweep

Coarse sweep over:

- `25%`
- `50%`
- `75%`
- `90%`

Selected layers:

- `8`
- `17`
- `26`
- `31`

Observed results:

- `layer 8`: usable but clearly weaker
- `layer 17`: strongest overall
- `layer 26`: worse
- `layer 31`: also worse

Best result from sweep 1:

- `best_layer_index = 17`
- `best_eval_action_accuracy = 0.7656`

Conclusion: the teacher signal clearly exists, and it peaks in the middle of the Gemma backbone rather than late in the stack.

## Fine Sweep Around The Best Region

A second sweep was run around the promising zone:

- `14`
- `15`
- `16`
- `17`
- `18`
- `19`
- `20`

Observed results:

- `layer 14`: good, but weaker than the best group
- `layer 15`: very strong
- `layer 16`: very strong, especially on halting
- `layer 17`: best overall
- `layer 18`: still strong but slightly worse
- `layer 19`: clear drop
- `layer 20`: clear drop

Best result from sweep 2:

- `best_layer_index = 17`
- `best_eval_action_accuracy = 0.8203`
- `eval_halt_step_mae = 0.1641`

Interpretation:

- the optimal zone is approximately `layers 15-18`
- `layer 17` is the best default teacher target
- `layer 16` is a strong backup/ablation candidate

## Why Vision/Audio Were Not Included Yet

The current teacher prompt is still text/symbolic:

- ARC-like symbolic grid
- family metadata
- text prompt for action/halting reasoning

Because the teacher input is text-only right now, adding Gemma vision/audio towers into this sweep would not provide a fair multimodal comparison. It would mostly add noise.

So the decision was:

- current sweep: language path only
- later extension: multimodal teacher distillation once real image/audio teacher inputs are wired in

## Distillation Decision

The supervised-only student plateaued around `0.51` action accuracy.

The Gemma probe on `layer 17` reached about `0.82` action accuracy.

That gap is large enough to justify a real teacher-guided distillation stage.

So the next architecture step was:

- keep the student architecture exportable
- add a projection head from student hidden state to teacher hidden dimension
- distill from Gemma `layer 17`

Training losses now combine:

- discrete action classification
- continuous action regression
- halting-step classification
- teacher representation matching

## Current Recommended Teacher Layer

Primary choice:

- `layer 17`

Secondary ablation candidate:

- `layer 16`

## Current Status

Implemented and validated in the repo:

- supervised student training CLI
- Gemma hidden-layer sweep CLI
- Gemma-guided student distillation CLI
- ONNX/TensorRT export path for the student

Practical next run:

- run distillation with `teacher_layer_index=17`
- compare distilled student metrics against the supervised-only baseline
- if needed, run one ablation with `teacher_layer_index=16`
