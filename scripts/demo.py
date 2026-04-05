#!/usr/bin/env python3
"""
Drone navigation demo — Gradio interactive web app.

Three tabs:
  1. Live inference   — upload T frames → predicted action chunk + CoT + arrow overlay
  2. Sequence replay  — browse AU-AIR sequences with GT vs model predictions side-by-side
  3. Metrics dashboard — load benchmark_auair.json, compare models with charts

Launch:
    python scripts/demo.py
    python scripts/demo.py --trm-checkpoint artifacts/checkpoints/trm_student_distilled/best_student.pt
    python scripts/demo.py --lora-path artifacts/teacher_lora/gemma_e4b_auair --sequences data/auair_sequences.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Action rendering helpers
# ---------------------------------------------------------------------------

ACTION_LABELS = ["↑ North", "↓ South", "→ East", "← West", "⬆ Up", "⬇ Down", "↻ Yaw+", "↺ Yaw−"]
ACTION_COLORS = ["#00d4aa", "#ff6b6b", "#ffd93d", "#a29bfe", "#74b9ff", "#fd79a8", "#e17055", "#6c5ce7"]
ACTION_ARROWS = {
    0: (0, -1),   # north → up on image
    1: (0,  1),   # south → down
    2: (1,  0),   # east  → right
    3: (-1, 0),   # west  → left
    4: (0, -1),   # up    → up (same as north visually)
    5: (0,  1),   # down  → down
    6: (1,  0),   # yaw+  → clockwise
    7: (-1, 0),   # yaw−  → counter-clockwise
}


def draw_action_overlay(image: Image.Image, action_idx: int, confidence: float = 1.0) -> Image.Image:
    """Draw action arrow + label overlay on a drone frame."""
    img = image.copy().resize((480, 360))
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    cx, cy = w // 2, h // 2
    color = ACTION_COLORS[action_idx % len(ACTION_COLORS)]

    # Semi-transparent overlay at bottom
    draw.rectangle([(0, h - 60), (w, h)], fill=(0, 0, 0, 160))

    # Arrow
    dx, dy = ACTION_ARROWS[action_idx]
    arrow_len = 60
    ex, ey = cx + dx * arrow_len, cy + dy * arrow_len
    draw.line([(cx, cy), (ex, ey)], fill=color, width=5)
    # Arrowhead
    for sign in (-1, 1):
        perp = (-dy * sign * 0.4, dx * sign * 0.4)
        ax = int(ex - dx * 18 + perp[0] * 18)
        ay = int(ey - dy * 18 + perp[1] * 18)
        draw.line([(ex, ey), (ax, ay)], fill=color, width=5)

    # Label
    label = ACTION_LABELS[action_idx]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, h - 48), f"{label}  conf={confidence:.0%}", fill=color, font=font)

    return img


def make_action_bar_chart(action_logits: list[float] | None) -> Image.Image:
    """Render action probability bar chart as PIL image."""
    import io
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return Image.new("RGB", (400, 200), color=(30, 30, 30))

    if action_logits is None:
        action_logits = [0.0] * 8

    import torch
    probs = torch.softmax(torch.tensor(action_logits, dtype=torch.float32), dim=0).numpy()
    labels = [l.split(" ")[1] for l in ACTION_LABELS]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    bars = ax.barh(labels, probs, color=[c for c in ACTION_COLORS])
    ax.set_xlim(0, 1)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.set_xlabel("Probability", color="white")
    ax.set_title("Action Distribution", color="white", pad=8)
    for bar, p in zip(bars, probs):
        ax.text(p + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", color="white", fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _load_trm(checkpoint_path: str, T: int = 4, C: int = 4):
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]
    import torch
    from arc_drone.model import TRMReasoner
    from arc_drone.config import ReasonerConfig
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    rc_dict = ckpt.get("reasoner_config", {})
    rc = ReasonerConfig(
        hidden_size=rc_dict.get("hidden_size", 96),
        refinement_steps=rc_dict.get("refinement_steps", 6),
        halting_threshold=rc_dict.get("halting_threshold", 0.82),
        action_chunk_size=C,
        temporal_window=T,
    )
    model = TRMReasoner(rc)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    _model_cache[checkpoint_path] = model
    return model


def _load_gemma(model_id: str, lora_path: str | None = None):
    key = f"{model_id}|{lora_path}"
    if key in _model_cache:
        return _model_cache[key]
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    import torch.nn as _nn
    if not hasattr(_nn.Module, "set_submodule"):
        def _ssm(self, target, module):
            atoms = target.split(".")
            mod = self
            for a in atoms[:-1]:
                mod = mod.get_submodule(a)
            setattr(mod, atoms[-1], module)
        _nn.Module.set_submodule = _ssm

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0},
        dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True,
    )
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    _model_cache[key] = (processor, model)
    return processor, model


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def infer_trm(frames: list[Image.Image], checkpoint_path: str, C: int = 4) -> dict:
    import torch
    T = len(frames)
    model = _load_trm(checkpoint_path, T=T, C=C)

    # Convert frames to ARC grids (same as AuAirStudentDataset)
    import numpy as np
    grids = []
    for img in frames:
        arr = np.array(img.resize((30, 30)).convert("L"), dtype=np.float32)
        arr = (arr / 255.0 * 9).astype(np.int64)
        grids.append(arr)
    grids_t = torch.tensor(np.stack(grids), dtype=torch.long).unsqueeze(0)  # (1,T,H,W)

    with torch.no_grad():
        out = model(grids_t)

    action_logits = out.action_chunk_logits[0]  # (C, 8)
    pred_actions = action_logits.argmax(dim=-1).tolist()
    halt_pred = out.halted_at_step[0].item()
    confidences = torch.softmax(action_logits, dim=-1).max(dim=-1).values.tolist()

    return {
        "actions": pred_actions,
        "halts": [round(halt_pred)] * C,
        "confidences": confidences,
        "action_logits": action_logits[0].tolist(),
        "cot": f"TRM: {out.refinement_steps_taken[0].item()} refinement steps",
    }


def infer_gemma(frames: list[Image.Image], model_id: str, lora_path: str | None,
                T: int = 4, C: int = 4) -> dict:
    import torch
    import re
    processor, model = _load_gemma(model_id, lora_path)

    prompt_text = (
        f"You are an autonomous drone navigation assistant. "
        f"You are given {T} consecutive aerial frames.\n\n"
        "Analyze object positions and motion across frames, then predict the next "
        f"{C} drone actions.\n\nOutput EXACTLY:\n"
        + "\n".join(f"Action_{i}: <0-7>  Halt_{i}: <1-6>" for i in range(C))
        + "\n\nAction index: 0=north 1=south 2=east 3=west 4=up 5=down 6=yaw_right 7=yaw_left"
    )
    content = [{"type": "image"} for _ in range(T)]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=prompt, images=frames[-T:], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 pad_token_id=processor.tokenizer.eos_token_id)
    il = inputs["input_ids"].shape[1]
    cot = processor.tokenizer.decode(out_ids[0][il:], skip_special_tokens=True).strip()

    actions, halts, confidences = [], [], []
    for i in range(C):
        am = re.search(rf"Action_{i}:\s*(\d+)", cot)
        hm = re.search(rf"Halt_{i}:\s*(\d+)", cot)
        actions.append(int(am.group(1)) if am and 0 <= int(am.group(1)) <= 7 else 0)
        halts.append(int(hm.group(1)) if hm and 1 <= int(hm.group(1)) <= 6 else 3)
        confidences.append(1.0 - (halts[-1] - 1) / 5.0)

    return {"actions": actions, "halts": halts, "confidences": confidences,
            "action_logits": None, "cot": cot}


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(
    model_id: str,
    lora_path: str | None,
    trm_checkpoint: str | None,
    sequences_path: str | None,
    benchmark_path: str | None,
    temporal_window: int,
    action_chunk_size: int,
) -> "gr.Blocks":
    import gradio as gr

    T, C = temporal_window, action_chunk_size

    # ── Tab 1: Live inference ────────────────────────────────────────────────
    def run_inference(frame_0, frame_1, frame_2, frame_3, model_choice):
        raw_frames = [frame_0, frame_1, frame_2, frame_3]
        frames = []
        for f in raw_frames[:T]:
            if f is None:
                frames.append(Image.new("RGB", (640, 480), color=(40, 40, 40)))
            else:
                frames.append(Image.fromarray(f).convert("RGB"))

        try:
            if model_choice == "TRM Student" and trm_checkpoint:
                result = infer_trm(frames, trm_checkpoint, C=C)
            else:
                result = infer_gemma(frames, model_id, lora_path if "LoRA" in model_choice else None,
                                     T=T, C=C)
        except Exception as e:
            return [None] * T, f"❌ Error: {e}", None, "—"

        # Overlay arrows on each frame
        overlaid = []
        for i, frame in enumerate(frames):
            a = result["actions"][min(i, len(result["actions"]) - 1)]
            conf = result["confidences"][min(i, len(result["confidences"]) - 1)]
            overlaid.append(draw_action_overlay(frame, a, conf))

        # Action chunk summary
        chunk_summary = " → ".join(
            f"**{ACTION_LABELS[a]}** (h={h})"
            for a, h in zip(result["actions"], result["halts"])
        )

        bar_chart = make_action_bar_chart(result.get("action_logits"))
        cot_text = result.get("cot", "")

        return overlaid, chunk_summary, bar_chart, cot_text

    # ── Tab 2: Sequence replay ───────────────────────────────────────────────
    sequences: list[dict] = []
    if sequences_path and Path(sequences_path).exists():
        with open(sequences_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sequences.append(json.loads(line))

    def load_sequence(seq_idx):
        if not sequences:
            return [None] * T, "No sequences loaded.", "—", "—"
        seq = sequences[int(seq_idx) % len(sequences)]
        image_paths = seq.get("image_paths", [])
        frames = []
        for p in image_paths[-T:]:
            try:
                frames.append(Image.open(p).convert("RGB"))
            except Exception:
                frames.append(Image.new("RGB", (640, 480), color=(40, 40, 40)))
        while len(frames) < T:
            frames.insert(0, Image.new("RGB", (640, 480), color=(40, 40, 40)))

        gt_actions = (list(seq.get("action_indices", [0] * C)) * C)[:C]
        gt_halts = (list(seq.get("halt_steps", [3] * C)) * C)[:C]
        gt_text = " → ".join(f"{ACTION_LABELS[a]} (h={h})" for a, h in zip(gt_actions, gt_halts))

        telemetry = seq.get("telemetry", {})
        tel_text = (
            f"lx={telemetry.get('linear_x', 0):.2f}  "
            f"ly={telemetry.get('linear_y', 0):.2f}  "
            f"lz={telemetry.get('linear_z', 0):.2f}  "
            f"alt={telemetry.get('altitude_m', 0):.1f}m  "
            f"clip={seq.get('clip_id', '?')}  "
            f"frame={seq.get('frame_index', 0)}"
        )
        sample_info = f"**{seq.get('sample_id', '')}** — {tel_text}"
        return frames, gt_text, sample_info, seq.get("reasoning_trace", "—")

    # ── Tab 3: Metrics dashboard ─────────────────────────────────────────────
    def load_metrics(report_path):
        path = Path(report_path)
        if not path.exists():
            return "No report found at that path.", None, None
        data = json.loads(path.read_text())
        results = data.get("results", [])
        if not results:
            return "Empty report.", None, None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io

            names = [r["model_name"] for r in results]
            act_accs = [r["action_acc"] for r in results]
            parse_rates = [r.get("parse_rate", 100) for r in results]
            latencies = [r.get("ms_per_sample", 0) for r in results]
            C_steps = len(results[0].get("chunk_acc", [1]))
            chunk_data = {r["model_name"]: r.get("chunk_acc", [0] * C_steps) for r in results}

            # Chart 1: action acc + parse rate
            fig1, ax = plt.subplots(figsize=(7, 3.5))
            fig1.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            x = np.arange(len(names))
            w = 0.35
            b1 = ax.bar(x - w/2, act_accs, w, label="Action acc %", color="#00d4aa")
            b2 = ax.bar(x + w/2, parse_rates, w, label="Parse rate %", color="#ffd93d")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, color="white", fontsize=9)
            ax.tick_params(colors="white"); ax.legend(facecolor="#333", labelcolor="white")
            ax.set_title("Model comparison — Action accuracy & Parse rate", color="white")
            ax.spines[:].set_color("#333")
            for b in list(b1) + list(b2):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                        f"{b.get_height():.1f}", ha="center", color="white", fontsize=8)
            plt.tight_layout()
            buf1 = io.BytesIO(); plt.savefig(buf1, format="png", facecolor=fig1.get_facecolor())
            plt.close(); buf1.seek(0); img1 = Image.open(buf1).copy()

            # Chart 2: chunk accuracy per step
            fig2, ax2 = plt.subplots(figsize=(7, 3.5))
            fig2.patch.set_facecolor("#1a1a2e"); ax2.set_facecolor("#16213e")
            for i, (model_name, accs) in enumerate(chunk_data.items()):
                ax2.plot(range(C_steps), accs, marker="o", label=model_name,
                         color=ACTION_COLORS[i % len(ACTION_COLORS)], linewidth=2)
            ax2.set_xlabel("Action step", color="white"); ax2.set_ylabel("Accuracy %", color="white")
            ax2.set_xticks(range(C_steps)); ax2.set_xticklabels([f"a{i}" for i in range(C_steps)])
            ax2.tick_params(colors="white"); ax2.legend(facecolor="#333", labelcolor="white")
            ax2.set_title("Chunk accuracy per step", color="white"); ax2.spines[:].set_color("#333")
            plt.tight_layout()
            buf2 = io.BytesIO(); plt.savefig(buf2, format="png", facecolor=fig2.get_facecolor())
            plt.close(); buf2.seek(0); img2 = Image.open(buf2).copy()

        except Exception as e:
            img1 = img2 = None

        summary_lines = [f"**{r['model_name']}**: act={r['action_acc']:.1f}%  "
                         f"parse={r.get('parse_rate', 100):.0f}%  "
                         f"lat={r.get('ms_per_sample', 0):.0f}ms"
                         for r in results]
        summary = "\n".join(summary_lines)
        return summary, img1, img2

    # ── Build Gradio UI ──────────────────────────────────────────────────────
    available_models = ["Gemma 4 Vanilla"]
    if lora_path:
        available_models.append("Gemma 4 + LoRA (fine-tuned)")
    if trm_checkpoint:
        available_models.append("TRM Student")

    with gr.Blocks(
        theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"),
        title="Drone Navigation Demo",
        css=".gradio-container { max-width: 1100px !important }"
    ) as app:
        gr.Markdown("# 🚁 Drone Navigation — Live Demo\nNVARC-inspired pipeline · Gemma 4 + TRM student")

        with gr.Tabs():

            # ── Tab 1: Live inference ────────────────────────────────────────
            with gr.Tab("🎯 Live Inference"):
                gr.Markdown("Upload **4 consecutive drone frames** (oldest → newest). The model predicts the next action chunk.")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            f0 = gr.Image(label="Frame t-3", type="numpy", height=180)
                            f1 = gr.Image(label="Frame t-2", type="numpy", height=180)
                            f2 = gr.Image(label="Frame t-1", type="numpy", height=180)
                            f3 = gr.Image(label="Frame t",   type="numpy", height=180)
                        model_choice = gr.Dropdown(available_models, value=available_models[-1],
                                                   label="Model")
                        run_btn = gr.Button("▶ Predict", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            o0 = gr.Image(label="t-3 + action", height=180)
                            o1 = gr.Image(label="t-2 + action", height=180)
                            o2 = gr.Image(label="t-1 + action", height=180)
                            o3 = gr.Image(label="t + action",   height=180)
                        chunk_out = gr.Markdown("**Predicted chunk:** —")
                        bar_out = gr.Image(label="Action distribution", height=180)
                        cot_out = gr.Textbox(label="CoT / reasoning trace", lines=6, max_lines=12)

                run_btn.click(
                    run_inference,
                    inputs=[f0, f1, f2, f3, model_choice],
                    outputs=[gr.Gallery(label="Overlay", columns=4), chunk_out, bar_out, cot_out],
                )
                # Simplified version with 4 separate outputs
                run_btn.click(
                    run_inference,
                    inputs=[f0, f1, f2, f3, model_choice],
                    outputs=[[o0, o1, o2, o3], chunk_out, bar_out, cot_out],
                )

            # ── Tab 2: Sequence replay ───────────────────────────────────────
            with gr.Tab("🎬 Sequence Replay"):
                gr.Markdown(f"Browse AU-AIR sequences ({len(sequences)} loaded). Shows GT telemetry labels.")
                with gr.Row():
                    seq_slider = gr.Slider(0, max(len(sequences) - 1, 1), step=1, label="Sequence index", value=0)
                    load_btn = gr.Button("Load")
                with gr.Row():
                    rframes = [gr.Image(label=f"Frame {i}", height=200) for i in range(T)]
                gt_out = gr.Markdown("**GT actions:** —")
                info_out = gr.Markdown("—")
                cot_replay = gr.Textbox(label="Reasoning trace (if annotated)", lines=5)

                load_btn.click(load_sequence, inputs=[seq_slider],
                               outputs=rframes + [gt_out, info_out, cot_replay])
                seq_slider.change(load_sequence, inputs=[seq_slider],
                                  outputs=rframes + [gt_out, info_out, cot_replay])

            # ── Tab 3: Metrics dashboard ─────────────────────────────────────
            with gr.Tab("📊 Metrics Dashboard"):
                gr.Markdown("Load a `benchmark_auair.json` report from `scripts/benchmark_auair.py`.")
                with gr.Row():
                    report_input = gr.Textbox(
                        value=benchmark_path or "artifacts/benchmark_auair.json",
                        label="Path to benchmark report JSON"
                    )
                    load_metrics_btn = gr.Button("Load metrics")
                metrics_summary = gr.Markdown("—")
                with gr.Row():
                    chart1 = gr.Image(label="Accuracy comparison", height=300)
                    chart2 = gr.Image(label="Chunk accuracy per step", height=300)

                load_metrics_btn.click(load_metrics, inputs=[report_input],
                                       outputs=[metrics_summary, chart1, chart2])

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Drone navigation Gradio demo.")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--trm-checkpoint", default=None)
    parser.add_argument("--sequences", default=None,
                        help="JSONL from parse_auair.py for sequence replay tab")
    parser.add_argument("--benchmark", default=None,
                        help="JSON from benchmark_auair.py for metrics tab")
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, default=4)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("Install gradio: pip install gradio matplotlib")
        sys.exit(1)

    app = build_app(
        model_id=args.model_id,
        lora_path=args.lora_path,
        trm_checkpoint=args.trm_checkpoint,
        sequences_path=args.sequences,
        benchmark_path=args.benchmark,
        temporal_window=args.temporal_window,
        action_chunk_size=args.action_chunk_size,
    )
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
