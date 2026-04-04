import numpy as np
import torch

from arc_drone.arc_types import ArcGrid, BenchmarkTask, DroneAction
from arc_drone.teacher_finetuning_moe import TeacherMoEHybridDataset


class _FakeTokenizer:
    eos_token = "<eos>"


class _FakeBatch(dict):
    def __getattr__(self, name: str):
        return self[name]


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.prompt = "<multimodal-prompt>"
        self.image_seq_length = 140
        self.chat_messages = None
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        self.chat_messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return self.prompt

    def __call__(
        self,
        *,
        text: str,
        images=None,
        return_tensors: str,
        padding: str | None = None,
        max_length: int | None = None,
        truncation: bool | None = None,
    ) -> _FakeBatch:
        self.calls.append(
            {
                "text": text,
                "images": images,
                "padding": padding,
                "max_length": max_length,
                "truncation": truncation,
            }
        )
        if text == self.prompt:
            input_ids = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
        elif text.endswith(self.tokenizer.eos_token):
            input_ids = torch.tensor([[11, 12, 13, 14, 21, 22, 23]], dtype=torch.long)
        else:
            raise AssertionError(f"Unexpected processor text: {text!r}")

        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        return _FakeBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )


def test_teacher_moe_dataset_uses_chat_template_for_multimodal_prompt() -> None:
    processor = _FakeProcessor()
    task = BenchmarkTask(
        task_id="task-1",
        family="forward",
        input_grid=ArcGrid(np.array([[1, 2], [3, 4]], dtype=np.int64)),
        target_grid=ArcGrid(np.array([[1, 2], [3, 4]], dtype=np.int64)),
        target_action=DroneAction((0.3, 0.0, 0.0), yaw_rate=0.0, halt_probability=0.9),
        metadata={
            "reasoning_trace": "Reasoning: Counting task. Use the object density to choose the action.",
            "isaac_scene": {
                "entities": [
                    {
                        "position": (0.15, -0.30, 0.0),
                        "color_id": 3,
                        "material": "plastic",
                        "prim_type": "sphere",
                        "semantics": {"id": "obj_0_1"},
                    }
                ],
                "replicator_randomization": {"texture_jitter": True},
            },
        },
    )

    dataset = TeacherMoEHybridDataset([task], processor, max_length=32)
    payload = dataset[0]

    assert processor.chat_messages is not None
    content = processor.chat_messages[0]["content"]
    assert content[0] == {"type": "image"}
    assert "--- PHYSICAL SENSORS ---" in content[1]["text"]
    assert "obj_0_1" in content[1]["text"]
    assert "Replicator randomization" in content[1]["text"]
    assert "--- LOGICAL GRID ---" in content[1]["text"]
    assert "Predict the best drone action family and halting step." in content[1]["text"]
    assert "Reasoning: Counting task." in processor.calls[0]["text"]
    assert all("<image>" not in call["text"] for call in processor.calls)
    assert all(call["images"] is not None for call in processor.calls)
    assert processor.calls[0]["max_length"] == 32 + processor.image_seq_length
    assert payload["labels"].tolist() == [-100, -100, -100, -100, 21, 22, 23]
    assert payload["pixel_values"].shape == (3, 2, 2)
