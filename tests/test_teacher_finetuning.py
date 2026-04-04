import numpy as np
import torch

from arc_drone.arc_types import ArcGrid, BenchmarkTask, DroneAction
from arc_drone.teacher_finetuning import TeacherHybridDataset


class _FakeTokenizer:
    eos_token = "<eos>"
    image_token = "<|image|>"


class _FakeBatch(dict):
    def __getattr__(self, name: str):
        return self[name]


class _FakeProcessorNoChatTemplate:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_seq_length = 140
        self.calls: list[dict[str, object]] = []
        self.chat_template = None

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("apply_chat_template should not be called for processors without a chat template")

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
        if text.endswith(self.tokenizer.eos_token):
            input_ids = torch.tensor([[11, 12, 13, 14, 21, 22, 23]], dtype=torch.long)
        else:
            input_ids = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)

        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        return _FakeBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )


def test_teacher_dataset_falls_back_when_chat_template_is_missing() -> None:
    processor = _FakeProcessorNoChatTemplate()
    task = BenchmarkTask(
        task_id="task-1",
        family="forward",
        input_grid=ArcGrid(np.array([[1, 2], [3, 4]], dtype=np.int64)),
        target_grid=ArcGrid(np.array([[1, 2], [3, 4]], dtype=np.int64)),
        target_action=DroneAction((0.3, 0.0, 0.0), yaw_rate=0.0, halt_probability=0.9),
    )

    dataset = TeacherHybridDataset([task], processor, max_length=32)
    payload = dataset[0]

    assert processor.calls[0]["text"].startswith("<|image|>\n")
    assert processor.calls[0]["max_length"] == 32 + processor.image_seq_length
    assert all(call["images"] is not None for call in processor.calls)
    assert payload["labels"].tolist() == [-100, -100, -100, -100, 21, 22, 23]
