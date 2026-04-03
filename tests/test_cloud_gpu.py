from arc_drone.cloud_gpu import GpuSmokeReport, format_gpu_smoke_report


def test_format_gpu_smoke_report_includes_trtexec_command() -> None:
    report = GpuSmokeReport(
        device_name="NVIDIA A100-SXM4-40GB",
        batch_size=1,
        latency_ms=12.345,
        halted_at_step=4,
        onnx_output_path="artifacts/onnx/trm_reasoner.onnx",
        engine_output_path="artifacts/trt/trm_reasoner.plan",
        trtexec_command=(
            "trtexec",
            "--onnx=artifacts/onnx/trm_reasoner.onnx",
            "--saveEngine=artifacts/trt/trm_reasoner.plan",
            "--int8",
            "--skipInference",
        ),
    )

    text = format_gpu_smoke_report(report)

    assert "CUDA smoke test OK" in text
    assert "latency_ms=12.345" in text
    assert "trtexec --onnx=artifacts/onnx/trm_reasoner.onnx" in text
