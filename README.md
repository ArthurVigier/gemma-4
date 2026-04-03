# ARC-Drone-Bench

Workspace for porting ARC Prize style reasoning components to a simulated drone stack.

- primary simulator stack: `Gazebo Harmonic + PX4 SITL + ROS 2 Jazzy`
- secondary backend: `AirSim`
- target pipeline: simulated image -> ARC-like grid -> abstract reasoning -> drone action
- target model family: `Gemma-4-E2B/E4B` distilled into a TinyLM or TRM-like student under `50M` parameters
- target inference path: `ONNX -> TensorRT / TensorRT-LLM`

Current reference status:

- see `docs/stack_2026_validated.md` for the validated 2026 stack and sources
- `ROS 2 Iron` is no longer a valid target for a new setup
- `ROS 2 Humble + Gazebo Harmonic` is still tolerated, but it is not the default
- heavy training and distillation must run on cloud GPUs

## Structure

- `src/arc_drone/pipeline_vision.py`: vision -> ARC-like 10-color grid conversion
- `src/arc_drone/arc_drone_bench.py`: benchmark generation and evaluation
- `src/arc_drone/model.py`: TRM-like core with recursive self-refinement and adaptive halting
- `src/arc_drone/bringup.py`: Gazebo/PX4/ROS2 bringup helpers
- `src/arc_drone/gazebo_px4_adapter.py`: Gazebo Harmonic + PX4 SITL + ROS 2 adapter
- `src/arc_drone/live_benchmark.py`: live benchmark episodes and automatic JSONL export
- `src/arc_drone/live_ros2_app.py`: executable ROS2 loop for the node
- `src/arc_drone/benchmark_export.py`: benchmark episode export built from supervision snapshots/events
- `src/arc_drone/supervision.py`: PX4 supervision JSON payloads and transition summaries
- `launch/arc_drone_gazebo_px4.launch.py`: full bringup for MicroXRCEAgent + PX4 + Gazebo + bridges + node
- `src/arc_drone/ros_node.py`: ROS2 node entrypoint
- `src/arc_drone/export_tensorrt.py`: ONNX export and TensorRT commands
- `tests/`: unit tests

## Quick Start

```bash
python3 -m pip install -e '.[dev]'
```

## Tests

```bash
pytest
```

## ROS2 / Gazebo / PX4 Bringup

Example:

```bash
ros2 launch /path/to/gemma-4/launch/arc_drone_gazebo_px4.launch.py \
  px4_autopilot_path:=/path/to/PX4-Autopilot \
  px4_make_target:=gz_x500_depth
```

Useful live benchmark options:

```bash
python3 -m arc_drone.live_ros2_app \
  --benchmark-output-path artifacts/benchmark/live_benchmark_metrics.jsonl \
  --benchmark-task-count 200 \
  --benchmark-max-episode-steps 100 \
  --benchmark-ready-timeout-steps 40 \
  --benchmark-rotate-max-rows 1000
```

Default bridged topics:

- Gazebo clock: `/world/default/clock` -> `/clock`
- Gazebo image: `/camera` -> `/camera/image_raw`
- Gazebo camera info: `/camera_info` -> `/camera/camera_info`

These topics can be overridden at launch time if the PX4 vehicle model exposes different sensor names.

## Supervision Topics

The node publishes:

- `arc_drone/control_state` as `std_msgs/String`
  JSON with the aggregated PX4 state, NED pose/velocity, and the latest action
- `arc_drone/control_events` as `std_msgs/String`
  JSON event per transition or incident such as `ready`, `armed`, `offboard`, `failsafe`, or `command_error`
- `arc_drone/benchmark_metrics` as `std_msgs/String`
  JSON benchmark episode payload published when an episode finishes

## Benchmark Export

`BenchmarkSupervisionExporter` combines:

- ARC metrics: `grid_accuracy`, `action_accuracy`, `latency_ms`, `energy_joules`
- supervision metrics: `time_to_ready_ms`, offboard/arming transitions, command errors, failsafe

Output:

- JSONL export through `export_to_jsonl(...)`
- automatic JSONL writing in the live loop at the end of each episode
- automatic rotation:
  - one timestamped filename per run
  - row-count rotation via `rotate_max_rows_per_file`

Live episode termination uses real control outcomes instead of a fixed step count:

- `success`
- `failsafe`
- `command_error`
- `offboard_lost`
- `ready_timeout`
- `max_steps_guard`

The success condition is now simulator-aware:

- symbolic success requires exact ARC grid match plus correct action
- physical success requires entering the task target waypoint/zone in PX4 NED coordinates
- final success requires both, plus `ready_for_control` when configured

## Architecture Notes

This scaffold validates:

1. ARC-like symbolic representations from simulated imagery
2. synthetic `ARC-Drone-Bench` task generation
3. a compact recursive core with adaptive halting
4. ROS2 / TensorRT integration points

Training or distilling `Gemma-4-E2B/E4B` with `Unsloth`, `PEFT`, `QLoRA`, and `BitsAndBytes` is not executed locally in this scaffold. That phase should run on cloud GPUs.

For that stage, I recommend renting a GPU with at least 40 GB of VRAM, for example an A100 40/80GB on RunPod or Vast.ai.
