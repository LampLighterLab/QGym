import json
import sqlite3
import xml.etree.ElementTree as ET

import pytest

from scripts.profile_simulation import (
    capture_command,
    cuda_capture_summary,
    get_args,
    host_flamegraph_svg,
    mark_profiled_result,
    trim_to_timed_stacks,
)


def test_host_profile_excludes_setup_and_preserves_timed_stack_weights():
    raw = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {
            "frames": [
                {"name": "main"},
                {"name": "setup"},
                {"name": "profiled_steps"},
                {"name": "step"},
                {"name": "native_driver_wait"},
            ]
        },
        "profiles": [
            {
                "name": "worker",
                "type": "sampled",
                "unit": "seconds",
                "startValue": 0,
                "endValue": 0.015,
                "samples": [[0, 1], [0, 2, 3], [0, 2, 3, 4]],
                "weights": [0.010, 0.002, 0.003],
            }
        ],
    }

    trimmed, count = trim_to_timed_stacks(raw)

    assert count == 2
    assert trimmed["shared"] == raw["shared"]
    result = trimmed["profiles"][0]
    assert result["samples"] == [[2, 3], [2, 3, 4]]
    assert result["weights"] == [0.002, 0.003]
    assert result["endValue"] == pytest.approx(0.005)
    assert raw["profiles"][0]["endValue"] == 0.015


def test_missing_timed_samples_is_an_explicit_capture_failure():
    with pytest.raises(RuntimeError, match="No profiled_steps stack samples"):
        trim_to_timed_stacks({"shared": {"frames": []}, "profiles": []})


def test_svg_flamegraph_uses_sample_weights_and_escapes_native_names():
    profile = {
        "shared": {
            "frames": [
                {"name": "profiled_steps"},
                {"name": "native<shim>&wait"},
                {"name": "step", "file": "/repo/code.py", "line": 5},
            ]
        },
        "profiles": [
            {
                "name": "main thread",
                "unit": "seconds",
                "samples": [[0, 1], [0, 2]],
                "weights": [0.003, 0.002],
            }
        ],
    }

    svg = ET.fromstring(host_flamegraph_svg(profile))

    ns = {"svg": "http://www.w3.org/2000/svg"}
    frames = {
        group.find("svg:title", ns).text.splitlines()[0]: group.find("svg:rect", ns)
        for group in svg.findall("svg:g", ns)
    }
    native_width = float(frames["native<shim>&wait"].get("width"))
    step_width = float(frames["step (/repo/code.py:5)"].get("width"))
    assert native_width / step_width == pytest.approx(3 / 2)
    assert native_width + step_width == pytest.approx(
        float(frames["profiled_steps"].get("width"))
    )
    text = " ".join(svg.itertext())
    assert "not GPU time" in text
    assert "Native symbols/source depend" in text
    assert "2 samples" in text
    assert not svg.findall(".//svg:script", ns)
    assert not svg.findall(".//svg:image", ns)


@pytest.mark.parametrize("tool", ["nsys", "py-spy"])
def test_capture_commands_use_one_bounded_benchmark_batch(tool, tmp_path):
    args = get_args(
        [
            "--tool",
            tool,
            "--backend",
            "vsim",
            "--sets",
            "4096",
            "--profile",
            "task_empty",
            "--steps",
            "50",
            "--output",
            str(tmp_path),
        ]
    )

    command = capture_command(args, f"/tools/{tool}")

    assert "scripts.benchmark_simulation" in command
    assert command[command.index("--repeats") + 1] == "1"
    assert command[command.index("--warmup") + 1] == "100"
    assert command[command.index("--sets") + 1] == "4096"
    if tool == "nsys":
        assert "--cuda-graph-trace=node" in command
        assert "--capture-range=cudaProfilerApi" in command
        assert "--capture" in command
        assert "--sample=none" in command
    else:
        assert "--native" in command
        assert "--idle" in command
        assert "--capture" not in command


def test_cuda_summary_counts_native_kernels_and_preserves_overlap_caveat(tmp_path):
    path = tmp_path / "capture.sqlite"
    with sqlite3.connect(path) as database:
        database.executescript(
            "CREATE TABLE StringIds (id INTEGER, value TEXT);"
            "INSERT INTO StringIds VALUES (1, 'native_solver'), (2, 'tensor_copy');"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, demangledName INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(0, 10, 1), (5, 15, 1), (20, 25, 2);"
        )

    result = cuda_capture_summary(path)

    assert result["kernel_events"] == 3
    assert result["top_kernels_by_summed_duration"][0] == {
        "name": "native_solver",
        "calls": 2,
        "summed_duration_ns": 20,
    }
    assert "not wall time" in result["note"]


@pytest.mark.parametrize("tool", ["nsys", "py-spy"])
def test_profiled_worker_durations_cannot_enter_performance_gate(tool, tmp_path):
    from scripts.benchmark_simulation import compare_pairs

    reference = {
        "schema": 1,
        "protocol": {"steps": 100},
        "config": {"sim_dt": 0.01},
        "hardware": {"device": "test"},
        "packages": {},
        "native_topology": None,
        "reset_count": 0,
        "capture": False,
        "median_s": 1.0,
        "durations_s": [1.0],
    }
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(reference))

    mark_profiled_result(path, tool)

    captured = json.loads(path.read_text())
    assert captured["capture"] is True
    assert captured["profiler"] == tool
    assert captured["performance_gate"] is False
    assert captured["durations_s"] == reference["durations_s"]
    with pytest.raises(ValueError, match="profiler timings"):
        compare_pairs([reference] * 5, [captured] * 5)
