"""Profile the fixed 100 Hz simulation benchmark in a fresh process.

Run once with --tool nsys for a native CUDA timeline, and separately with
--tool py-spy for a sampled host-stack flamegraph. Nsight requires its system
installation; py-spy comes from `uv sync --frozen --group profiling`. For VSim,
launch this module with the usual --env-file .env.vsim process environment.

Profiling durations are diagnostic, never performance-gate measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TIMED_FUNCTION = "profiled_steps"


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("nsys", "py-spy"), required=True)
    parser.add_argument("--task", choices=("go2trot", "pendulum"), default="go2trot")
    parser.add_argument("--backend", choices=("cpu", "warp", "vsim"), required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--sets", type=int, default=1)
    parser.add_argument("--profile", default="task_timeout")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--rate",
        type=int,
        default=200,
        help="host stack samples per second; inspect capture lag",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new artifact directory"
    )
    args = parser.parse_args(argv)
    if args.tool == "nsys" and args.backend == "cpu":
        parser.error("use --tool py-spy for the CPU baseline; nsys captures CUDA here")
    return args


def worker_command(args):
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_simulation",
        "run",
        "--task",
        args.task,
        "--backend",
        args.backend,
        "--num-envs",
        str(args.num_envs),
        "--sets",
        str(args.sets),
        "--profile",
        args.profile,
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--repeats",
        "1",
        "--seed",
        str(args.seed),
        "--output",
        str(args.output / "workload.json"),
    ]
    if args.tool == "nsys":
        command.append("--capture")
    return command


def capture_command(args, executable):
    worker = worker_command(args)
    if args.tool == "nsys":
        return [
            executable,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--kill=none",
            "--export=sqlite",
            "--force-overwrite=false",
            f"--output={args.output / 'cuda'}",
            *worker,
        ]
    return [
        executable,
        "record",
        "--format",
        "speedscope",
        "--rate",
        str(args.rate),
        "--native",
        "--idle",
        "--output",
        str(args.output / "host.raw.speedscope.json"),
        "--",
        *worker,
    ]


def trim_to_timed_stacks(profile):
    """Keep actual sampled stacks under the timed batch, excluding setup/warmup.

    The output is a compacted sampled profile, not an aligned wall-clock timeline.
    Only threads whose stack includes the timed Python function are retained.
    Native-only worker threads are excluded; the separate Nsight capture records
    GPU execution.
    """
    frames = profile["shared"]["frames"]
    roots = {i for i, frame in enumerate(frames) if frame["name"] == TIMED_FUNCTION}
    result = {**profile, "profiles": [], "activeProfileIndex": 0}
    sample_count = 0
    for source in profile["profiles"]:
        if source["type"] != "sampled":
            raise ValueError("expected py-spy's sampled speedscope format")
        samples, weights = [], []
        for stack, weight in zip(source["samples"], source["weights"], strict=True):
            for index, frame in enumerate(stack):
                if frame in roots:
                    samples.append(stack[index:])
                    weights.append(weight)
                    break
        if samples:
            result["profiles"].append(
                {
                    **source,
                    "name": source["name"] + " — timed host stacks (compacted)",
                    "startValue": 0,
                    "endValue": sum(weights),
                    "samples": samples,
                    "weights": weights,
                }
            )
            sample_count += len(samples)
    if not sample_count:
        raise RuntimeError(
            f"No {TIMED_FUNCTION} stack samples captured. Keep the raw profile/log; "
            "check profiler access and increase --steps if the batch was too short."
        )
    return result, sample_count


def host_flamegraph_svg(profile):
    """Render filtered sampled stacks as a standalone, weighted SVG flamegraph.

    Width is aggregate sample weight, not elapsed GPU time. The layout groups
    equal call paths; horizontal placement does not represent event chronology.
    """

    def node(label):
        return {"label": label, "weight": 0.0, "children": {}}

    root = node("Timed host stack samples (including waits)")
    frames = profile["shared"]["frames"]
    depth = 1
    samples = 0
    units = {source["unit"] for source in profile["profiles"]}
    if len(units) != 1:
        raise ValueError("flamegraph requires one shared sample-weight unit")
    unit = units.pop()
    for index, source in enumerate(profile["profiles"]):
        thread = root["children"].setdefault(index, node(source["name"]))
        for stack, weight in zip(source["samples"], source["weights"], strict=True):
            root["weight"] += weight
            thread["weight"] += weight
            current = thread
            for frame_id in stack:
                frame = frames[frame_id]
                label = frame["name"]
                if "file" in frame:
                    label += f" ({frame['file']}:{frame.get('line', '?')})"
                current = current["children"].setdefault(frame_id, node(label))
                current["weight"] += weight
            depth = max(depth, len(stack) + 2)
            samples += 1

    width, row_height = 1400, 20
    top, bottom = 76, 42
    height = top + depth * row_height + bottom
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(width),
        height=str(height),
        viewBox=f"0 0 {width} {height}",
        role="img",
        attrib={"aria-label": "Sampled host-stack flamegraph"},
    )
    ET.SubElement(svg, "style").text = (
        "text{font:12px monospace;fill:#171717}"
        ".frame:hover rect{stroke:#111;stroke-width:2}"
    )
    ET.SubElement(svg, "rect", width="100%", height="100%", fill="#fafafa")
    ET.SubElement(svg, "text", x="10", y="22").text = (
        f"Timed host stacks — {samples} samples; "
        f"aggregate weight {root['weight']:.6g} {unit}"
    )
    ET.SubElement(svg, "text", x="10", y="42").text = (
        "Width = sample weight including waits; not GPU time. "
        "Native symbols/source depend on installed binaries."
    )
    ET.SubElement(svg, "text", x="10", y="62").text = (
        "Setup/warmup and native-only worker threads are excluded. "
        "Hover for full frames; inspect the CUDA timeline separately."
    )
    usable_width = width - 20

    def draw(current, x, level):
        span = usable_width * current["weight"] / root["weight"]
        y = height - bottom - (level + 1) * row_height
        group = ET.SubElement(svg, "g", attrib={"class": "frame"})
        percent = 100 * current["weight"] / root["weight"]
        ET.SubElement(
            group, "title"
        ).text = f"{current['label']}\n{current['weight']:.6g} {unit} ({percent:.2f}%)"
        shade = hashlib.sha256(current["label"].encode()).digest()[0]
        ET.SubElement(
            group,
            "rect",
            x=f"{x:.3f}",
            y=str(y),
            width=f"{span:.3f}",
            height=str(row_height - 1),
            fill=f"rgb(245,{145 + shade // 3},80)",
        )
        letters = max(0, int((span - 8) / 7.3))
        if letters > 3:
            label = current["label"]
            if len(label) > letters:
                label = label[: letters - 1] + "…"
            ET.SubElement(group, "text", x=f"{x + 4:.3f}", y=str(y + 14)).text = label
        for child in sorted(current["children"].values(), key=lambda row: row["label"]):
            draw(child, x, level + 1)
            x += usable_width * child["weight"] / root["weight"]

    draw(root, 10, 0)
    ET.SubElement(svg, "text", x="10", y=str(height - 14)).text = (
        "Grouped call paths; horizontal position is not chronology. "
        "Raw and filtered Speedscope files retain the underlying samples."
    )
    return ET.tostring(svg, encoding="unicode") + "\n"


def cuda_capture_summary(path):
    """Confirm that the native CUDA capture contains individual kernel events."""
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as database:
        count = database.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        if not count:
            raise RuntimeError("Nsight capture contains no CUDA kernel events")
        rows = database.execute(
            "SELECT s.value, COUNT(*), SUM(k.end-k.start) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.demangledName "
            "GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 20"
        ).fetchall()
    return {
        "kernel_events": count,
        "top_kernels_by_summed_duration": [
            {"name": name, "calls": calls, "summed_duration_ns": duration}
            for name, calls, duration in rows
        ],
        "note": (
            "Summed kernel durations may overlap across streams; "
            "they are not wall time."
        ),
    }


def mark_profiled_result(path, tool):
    """Exclude externally profiled worker timings from benchmark comparisons."""
    result = json.loads(path.read_text())
    result["capture"] = True
    result["profiler"] = tool
    result["performance_gate"] = False
    path.write_text(json.dumps(result, indent=2) + "\n")


def main(argv=None):
    args = get_args(argv)
    executable = shutil.which(args.tool)
    if executable is None:
        raise SystemExit(
            f"{args.tool} is unavailable. Install Nsight Systems for nsys, or run "
            "`uv sync --frozen --group profiling` for py-spy."
        )
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    command = capture_command(args, executable)
    manifest = {
        "status": "starting",
        "tool": args.tool,
        "version": subprocess.check_output(
            [executable, "--version"], text=True
        ).strip(),
        "command": command,
        "frequency_hz": 100,
        "scope": (
            "Native CUDA kernel/API timeline across streams, including graph nodes; "
            "NVTX workload regions. Native CPU stack sampling is disabled."
            if args.tool == "nsys"
            else "Sampled Python/native host stacks below the timed-batch function, "
            "including idle/waiting stacks; no GPU timing or native-only "
            "worker threads. "
            "Native symbol/source resolution depends on the installed binaries."
        ),
        "performance_gate": False,
    }
    manifest_path = args.output / "capture.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "capture.log").open("w") as log:
        subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    # py-spy does not use CUDA capture APIs, but its timings are still profiled.
    mark_profiled_result(args.output / "workload.json", args.tool)
    if args.tool == "py-spy":
        raw = json.loads((args.output / "host.raw.speedscope.json").read_text())
        trimmed, count = trim_to_timed_stacks(raw)
        (args.output / "host.speedscope.json").write_text(json.dumps(trimmed) + "\n")
        (args.output / "host.flamegraph.svg").write_text(host_flamegraph_svg(trimmed))
        manifest["timed_stack_samples"] = count
        manifest["view"] = (
            "Open host.flamegraph.svg directly, or load host.speedscope.json "
            "in speedscope; raw capture retains setup."
        )
    else:
        manifest["cuda"] = cuda_capture_summary(args.output / "cuda.sqlite")
        manifest["view"] = (
            "Open cuda.nsys-rep in Nsight Systems for the native CUDA timeline."
        )
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
