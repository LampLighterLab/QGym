"""Capture the four VSim Milestone-0 domain-randomization discriminators."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


CASES = (
    ("one_set_off", "off"),
    ("many_sets_fixed_friction", "friction-fixed"),
    ("many_sets_sampled_friction", "friction"),
    ("one_set_episode_pd", "pd"),
)


def run(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def topology(result):
    return result["protocol"]["native_topology_details"]


def validate_results(results, num_envs):
    off = results["one_set_off"]
    fixed = results["many_sets_fixed_friction"]
    sampled = results["many_sets_sampled_friction"]
    pd = results["one_set_episode_pd"]

    one_set = {
        "environment_set_count": 1,
        "total_environment_count": num_envs,
        "min_environments_per_set": num_envs,
        "max_environments_per_set": num_envs,
    }
    many_sets = {
        "environment_set_count": num_envs,
        "total_environment_count": num_envs,
        "min_environments_per_set": 1,
        "max_environments_per_set": 1,
    }
    if topology(off) != one_set or topology(pd) != one_set:
        raise RuntimeError("off and PD cases must use one VSim environment set")
    if topology(fixed) != many_sets or topology(sampled) != many_sets:
        raise RuntimeError("fixed and sampled friction must use singleton sets")

    friction = fixed["applied_domain_randomization"]["contact_friction"]
    if friction != {"min": 1.0, "mean": 1.0, "max": 1.0, "std": 0.0}:
        raise RuntimeError(
            f"fixed-friction control did not apply exactly 1.0: {friction}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="go2trot")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target_env_steps", type=int, default=32768)
    parser.add_argument("--min_steps", type=int, default=50)
    parser.add_argument("--benchmark_repeats", type=int, default=5)
    parser.add_argument("--trace_repeats", type=int, default=1)
    args = parser.parse_args(argv)

    nsys = shutil.which("nsys")
    if nsys is None:
        parser.error("nsys is required for the Milestone-0 VSim capture")
    args.output.mkdir(parents=True, exist_ok=False)

    worker = Path(__file__).with_name("benchmark_domain_randomization.py")
    commands = {}
    for label, bundle in CASES:
        common = [
            sys.executable,
            str(worker),
            "--task",
            args.task,
            "--backend",
            "vsim",
            "--dr",
            bundle,
            "--num_envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--target_env_steps",
            str(args.target_env_steps),
            "--min_steps",
            str(args.min_steps),
        ]

        benchmark_json = args.output / f"{label}.json"
        benchmark = [
            *common,
            "--repeats",
            str(args.benchmark_repeats),
            "--out",
            str(benchmark_json),
        ]
        run(benchmark, args.output / f"{label}.log")

        trace_json = args.output / f"{label}.trace.json"
        report = args.output / label
        capture = [
            nsys,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--python-sampling=true",
            "--python-sampling-frequency=1000",
            "--force-overwrite=true",
            f"--output={report}",
            *common,
            "--repeats",
            str(args.trace_repeats),
            "--out",
            str(trace_json),
        ]
        run(capture, args.output / f"{label}.trace.log")
        commands[label] = {"benchmark": benchmark, "capture": capture}

    results = {
        label: json.loads((args.output / f"{label}.json").read_text())
        for label, _ in CASES
    }
    validate_results(results, args.num_envs)
    trace_results = {
        label: json.loads((args.output / f"{label}.trace.json").read_text())
        for label, _ in CASES
    }
    validate_results(trace_results, args.num_envs)
    for label, _ in CASES:
        report = args.output / f"{label}.nsys-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError(f"missing Nsight report for {label}: {report}")

    nsys_version = subprocess.run(
        [nsys, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "nsys_version": nsys_version,
        "protocol": {
            "task": args.task,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "target_env_steps": args.target_env_steps,
            "min_steps": args.min_steps,
            "benchmark_repeats": args.benchmark_repeats,
            "trace_repeats": args.trace_repeats,
        },
        "commands": commands,
        "artifacts": {
            label: {
                "benchmark": f"{label}.json",
                "trace_benchmark": f"{label}.trace.json",
                "nsys_report": f"{label}.nsys-rep",
            }
            for label, _ in CASES
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
