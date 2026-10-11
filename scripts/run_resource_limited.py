"""Run a desktop workload inside the shared, kernel-limited systemd slice.

This supervisor uses only the Python standard library. It stays outside the
workload slice, holds a per-user lock, and stops only its own service when the
host needs memory. Every command and its descendants belong to the same shared
slice; individual service limits are not a substitute for that aggregate cap.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


GIB = 1024**3
SLICE = "codex-workloads.slice"
MEMORY_MAX = 40 * GIB
SWAP_MAX = GIB
MIN_AVAILABLE = 12 * GIB
POLL_SECONDS = 2.0
SUPERVISOR_LEASE_SECONDS = 10.0
SWAP_IO_BYTES_PER_SECOND = 32 * 1024**2
SWAP_IO_SECONDS = 10.0
CGROUP_ROOT = Path("/sys/fs/cgroup")
REPO_ROOT = Path(__file__).resolve().parents[1]


class GuardError(RuntimeError):
    """The workload cannot be launched or monitored safely."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def key_values(path):
    return dict(line.split() for line in path.read_text().splitlines())


def read_meminfo(path=Path("/proc/meminfo")):
    result = {}
    for line in path.read_text().splitlines():
        name, value = line.split(":", 1)
        fields = value.split()
        if len(fields) == 2 and fields[1] == "kB":
            result[name] = int(fields[0]) * 1024
    for name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
        if name not in result:
            raise GuardError(f"Cannot read host {name} from {path}.")
    return result


def host_sample():
    memory = read_meminfo()
    vm = key_values(Path("/proc/vmstat"))
    return {
        "memory_total_bytes": memory["MemTotal"],
        "memory_available_bytes": memory["MemAvailable"],
        "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
        "swap_io_bytes": (int(vm["pswpin"]) + int(vm["pswpout"]))
        * os.sysconf("SC_PAGE_SIZE"),
    }


def numeric_limit(path):
    value = path.read_text().strip()
    if not value.isdigit():
        raise GuardError(
            f"Expected an enforced numeric limit in {path}, got {value!r}."
        )
    return int(value)


def verify_limits(cgroup):
    actual = (
        numeric_limit(cgroup / "memory.max"),
        numeric_limit(cgroup / "memory.swap.max"),
    )
    if actual != (MEMORY_MAX, SWAP_MAX):
        raise GuardError(
            f"{SLICE} has RAM/swap limits {actual}; expected "
            f"{(MEMORY_MAX, SWAP_MAX)} bytes. Run the explicit setup command."
        )
    return {"memory_max_bytes": actual[0], "swap_max_bytes": actual[1]}


def memory_sample(cgroup):
    return {
        "memory_current_bytes": int((cgroup / "memory.current").read_text()),
        "memory_peak_bytes": int((cgroup / "memory.peak").read_text()),
        "swap_current_bytes": int((cgroup / "memory.swap.current").read_text()),
        "swap_peak_bytes": int((cgroup / "memory.swap.peak").read_text()),
        "events": {
            name: int(value)
            for name, value in key_values(cgroup / "memory.events").items()
        },
    }


def systemctl(*args):
    result = subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode:
        raise GuardError(result.stderr.strip() or "systemctl failed.")
    return result.stdout.strip()


def slice_path(required=True):
    properties = dict(
        line.split("=", 1)
        for line in systemctl(
            "show", SLICE, "-p", "LoadState", "-p", "ControlGroup", "-p", "ActiveState"
        ).splitlines()
    )
    path = properties["ControlGroup"]
    if not path:
        if not required:
            return None
        raise GuardError(
            f"{SLICE} is inactive or missing. Run the setup command first."
        )
    relative = Path(path.removeprefix("/"))
    if ".." in relative.parts or relative.name != SLICE:
        raise GuardError(f"Unexpected slice cgroup path: {path!r}.")
    if f"user@{os.getuid()}.service" not in relative.parts:
        raise GuardError(f"{SLICE} is not inside this user's systemd manager.")
    return CGROUP_ROOT / relative


def current_cgroup():
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::/"):
            return CGROUP_ROOT / line[4:]
    raise GuardError("Unified cgroup v2 accounting is unavailable.")


def cgroup_pids(cgroup):
    pids = set()
    for path in cgroup.rglob("cgroup.procs"):
        try:
            pids.update(int(value) for value in path.read_text().split())
        except FileNotFoundError:
            # An empty service may disappear while the slice is inspected.
            continue
    return sorted(pids)


def ancestor_pids():
    result = set()
    pid = os.getpid()
    while pid:
        result.add(pid)
        status = Path(f"/proc/{pid}/status").read_text()
        pid = int(
            next(
                line.split()[1]
                for line in status.splitlines()
                if line.startswith("PPid:")
            )
        )
    return result


def existing_workers():
    """Find obvious legacy workers that bypassed the per-user guard lock."""
    ancestors = ancestor_pids()
    workers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            argv = [value for value in argv if value]
            if not argv:
                continue
            executable = Path(argv[0]).name
            if not re.fullmatch(
                r"python[\d.]*|pytest|uv|bash|zsh|sh|torchrun", executable
            ):
                continue
            workload = executable in ("pytest", "torchrun") or "pytest" in argv
            for arg in argv[1:]:
                name = Path(arg).name
                workload |= arg.startswith("scripts.") or (
                    name.endswith((".py", ".sh"))
                    and name.startswith(
                        ("train", "sweep", "experiment", "audit", "benchmark", "run_")
                    )
                    and name != Path(__file__).name
                )
            if workload:
                workers.append({"pid": int(entry.name), "command": argv})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise GuardError(
                f"Cannot inspect potential existing worker {entry.name}."
            ) from exc
    return workers


def runtime_directory():
    path = Path(f"/run/user/{os.getuid()}")
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise GuardError(
            f"Missing or incorrectly owned user runtime directory: {path}."
        )
    return path


@contextmanager
def workload_lock():
    path = runtime_directory() / "codex-workloads.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GuardError(
                "Another guarded workload is active; concurrent launches are disabled."
            ) from exc
        yield


def preflight():
    cgroup = slice_path()
    limits = verify_limits(cgroup)
    if current_cgroup().is_relative_to(cgroup):
        raise GuardError("The supervisor must stay outside the experiment slice.")
    pids = cgroup_pids(cgroup)
    if pids:
        raise GuardError(f"Existing workload descendants are still in {SLICE}: {pids}.")
    workers = existing_workers()
    if workers:
        raise GuardError(
            f"Existing unguarded experiment/test workers: {json.dumps(workers)}"
        )
    host = host_sample()
    if host["memory_available_bytes"] < MIN_AVAILABLE:
        raise GuardError("Host MemAvailable is below the 12 GiB desktop reserve.")
    return cgroup, {
        "slice": SLICE,
        "cgroup": str(cgroup),
        "limits": limits,
        "host": host,
        "aggregate_before": memory_sample(cgroup),
        "existing_workers": workers,
        "max_concurrent_guarded_commands": 1,
    }


def setup():
    with workload_lock():
        cgroup = slice_path(required=False)
        if cgroup is not None and cgroup_pids(cgroup):
            raise GuardError(
                "Refusing to change resource limits while workloads are active."
            )
        configured = dict(
            line.split("=", 1)
            for line in systemctl(
                "show", SLICE, "-p", "MemoryMax", "-p", "MemorySwapMax"
            ).splitlines()
        )
        for name, target in (("MemoryMax", MEMORY_MAX), ("MemorySwapMax", SWAP_MAX)):
            if name in configured and configured[name].isdigit():
                if int(configured[name]) < target:
                    raise GuardError(
                        f"Refusing to raise existing {name}={configured[name]}. "
                        "Investigate the lower limit before changing it explicitly."
                    )
        print(
            f"Setting shared {SLICE} to 40 GiB RAM and 1 GiB swap for this login.",
            flush=True,
        )
        systemctl(
            "set-property",
            "--runtime",
            SLICE,
            "MemoryAccounting=yes",
            "MemoryMax=40G",
            "MemorySwapMax=1G",
        )
        systemctl("start", SLICE)
        _, report = preflight()
        print(json.dumps(report, indent=2))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def service_worker(argv):
    """Keep exact kernel peak accounting available after the command exits."""
    cgroup, result_path, stop_path = map(Path, argv[:3])
    command = argv[3:]
    own_group = current_cgroup()
    if own_group.parent != cgroup:
        raise GuardError(
            f"Worker is outside its assigned aggregate slice: {own_group}."
        )
    verify_limits(cgroup)
    if not os.access(own_group / "cgroup.kill", os.W_OK):
        raise GuardError("The worker cannot enforce emergency descendant cleanup.")
    ready_path = result_path.parent / "allow"
    deadline = time.monotonic() + 20
    while not ready_path.exists():
        if stop_path.exists() or not result_path.parent.is_dir():
            raise GuardError("Launch cancelled by the resource supervisor.")
        if time.monotonic() >= deadline:
            raise GuardError("The resource supervisor did not authorize execution.")
        time.sleep(0.1)
    if stop_path.exists():
        raise GuardError("Launch cancelled by the resource supervisor.")
    if read_meminfo()["MemAvailable"] < MIN_AVAILABLE:
        raise GuardError("Host memory fell below the desktop reserve before execution.")
    previous = host_sample()
    heartbeat_stamp = ready_path.stat().st_mtime_ns
    last_heartbeat = last_time = time.monotonic()
    busy_seconds = 0.0
    reason = None
    interruption = []

    def request_stop(signum, _frame):
        interruption.append(signum)

    old_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while process.poll() is None:
            verify_limits(cgroup)
            now = time.monotonic()
            host = host_sample()
            reason, busy_seconds = stop_reason(
                host, previous, max(now - last_time, 1e-6), busy_seconds
            )
            if not ready_path.exists() or stop_path.exists():
                reason = "supervisor_cancelled"
            else:
                stamp = ready_path.stat().st_mtime_ns
                if stamp != heartbeat_stamp:
                    heartbeat_stamp, last_heartbeat = stamp, now
                if now - last_heartbeat >= SUPERVISOR_LEASE_SECONDS:
                    reason = "supervisor_heartbeat_expired"
            if interruption:
                reason = f"signal_{interruption[0]}"
            if reason is not None:
                break
            previous, last_time = host, now
            time.sleep(POLL_SECONDS)
        if reason is None:
            remaining = set(cgroup_pids(own_group)) - {os.getpid()}
            if remaining:
                reason = "command_exited_with_live_descendants"
        if reason is not None:
            # Save evidence before an emergency group kill also stops this wrapper.
            write_json(
                result_path,
                {
                    "returncode": 1,
                    "stop_reason": reason,
                    "cgroup": str(own_group),
                    **memory_sample(own_group),
                },
            )
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                process.poll()
                if not set(cgroup_pids(own_group)) - {os.getpid()}:
                    return 1
                time.sleep(0.1)
            (own_group / "cgroup.kill").write_text("1")
            raise GuardError("Emergency cgroup kill did not stop the worker.")
    except BaseException:
        # Errors in the monitor must never leave its command running without it.
        if set(cgroup_pids(own_group)) - {os.getpid()}:
            (own_group / "cgroup.kill").write_text("1")
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    returncode = process.wait()
    write_json(
        result_path,
        {
            "returncode": returncode,
            "cgroup": str(own_group),
            **memory_sample(own_group),
        },
    )
    return returncode if returncode >= 0 else 128 - returncode


def stop_service(unit, stop_path, cgroup):
    stop_path.touch()
    service_group = cgroup / unit
    try:
        systemctl("stop", unit)
    except (GuardError, OSError, subprocess.SubprocessError):
        # A broken user bus must not disable enforcement. Only this unique
        # service is killed; the shared parent and unrelated services are untouched.
        pass
    if service_group.is_dir() and cgroup_pids(service_group):
        (service_group / "cgroup.kill").write_text("1")
    deadline = time.monotonic() + 5
    while service_group.is_dir() and cgroup_pids(service_group):
        if time.monotonic() >= deadline:
            raise GuardError(f"Emergency cleanup could not stop descendants of {unit}.")
        time.sleep(0.1)


def stop_reason(host, previous, elapsed, busy_seconds):
    if host["memory_available_bytes"] < MIN_AVAILABLE:
        return "host_available_below_12_gib", busy_seconds
    rate = max(0, host["swap_io_bytes"] - previous["swap_io_bytes"]) / elapsed
    busy_seconds = busy_seconds + elapsed if rate >= SWAP_IO_BYTES_PER_SECOND else 0.0
    if busy_seconds >= SWAP_IO_SECONDS:
        return "sustained_host_swap_io", busy_seconds
    return None, busy_seconds


def command_worker_count(command):
    counts = [1]
    for index, arg in enumerate(command):
        for flag in ("--jobs", "--workers", "--max-workers", "--n-jobs"):
            if arg == flag and index + 1 < len(command):
                value = command[index + 1]
            elif arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
            else:
                continue
            if not value.isdigit() or int(value) < 1:
                raise GuardError(f"{flag} must specify a positive worker count.")
            counts.append(int(value))
    return max(counts)


def validate_command(command, allow_parallel=False):
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise GuardError("A command is required after --.")
    if command_worker_count(command) != 1 and not allow_parallel:
        raise GuardError(
            "Initial desktop policy requires one worker; increasing concurrency "
            "requires --peak-report from a complete representative one-worker run."
        )
    executable = shutil.which(command[0])
    if executable is None:
        raise GuardError(f"Command executable not found: {command[0]}.")
    # Resolving a venv's Python symlink would silently bypass that environment.
    return [os.path.abspath(executable), *command[1:]]


def projected_memory_budget(peak_report, workers):
    if peak_report is None:
        return MEMORY_MAX
    evidence = json.loads(peak_report.read_text())
    if (
        evidence["status"] != "completed"
        or not evidence["peak_is_complete_worker_measurement"]
        or evidence["worker_count"] != 1
    ):
        raise GuardError("Peak evidence must be a successful complete one-worker run.")
    peak = evidence["worker"]["memory_peak_bytes"]
    if not isinstance(peak, int) or peak <= 0:
        raise GuardError("Peak evidence does not contain a valid measured memory peak.")
    budget = math.ceil(1.5 * peak) * workers
    if budget > MEMORY_MAX:
        raise GuardError("Measured peak plus 50% headroom exceeds the shared RAM cap.")
    return budget


def run(command, report_path, peak_report=None):
    command = validate_command(command, allow_parallel=peak_report is not None)
    workers = command_worker_count(command)
    memory_budget = projected_memory_budget(peak_report, workers)
    unit = f"codex-worker-{uuid.uuid4().hex}.service"
    report = {
        "started_at": utc_now(),
        "command": command,
        "unit": unit,
        "status": "preflight",
        "minimum_host_available_bytes": None,
        "sampled_aggregate_peak_bytes": 0,
        "sampled_aggregate_swap_peak_bytes": 0,
        "peak_is_complete_worker_measurement": False,
        "worker_count": workers,
        "projected_memory_budget_bytes": memory_budget,
        "representative_peak_report": None if peak_report is None else str(peak_report),
    }
    write_json(report_path, report)
    try:
        with (
            workload_lock(),
            tempfile.TemporaryDirectory(
                prefix="codex-workload-", dir=runtime_directory()
            ) as temporary,
        ):
            cgroup, initial = preflight()
            report["preflight"] = initial
            if (
                initial["host"]["memory_available_bytes"]
                < MIN_AVAILABLE + memory_budget
            ):
                raise GuardError(
                    "Projected worker memory would leave less than 12 GiB available. "
                    "Without representative peak evidence, the projection is the "
                    "full 40 GiB cap."
                )
            result_path = Path(temporary) / "result.json"
            stop_path = Path(temporary) / "stop"
            report["worker_result_path"] = str(result_path)
            service_command = [
                "systemd-run",
                "--user",
                f"--unit={unit}",
                f"--slice={SLICE}",
                "--wait",
                "--pipe",
                "--expand-environment=no",
                "--property=KillMode=control-group",
                "--property=TimeoutStopSec=10s",
                "--property=KillSignal=SIGINT",
                "--property=OOMPolicy=kill",
                f"--working-directory={Path.cwd()}",
                *[
                    f"--setenv={name}"
                    for name in sorted(os.environ)
                    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name)
                ],
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                str(cgroup),
                str(result_path),
                str(stop_path),
                *command,
            ]
            interruption = []

            def request_stop(signum, _frame):
                interruption.append(signum)

            old_handlers = {
                signum: signal.signal(signum, request_stop)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }
            process = None
            reason = None
            try:
                process = subprocess.Popen(service_command, start_new_session=True)
                previous = initial["host"]
                last_time = time.monotonic()
                busy_seconds = 0.0
                report["status"] = "running"
                while process.poll() is None:
                    verify_limits(cgroup)
                    aggregate = memory_sample(cgroup)
                    host = host_sample()
                    now = time.monotonic()
                    reason, busy_seconds = stop_reason(
                        host, previous, max(now - last_time, 1e-6), busy_seconds
                    )
                    if interruption:
                        reason = f"signal_{interruption[0]}"
                    report["last_sample"] = {
                        "time": utc_now(),
                        "aggregate": aggregate,
                        "host": host,
                    }
                    minimum = report["minimum_host_available_bytes"]
                    report["minimum_host_available_bytes"] = min(
                        host["memory_available_bytes"],
                        host["memory_available_bytes"] if minimum is None else minimum,
                    )
                    report["sampled_aggregate_peak_bytes"] = max(
                        report["sampled_aggregate_peak_bytes"],
                        aggregate["memory_current_bytes"],
                    )
                    report["sampled_aggregate_swap_peak_bytes"] = max(
                        report["sampled_aggregate_swap_peak_bytes"],
                        aggregate["swap_current_bytes"],
                    )
                    write_json(report_path, report)
                    if reason is not None:
                        stop_service(unit, stop_path, cgroup)
                        break
                    (Path(temporary) / "allow").touch()
                    previous, last_time = host, now
                    time.sleep(POLL_SECONDS)
                returncode = process.wait(timeout=20)
                service_group = cgroup / unit
                if service_group.is_dir() and cgroup_pids(service_group):
                    reason = "systemd_run_exited_with_live_workers"
                    stop_service(unit, stop_path, cgroup)
            except BaseException:
                if process is not None:
                    try:
                        stop_service(unit, stop_path, cgroup)
                    except (
                        GuardError,
                        OSError,
                        subprocess.SubprocessError,
                    ) as cleanup_error:
                        report["cleanup_error"] = str(cleanup_error)
                raise
            finally:
                for signum, handler in old_handlers.items():
                    signal.signal(signum, handler)
            final = memory_sample(cgroup)
            report["aggregate_after"] = final
            before_events = initial["aggregate_before"]["events"]
            report["aggregate_event_changes"] = {
                name: count - before_events[name]
                for name, count in final["events"].items()
            }
            if result_path.exists():
                worker = json.loads(result_path.read_text())
                report["worker"] = worker
                if "stop_reason" in worker and reason is None:
                    reason = worker["stop_reason"]
                report["peak_is_complete_worker_measurement"] = (
                    returncode == 0 and worker["returncode"] == 0 and reason is None
                )
            if any(
                report["aggregate_event_changes"][name] > 0
                for name in ("oom", "oom_kill")
            ):
                reason = "aggregate_memory_limit"
                report["peak_is_complete_worker_measurement"] = False
            report["returncode"] = returncode
            report["stop_reason"] = reason
            report["status"] = (
                "completed" if returncode == 0 and reason is None else "failed"
            )
            report["finished_at"] = utc_now()
            write_json(report_path, report)
            print(f"Resource report: {report_path}", flush=True)
            if report["status"] == "failed":
                print(
                    "Workload stopped. Investigate and reduce it before retrying; "
                    "no automatic retry was made.",
                    file=sys.stderr,
                )
                return returncode if returncode > 0 else 1
            if not report["peak_is_complete_worker_measurement"]:
                raise GuardError(
                    "Command exited but complete worker peak accounting is missing."
                )
            return 0
    except (GuardError, OSError, ValueError, subprocess.SubprocessError) as exc:
        report.update(status="failed", error=str(exc), finished_at=utc_now())
        write_json(report_path, report)
        raise GuardError(f"{exc} Resource report: {report_path}") from exc


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv and argv[0] == "_worker":
            return service_worker(argv[1:])
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="action", required=True)
        commands.add_parser(
            "setup", help="Explicitly establish the shared runtime limits."
        )
        commands.add_parser(
            "check", help="Verify limits, host memory, and absence of existing workers."
        )
        run_parser = commands.add_parser(
            "run", help="Run one command under the shared cap and monitor it."
        )
        run_parser.add_argument("--report", type=Path)
        run_parser.add_argument(
            "--peak-report",
            type=Path,
            help="Complete one-worker report for a measured memory budget.",
        )
        run_parser.add_argument("command", nargs=argparse.REMAINDER)
        args = parser.parse_args(argv)
        if args.action == "setup":
            setup()
        elif args.action == "check":
            with workload_lock():
                _, report = preflight()
                print(json.dumps(report, indent=2))
        else:
            path = args.report
            if path is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                path = (
                    REPO_ROOT
                    / "logs"
                    / "resource_guard"
                    / f"{stamp}-{uuid.uuid4().hex[:8]}.json"
                )
            return run(args.command, path.resolve(), args.peak_report)
        return 0
    except (GuardError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Resource guard: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
