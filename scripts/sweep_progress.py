"""Terminal progress display for scripts/sweep_autoencoder.py.

Display only: nothing here affects what is trained or how it is scored. The
sweep calls three methods per run (start_run, on_line for each line of trainer
output, finish_run) and close() at the end. How that is shown depends on mode:

  "table"   one table of all runs, redrawn in place (needs a terminal)
  "lines"   one line per finished run (stdout piped or redirected)
  "stream"  trainer output echoed live, plus one line per finished run
"""

import re
import shutil
import sys
import time

RULE = "-" * 78
BAR_WIDTH = 20
# trainer progress block header and its pooled held-out line
STEP_RE = re.compile(r"Step (\d+)/(\d+)")
HELD_OUT_RE = re.compile(r"held-out \(pooled\):\s*([0-9.eE+-]+)")


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


class SweepProgress:
    def __init__(self, jobs, num_steps, log_dir, mode):
        self.rows = [
            {"config": f"{j.model} L{j.latent_channels}", "clip": j.clip,
             "seed": j.seed, "status": "pending", "step": 0, "mse": None,
             "seconds": None}
            for j in jobs
        ]  # fmt: skip
        self.num_steps, self.log_dir, self.mode = num_steps, log_dir, mode
        self.start = time.time()
        self.run_start = None
        self.first_mark = None  # (time, step) of the running job's first Step line
        self.rate = None  # steps/s of the running job, measured between Step lines
        self.current = None  # index of the running row
        self.last_line = ""
        self.height = 0  # lines drawn last time, to move the cursor back over
        self.drawn_at = 0.0
        self.failures = []

    # --- called by the sweep ------------------------------------------------

    def start_run(self, i, log_path):
        self.current, self.run_start = i, time.time()
        self.first_mark, self.rate = None, None
        self.rows[i]["status"] = "running"
        if self.mode == "stream":
            row = self.rows[i]
            print(
                f"[{i + 1:2d}/{len(self.rows)}] hold out walk{row['clip']}, "
                f"seed {row['seed']}\n    log: {log_path}",
                flush=True,
            )
        self.draw(force=True)

    def on_line(self, line):
        """One line of trainer output from the running job."""
        if self.mode == "stream":
            sys.stdout.write(line)
            sys.stdout.flush()
            return
        row = self.rows[self.current]
        if match := STEP_RE.search(line):
            row["step"] = int(match.group(1))
            # rate from Step lines only, so process startup doesn't count as slow steps
            if self.first_mark is None:
                self.first_mark = (time.time(), row["step"])
            elif row["step"] > self.first_mark[1]:
                t0, step0 = self.first_mark
                self.rate = (row["step"] - step0) / (time.time() - t0)
        elif match := HELD_OUT_RE.search(line):
            row["mse"] = float(match.group(1))  # latest validation, provisional
        if line.strip():
            self.last_line = line.strip()
        self.draw()

    def finish_run(self, i, scores, log_path):
        """scores: the sweep's score() dict, or None if the run failed."""
        row = self.rows[i]
        row["seconds"] = time.time() - self.run_start  # includes scoring
        if scores is not None:
            row["status"], row["step"], row["mse"] = (
                "done",
                self.num_steps,
                scores["model"],
            )
        else:
            row["status"] = "FAILED"
            self.failures.append(log_path)
        self.current = None
        if self.mode == "table":
            self.draw(force=True)
        else:
            self.print_finished(i, scores, log_path)

    def close(self):
        if self.failures:
            print("failed runs (full output in):")
            for path in self.failures:
                print(f"  {path}")
        print(f"\nfinished in {format_duration(time.time() - self.start)}")

    # --- line modes -----------------------------------------------------------

    def print_finished(self, i, scores, log_path):
        row, eta = self.rows[i], self.eta()
        if scores is None:
            result = f"FAILED; full output in {log_path}"
        else:
            result = (
                f"held-out {scores['model']:.4f} (nearest window "
                f"{scores['nearest']:.4f}, PCA {scores['pca']:.4f})"
            )
        print(
            f"[{i + 1:2d}/{len(self.rows)}] walk{row['clip']} seed {row['seed']}: "
            f"{result}  {format_duration(row['seconds'])}"
            + (f", ETA {format_duration(eta)}" if eta is not None else ""),
            flush=True,
        )

    # --- table mode -----------------------------------------------------------

    def eta(self):
        """Seconds left = running job's remaining time + pending runs x per-run time.

        Running job: (num_steps - step) / its measured rate (falls back to
        per-run time minus time spent). Per-run time: mean of finished runs, or
        before any has finished, startup (run start to first Step line) +
        num_steps / rate. None until two Step lines have given a rate.
        """
        finished = [r["seconds"] for r in self.rows if r["seconds"] is not None]
        pending = sum(r["status"] == "pending" for r in self.rows)
        running = self.rows[self.current] if self.current is not None else None
        if finished:
            per_run = sum(finished) / len(finished)
        elif running and self.rate:
            per_run = (self.first_mark[0] - self.run_start) + self.num_steps / self.rate
        else:
            return None
        if not running:
            current_left = 0.0
        elif self.rate:
            current_left = (self.num_steps - running["step"]) / self.rate
        else:  # new run, no rate yet
            current_left = max(per_run - (time.time() - self.run_start), 0.0)
        return current_left + pending * per_run

    def row_text(self, row):
        status = row["status"]
        if status == "running":
            fraction = row["step"] / self.num_steps
            seconds = time.time() - self.run_start
        elif status == "done":
            fraction, seconds = 1.0, row["seconds"]
        else:
            fraction, seconds = 0.0, row["seconds"]
        filled = round(fraction * BAR_WIDTH)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled) if status != "pending" else ""
        pct = f"{100 * fraction:3.0f}%" if status != "pending" else ""
        mse = ""
        if row["mse"] is not None and status in ("running", "done"):
            mse = f"{row['mse']:.4f}" + ("*" if status == "running" else " ")
        time_text = format_duration(seconds) if seconds is not None else ""
        return (
            f" walk{row['clip']:<5s}{row['seed']:>4d}  {status:<8s} {bar:<{BAR_WIDTH}s}"
            f" {pct:>4s}  {mse:>10s}  {time_text:>6s}"
        )

    def lines(self, max_lines):
        done = sum(r["status"] == "done" for r in self.rows)
        failed = len(self.failures)
        eta = self.eta()
        header = (
            f" {done}/{len(self.rows)} done"
            + (f", {failed} FAILED" if failed else "")
            + f"   elapsed {format_duration(time.time() - self.start)}"
            + f"   ETA {format_duration(eta) if eta is not None else '…'}"
            # a slow GPU shows here, rather than looking like a wrong ETA
            + (
                f"   {self.rate:.1f} steps/s"
                if self.current is not None and self.rate
                else ""
            )
        )
        head = [
            RULE,
            header,
            RULE,
            f" {'held out':<9s}{'seed':>4s}  {'status':<8s} {'progress':<{BAR_WIDTH}s}"
            f" {'':>4s}  {'held-out MSE':>10s}  {'time':>6s}",
        ]
        foot = [
            RULE,
            " * latest validation of the running job; final scores when done",
            f" trainer: {self.last_line if self.current is not None else '—'}",
            f" logs: {self.log_dir}",
        ]
        # keep the table on screen: show a window of rows around the running one
        room = max(max_lines - len(head) - len(foot), 3)
        rows = self.rows
        if len(rows) > room:
            anchor = self.current if self.current is not None else 0
            first = min(max(anchor - room // 3, 0), len(rows) - room)
            rows = rows[first : first + room]
        return head + [self.row_text(r) for r in rows] + foot

    def draw(self, force=False):
        """Redraw the table in place (ANSI cursor-up + clear), at most 4x/s."""
        if self.mode != "table" or (not force and time.time() - self.drawn_at < 0.25):
            return
        width, height = shutil.get_terminal_size()
        lines = [line[: width - 1] for line in self.lines(height - 1)]
        out = f"\x1b[{self.height}F\x1b[J" if self.height else ""
        sys.stdout.write(out + "\n".join(lines) + "\n")
        sys.stdout.flush()
        self.height, self.drawn_at = len(lines), time.time()
