"""Fetch Q2's pinned SDK source before installing the optional Unitree extra.

The upstream wheel omits native CRC libraries, so uv uses an editable checkout.
Run this with Python 3.11+ before uv sync; it needs only Python and Git.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tomllib


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch_sdk(project_root: Path) -> Path:
    """Create the pinned checkout, or validate an existing one without changing it."""
    project_root = project_root.resolve()
    with (project_root / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    pin = config["tool"]["q2"]["unitree-sdk"]
    source = config["tool"]["uv"]["sources"]["unitree-sdk2py"]
    target = (project_root / source["path"]).resolve()
    revision = pin["rev"]

    if target.exists():
        if not target.is_dir() or not (target / ".git").exists():
            raise RuntimeError(
                f"{target} already exists and is not a Git checkout. "
                "Move it aside yourself before fetching the SDK."
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--no-checkout", pin["git"], str(target))
        git("checkout", "--detach", revision, cwd=target)

    head = git("rev-parse", "HEAD", cwd=target)
    if head != revision:
        raise RuntimeError(
            f"SDK checkout {target} is at {head}; pyproject.toml requires {revision}. "
            "Preserve your work and explicitly check out the required revision, "
            "or move the checkout aside and run this script again."
        )
    if git("status", "--porcelain", "--untracked-files=all", cwd=target):
        raise RuntimeError(
            f"SDK checkout {target} has local changes. Preserve or resolve them "
            "yourself before using the pinned SDK; this script will not overwrite them."
        )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    try:
        target = fetch_sdk(args.project_root)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        print(f"Unitree SDK fetch failed: {detail}", file=sys.stderr)
        return 1
    print(f"Pinned Unitree SDK source ready: {target}")
    print("Next: follow the CycloneDDS prerequisites in README_DEPLOY.md, then run")
    print("  uv sync --frozen --extra unitree_sdk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
