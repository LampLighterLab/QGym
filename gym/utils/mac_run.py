"""Launch scripts/play.py under mjpython so the MuJoCo viewer works on macOS.

Usage:
    uv run --frozen mac_run --task=go2trot [play.py args...]

uv-managed Pythons locate libpython via @executable_path, which breaks when
mjpython dlopens them from inside its app bundle. Pointing DYLD_LIBRARY_PATH
at the interpreter's LIBDIR fixes the lookup.
"""

import os
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAY_SCRIPT = REPO_ROOT / "scripts" / "play.py"


def main():
    args = sys.argv[1:]
    if sys.platform != "darwin":
        os.execv(sys.executable, [sys.executable, str(PLAY_SCRIPT), *args])

    mjpython = Path(sys.executable).parent / "mjpython"
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
        p
        for p in (sysconfig.get_config_var("LIBDIR"), env.get("DYLD_LIBRARY_PATH"))
        if p
    )
    os.execve(str(mjpython), [str(mjpython), str(PLAY_SCRIPT), *args], env)
