"""Simulation and offline deployment entry points must work without the SDK."""

import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SDK = """\
import importlib.abc
import sys

class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"unitree_sdk2py", "cyclonedds"}:
            raise ModuleNotFoundError("SDK deliberately unavailable", name=fullname)

sys.meta_path.insert(0, BlockSDK())
"""


def _without_sdk(code):
    return subprocess.run(
        [sys.executable, "-c", BLOCK_SDK + textwrap.dedent(code)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_simulation_and_offline_deployment_import_without_sdk():
    result = _without_sdk("""
        import gym.envs
        import learning
        import go2_deploy.deploy
        import go2_deploy.rl_controller
        from go2_deploy.utility import deploy_utility

        assert "unitree_sdk2py" not in sys.modules
        assert "cyclonedds" not in sys.modules
    """)
    assert result.returncode == 0, result.stderr


def test_deploy_help_without_sdk():
    result = _without_sdk("""
        import runpy

        sys.argv = ["go2_deploy.deploy", "--help"]
        runpy.run_module("go2_deploy.deploy", run_name="__main__")
    """)
    assert result.returncode == 0, result.stderr
    assert "interface" in result.stdout


def test_deploy_missing_sdk_explains_extra():
    result = _without_sdk("""
        import runpy

        sys.platform = "linux"
        sys.argv = ["go2_deploy.deploy", "eth0"]
        runpy.run_module("go2_deploy.deploy", run_name="__main__")
    """)
    assert result.returncode == 2, result.stderr
    assert "--extra unitree_sdk" in result.stderr
    assert "README_DEPLOY.md" in result.stderr
    assert "Traceback" not in result.stderr
