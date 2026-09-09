import json
import signal
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.run_full_domain_randomization_campaign as full_campaign
from scripts.eval_policy import crossed_parameter_samples
from scripts.run_full_domain_randomization_campaign import (
    EVAL_DOMAINS,
    Campaign,
    audit_completed_cells,
    campaign_cases,
    checkpoint_record_valid,
    checkpoint_valid,
    domain_arguments,
    evaluation_cases,
    mark_campaign_stopped,
    reject_engine_overflow,
    run_pool,
    run_stage,
    run_training_and_evaluation,
    sha256_file,
    speed_cases,
    subprocess_environment,
    summarize_evaluation,
    training_cases,
    validate_training,
)


def _protocol():
    return {
        "include_speed": True,
        "excluded_training": [],
        "backends": [
            {"label": "cpu", "physics_backend": "mujoco", "device": "cpu"},
            {"label": "warp", "physics_backend": "mujoco", "device": "cuda:0"},
            {"label": "vsim", "physics_backend": "vsim", "device": "cuda:0"},
        ],
        "bundles": {
            "off": "off",
            "friction": "friction-only",
            "pd": "pd-only",
            "mass": "mass-only",
            "all": "config",
        },
        "seeds": [7, 17, 27],
        "speed_common_num_envs": 256,
        "speed_production_num_envs": 4096,
        "train_iterations": 1000,
        "checkpoints": [100, 250, 500, 750, 1000],
        "intermediate_domains": ["nominal", "combined_in"],
        "eval_domains": EVAL_DOMAINS,
    }


def test_full_campaign_matrix_is_explicit_and_unique():
    protocol = _protocol()
    cases = campaign_cases(protocol)

    assert len(speed_cases(protocol)) == 25
    assert len(training_cases(protocol)) == 45
    assert len(evaluation_cases(protocol)) == 1575
    assert len({(case["stage"], case["label"]) for case in cases}) == len(cases)


def test_reduced_baseline_plan_excludes_cpu_dr_and_has_no_pending_speed_cells():
    args = full_campaign.get_args(
        [
            "--backends",
            "cpu",
            "warp",
            "vsim",
            "--bundles",
            "off",
            "all",
            "--exclude-training",
            "cpu:all",
            "--skip-speed",
            "--evaluate-after-training",
            "--seeds",
            "7",
            "--train-iterations",
            "500",
            "--checkpoints",
            "100",
            "250",
            "500",
            "--eval-domains",
            "nominal",
            "combined_in",
        ]
    )
    protocol = full_campaign.make_protocol(args)
    assert protocol["evaluation_schedule"] == "after_training_cell"
    assert speed_cases(protocol) == []
    training = training_cases(protocol)
    assert [case["label"] for case in training] == [
        "cpu_off_seed7",
        "warp_off_seed7",
        "vsim_off_seed7",
        "warp_all_seed7",
        "vsim_all_seed7",
    ]
    evaluations = evaluation_cases(protocol)
    assert len(evaluations) == 50
    assert all(case["training_label"] != "cpu_all_seed7" for case in evaluations)
    # CPU remains an evaluation target for GPU policies, including full DR.
    assert any(
        case["eval_backend"] == "cpu" and case["bundle"] == "all"
        for case in evaluations
    )
    cfg = protocol["environment_config_template"]
    assert cfg["control"]["ctrl_frequency"] == 100
    assert cfg["control"]["desired_sim_frequency"] == 100
    assert cfg["sim_dt"] == pytest.approx(0.01)
    assert protocol["rollout_steps_per_env"] == 16
    domain = protocol["eval_domains"]["combined_in"]
    dr = cfg["domain_randomization"]
    assert domain["link_mass_scale"] == dr["startup"]["link_mass_scale_range"]
    assert domain["damping_scale"] == dr["episode"]["scale_ranges"]["d_gains"]
    assert "pilot" in protocol["interpretation"]["seed_evidence"]


@pytest.mark.parametrize(
    "exclusions", [["cpu:off", "cpu:all"], ["warp:all"], ["cpu:all", "cpu:all"]]
)
def test_training_exclusions_reject_empty_or_mistyped_plans(exclusions):
    with pytest.raises(SystemExit):
        full_campaign.get_args(
            [
                "--backends",
                "cpu",
                "--bundles",
                "off",
                "all",
                "--exclude-training",
                *exclusions,
            ]
        )


def test_intermediate_and_final_evaluation_scope():
    cases = evaluation_cases(_protocol())
    intermediate = [case for case in cases if case["checkpoint_iteration"] == 250]
    final = [case for case in cases if case["checkpoint_iteration"] == 1000]

    assert len(intermediate) == 45 * 2
    assert all(case["train_backend"] == case["eval_backend"] for case in intermediate)
    assert {case["domain"] for case in intermediate} == {"nominal", "combined_in"}
    assert len(final) == 45 * 3 * len(EVAL_DOMAINS)


def test_eval_domain_arguments_name_every_enabled_axis():
    arguments = domain_arguments("combined_stress")

    assert arguments[:2] == ["--domain-randomization", "config"]
    for option in (
        "--contact_friction_grid",
        "--stiffness-scale-range",
        "--damping-scale-range",
        "--link-mass-scale-range",
    ):
        assert option in arguments


def test_training_command_uses_campaign_worker(monkeypatch):
    protocol = {
        **_protocol(),
        "task": "go2trot",
        "train_num_envs": 4096,
        "save_interval": 50,
    }
    cell = training_cases(protocol)[0]
    monkeypatch.setattr(
        full_campaign,
        "write_resolved_training_config",
        lambda *_args: None,
    )

    command = full_campaign.training_command(
        full_campaign.LOG_ROOT / "campaign",
        cell,
        protocol,
    )

    assert command[3:5] == ["-m", "scripts.train_domain_randomization"]
    assert command[command.index("--dr-bundle") + 1] == cell["bundle"]
    assert "--domain-randomization" not in command


def test_crossed_parameter_samples_balance_command_cases():
    command_cases = np.asarray(["stand", "forward", "stand", "forward"])
    values = crossed_parameter_samples(command_cases, 0.8, 1.2, width=3, seed=9)

    np.testing.assert_array_equal(
        values[command_cases == "stand"], values[command_cases == "forward"]
    )
    assert values.shape == (4, 3)
    assert values.min() >= 0.8
    assert values.max() <= 1.2


def test_checkpoint_validation_reads_embedded_iteration(tmp_path):
    path = tmp_path / "model_100.pt"
    checkpoint = {
        "actor_state_dict": {},
        "critic_state_dict": {},
        "optimizer_state_dict": {},
        "critic_optimizer_state_dict": {},
        "iter": 100,
    }
    torch.save(checkpoint, path)

    assert checkpoint_valid(path, 100)
    assert not checkpoint_valid(path, 50)

    record = {
        "iteration": 100,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    assert checkpoint_record_valid(record)
    record["sha256"] = "0" * 64
    assert not checkpoint_record_valid(record)


@pytest.mark.parametrize(
    "key",
    [
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "critic_optimizer_state_dict",
    ],
)
def test_checkpoint_validation_rejects_nonfinite_tensor_trees(tmp_path, key):
    path = tmp_path / "model_100.pt"
    checkpoint = {
        "actor_state_dict": {"weight": torch.ones(1)},
        "critic_state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": {"state": [{"moment": torch.ones(1)}]},
        "critic_optimizer_state_dict": {"state": [{"moment": torch.ones(1)}]},
        "iter": 100,
    }
    if "optimizer" in key:
        checkpoint[key]["state"][0]["moment"][0] = torch.inf
    else:
        checkpoint[key]["weight"][0] = torch.inf
    torch.save(checkpoint, path)

    assert not checkpoint_valid(path, 100)


def test_training_audit_rejects_nonfinite_vitals_and_blocks_evaluation(
    tmp_path, monkeypatch
):
    case = {
        "stage": "train",
        "label": "cpu_off_seed7",
        "backend": "cpu",
        "bundle": "off",
        "seed": 7,
        "artifact": "training/cpu_off_seed7.json",
        "summary": "cell_summaries/train/cpu_off_seed7.json",
        "status": "complete",
        "attempts": [],
    }
    protocol = {
        "task": "task",
        "backends": [{"label": "cpu", "physics_backend": "mujoco", "device": "cpu"}],
        "train_num_envs": 1,
        "train_iterations": 100,
        "save_interval": 50,
        "checkpoints": [100],
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "vitals.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"iteration": 99, "reward": 1.0}),
                json.dumps({"iteration": 100, "reward": float("inf")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = run_dir / "model_100.pt"
    torch.save(
        {
            "actor_state_dict": {"weight": torch.ones(1)},
            "critic_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {},
            "critic_optimizer_state_dict": {},
            "iter": 100,
        },
        checkpoint_path,
    )
    identity = {name: case[name] for name in ("label", "backend", "bundle", "seed")}
    resolved_path = tmp_path / "training" / "cpu_off_seed7.resolved.json"
    resolved_path.parent.mkdir()
    resolved_path.write_text(
        json.dumps({**identity, "environment": {}, "training": {}}),
        encoding="utf-8",
    )
    marker_path = tmp_path / case["artifact"]
    marker_path.write_text(
        json.dumps(
            {
                **identity,
                "run_dir": str(run_dir),
                "resolved_config": str(resolved_path),
                "checkpoints": [
                    {
                        "iteration": 100,
                        "path": str(checkpoint_path),
                        "bytes": checkpoint_path.stat().st_size,
                        "sha256": sha256_file(checkpoint_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.run_full_domain_randomization_campaign.resolved_config",
        lambda *args: {"environment": {}, "training": {}},
    )
    monkeypatch.setattr(
        "scripts.run_full_domain_randomization_campaign.backend_by_label",
        lambda label: label,
    )

    assert not validate_training(marker_path, case, protocol)

    evaluation = {
        "stage": "eval",
        "label": "dependent_eval",
        "training_label": case["label"],
        "eval_backend": "cpu",
        "artifact": "evaluation/dependent_eval.npz",
        "summary": "cell_summaries/eval/dependent_eval.json",
        "status": "pending",
        "attempts": [],
    }
    manifest = {
        "protocol": protocol,
        "cells": {
            f"train:{case['label']}": case,
            "eval:dependent_eval": evaluation,
        },
    }
    campaign = Campaign(tmp_path, manifest)

    assert audit_completed_cells(campaign) == [f"train:{case['label']}"]
    assert case["status"] == "pending"
    assert not run_stage(campaign, "eval", cpu_workers=1)
    assert evaluation["status"] == "pending"
    assert not evaluation["attempts"]


def test_subprocess_threads_follow_evaluation_backend():
    cpu = subprocess_environment({"eval_backend": "cpu"})
    gpu = subprocess_environment({"eval_backend": "vsim"})

    assert cpu["OMP_NUM_THREADS"] == "4"
    assert gpu["OMP_NUM_THREADS"] == "2"


def test_evaluation_summary_omits_metric_without_finite_samples(tmp_path):
    path = tmp_path / "evaluation.npz"
    metadata = {
        "survival": {"direction": "higher", "unit": "fraction"},
        "clearance": {"direction": "higher", "unit": "m"},
    }
    np.savez(
        path,
        hardware_metric_names=np.asarray(["survival", "clearance"]),
        hardware_metric_metadata=np.asarray(json.dumps(metadata)),
        metric_survival=np.asarray([1.0, 0.0]),
        metric_clearance=np.asarray([np.nan, np.nan]),
        mean_reward=np.asarray([1.0, 2.0]),
    )
    cell = {
        "label": "cell",
        "training_label": "cpu_off_seed7",
        "train_backend": "cpu",
        "bundle": "off",
        "seed": 7,
        "checkpoint_iteration": 100,
        "eval_backend": "cpu",
        "domain": "nominal",
    }

    summary = summarize_evaluation(path, cell)

    assert summary["metrics"]["survival"]["count"] == 2
    assert "clearance" not in summary["metrics"]


def test_campaign_rejects_physics_capacity_overflow(tmp_path):
    log = tmp_path / "child.log"
    log.write_text("training completed\n", encoding="utf-8")
    reject_engine_overflow(log)

    log.write_text("nefc overflow - please increase njmax to 200\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="nefc overflow"):
        reject_engine_overflow(log)


def test_stop_request_interrupts_children_and_prevents_new_cells(tmp_path, monkeypatch):
    manifest = {
        "protocol": {},
        "cells": {
            "train:active": {
                "stage": "train",
                "label": "active",
                "status": "running",
                "attempts": [],
            },
            "train:queued": {
                "stage": "train",
                "label": "queued",
                "status": "pending",
                "attempts": [],
            },
        },
    }
    campaign = Campaign(tmp_path, manifest)
    campaign.processes["train:active"] = SimpleNamespace(pid=4321)
    signals = []
    monkeypatch.setattr(
        "scripts.run_full_domain_randomization_campaign.os.killpg",
        lambda pid, requested_signal: signals.append((pid, requested_signal)),
    )
    monkeypatch.setattr(
        campaign,
        "run_cell",
        lambda key: pytest.fail(f"scheduled {key} after stop"),
    )

    mark_campaign_stopped(campaign, "test stop")

    assert signals == [(4321, signal.SIGINT)]
    assert run_pool(campaign, ["train:queued"], workers=1) == []
    assert manifest["status"] == "incomplete"
    assert manifest["stop_reason"] == "test stop"
    assert manifest["cells"]["train:active"]["status"] == "pending"
    assert manifest["cells"]["train:queued"]["status"] == "pending"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["progress"]["train"] == {"pending": 2}


def _scheduler_campaign(monkeypatch, work):
    cells = {
        "train:cpu_slow": {"stage": "train", "label": "cpu_slow", "backend": "cpu"},
        "train:warp_first": {
            "stage": "train",
            "label": "warp_first",
            "backend": "warp",
        },
        "train:vsim_next": {"stage": "train", "label": "vsim_next", "backend": "vsim"},
    }
    for training, backends in (
        ("cpu_slow", ("cpu",)),
        ("warp_first", ("warp", "vsim", "cpu")),
        ("vsim_next", ("vsim",)),
    ):
        for backend in backends:
            label = f"{training}_{backend}"
            cells[f"eval:{label}"] = {
                "stage": "eval",
                "label": label,
                "training_label": training,
                "eval_backend": backend,
            }
    for cell in cells.values():
        cell["status"] = "pending"
    campaign = SimpleNamespace(manifest={"cells": cells}, stop_event=threading.Event())
    campaign.request_stop = campaign.stop_event.set

    def run_cell(key):
        cells[key]["status"] = "running"
        try:
            success = work(campaign, key)
        except Exception:
            cells[key]["status"] = "failed"
            raise
        cells[key]["status"] = "complete" if success else "failed"
        return success

    campaign.run_cell = run_cell
    monkeypatch.setattr(full_campaign, "write_combined_summary", lambda _: None)
    return campaign


def test_evaluations_start_before_slow_cpu_training_and_take_gpu_priority(monkeypatch):
    cpu_started = threading.Event()
    gpu_evaluation_started = threading.Event()
    lock = threading.Lock()
    started = []
    active_gpu = 0

    def work(campaign, key):
        nonlocal active_gpu
        cells = campaign.manifest["cells"]
        cell = cells[key]
        backend = cell["backend"] if cell["stage"] == "train" else cell["eval_backend"]
        gpu = backend != "cpu"
        with lock:
            started.append(key)
            active_gpu += int(gpu)
            assert active_gpu <= 1, "Warp and VSim must share one GPU slot"
        try:
            if cell["stage"] == "eval":
                assert cells[f"train:{cell['training_label']}"]["status"] == "complete"
            if key == "train:cpu_slow":
                cpu_started.set()
                assert gpu_evaluation_started.wait(5), (
                    "GPU eval waited for CPU training"
                )
            elif key == "train:warp_first":
                assert cpu_started.wait(5)
            elif key == "eval:warp_first_warp":
                assert cells["train:cpu_slow"]["status"] == "running"
                gpu_evaluation_started.set()
            elif key == "train:vsim_next":
                assert cells["eval:warp_first_warp"]["status"] == "complete"
                assert cells["eval:warp_first_vsim"]["status"] == "complete"
            return True
        finally:
            with lock:
                active_gpu -= int(gpu)

    campaign = _scheduler_campaign(monkeypatch, work)

    assert run_training_and_evaluation(campaign, cpu_workers=1)
    assert len(started) == len(campaign.manifest["cells"])
    assert len(started) == len(set(started))


@pytest.mark.parametrize("raise_error", (False, True))
def test_failed_training_does_not_release_evaluations(monkeypatch, raise_error):
    started = []

    def work(campaign, key):
        started.append(key)
        if key == "train:warp_first":
            if raise_error:
                raise RuntimeError("invalid training artifact")
            return False
        return True

    campaign = _scheduler_campaign(monkeypatch, work)
    # A saved complete flag must not bypass run_cell's artifact validation.
    campaign.manifest["cells"]["train:warp_first"]["status"] = "complete"

    assert not run_training_and_evaluation(campaign, cpu_workers=2)
    assert "eval:vsim_next_vsim" in started
    for key, cell in campaign.manifest["cells"].items():
        if cell.get("training_label") == "warp_first":
            assert key not in started
            assert cell["status"] == "pending"


def test_interleaved_scheduler_stops_without_launching_more_cells(monkeypatch):
    started = []

    def work(campaign, key):
        started.append(key)
        campaign.request_stop()
        return True

    campaign = _scheduler_campaign(monkeypatch, work)
    # Keep one GPU lane so the first completion deterministically requests stop
    # before any second job can be admitted.
    campaign.manifest["cells"] = {
        key: cell
        for key, cell in campaign.manifest["cells"].items()
        if cell.get("backend") != "cpu" and cell.get("training_label") != "cpu_slow"
    }

    assert not run_training_and_evaluation(campaign, cpu_workers=1)
    assert started == ["train:warp_first"]
    assert campaign.manifest["cells"]["eval:warp_first_warp"]["status"] == "pending"


def test_main_dispatches_interleaved_training_and_evaluation_once(monkeypatch):
    campaign = _scheduler_campaign(monkeypatch, lambda campaign, key: True)
    campaign.save = lambda: None
    args = full_campaign.get_args(
        ["--evaluate-after-training", "--skip-speed", "--stages", "train", "eval"]
    )
    calls = []

    def run_interleaved(actual_campaign, workers):
        calls.append((actual_campaign, workers))
        return run_training_and_evaluation(actual_campaign, workers)

    monkeypatch.setattr(full_campaign, "get_args", lambda _: args)
    monkeypatch.setattr(
        full_campaign, "load_or_create_manifest", lambda *args: campaign.manifest
    )
    monkeypatch.setattr(full_campaign, "Campaign", lambda *args: campaign)
    monkeypatch.setattr(full_campaign, "run_training_and_evaluation", run_interleaved)
    monkeypatch.setattr(
        full_campaign,
        "run_stage",
        lambda *args: pytest.fail("interleaved stages must not also run separately"),
    )

    full_campaign.main()

    assert calls == [(campaign, args.cpu_workers)]
    assert campaign.manifest["status"] == "complete"
