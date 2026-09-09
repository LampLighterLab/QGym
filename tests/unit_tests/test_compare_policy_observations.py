import numpy as np

from scripts.compare_policy_observations import (
    component_statistics,
    field_statistics,
    joint_first_episode_mask,
)


def test_joint_first_episode_mask_requires_both_runs_to_remain_alive():
    reference = {"terminated": np.array([[False], [False], [True], [False]])}
    candidate = {"terminated": np.array([[False], [True], [False], [False]])}

    mask = joint_first_episode_mask(reference, candidate)

    np.testing.assert_array_equal(mask[:, 0], [True, False, False, False])


def test_component_statistics_reports_identical_distributions_as_zero_gap():
    values = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    valid = np.ones((3, 2), dtype=bool)

    rows = component_statistics(values, values, ["state.a", "state.b"], [2, 3], valid)

    assert [row["quantile_gap_pooled_std"] for row in rows] == [0.0, 0.0]
    assert [row["paired_rmse"] for row in rows] == [0.0, 0.0]


def test_field_statistics_aggregates_named_components():
    components = [
        {
            "name": "state.a",
            "quantile_gap_pooled_std": 0.25,
            "initial_max_abs_difference": 0.1,
        },
        {
            "name": "state.b",
            "quantile_gap_pooled_std": 0.75,
            "initial_max_abs_difference": 0.2,
        },
    ]

    rows = field_statistics(["state"], components)

    assert rows == [
        {
            "field": "state",
            "components": 2,
            "max_quantile_gap_pooled_std": 0.75,
            "mean_quantile_gap_pooled_std": 0.5,
            "max_initial_abs_difference": 0.2,
        }
    ]
