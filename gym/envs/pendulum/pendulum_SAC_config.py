from gym.envs.pendulum.pendulum_config import PendulumCfg, PendulumRunnerCfg


class PendulumSACCfg(PendulumCfg):
    """Use the same physical task, resets, and scales as PPO2."""

    class env(PendulumCfg.env):
        num_envs = 16


class PendulumSACRunnerCfg(PendulumRunnerCfg):
    runner_class_name = "OffPolicyRunner"

    class actor(PendulumRunnerCfg.actor):
        # ChimeraActor has a shared trunk and separate mean/log-std heads.
        # create_MLP takes `activations` (plural) inside these dictionaries.
        latent_nn = {"hidden_dims": [256, 256], "activations": "relu"}
        mean_nn = {"hidden_dims": []}
        std_nn = {"hidden_dims": []}
        nn_params = {"latent": latent_nn, "mean": mean_nn, "std": std_nn}
        std_init = 1.0
        log_std_max = 2.0
        log_std_min = -20.0

    class critic(PendulumRunnerCfg.critic):
        # Observations and rewards remain shared with PPO2.
        hidden_dims = [256, 256]
        activation = "relu"

    class algorithm:
        # SAC reference settings: RL Zoo Pendulum and SB3/Spinning Up.
        # One optimizer step per new transition, using small replay batches.
        initial_fill = 64  # 1024 uniform-action transitions at 16 environments
        storage_size = 1_000_000
        batch_size = 256
        max_gradient_steps = 16
        gamma = 0.99
        action_max = 1.0
        action_min = -1.0
        # Temperature is in our dt-integrated reward units, with auto-tuning.
        alpha = 0.01
        target_entropy = -1.0
        polyak = 0.995
        alpha_lr = 1e-3
        actor_lr = 1e-3
        critic_lr = 1e-3

    class runner(PendulumRunnerCfg.runner):
        experiment_name = "sac_pendulum"
        algorithm_class_name = "SAC"
        max_iterations = 1500  # 24,000 transitions, plus 1024 warmup transitions
        save_interval = 500
        num_steps_per_env = 1
