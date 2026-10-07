import torch

from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotCfgPPO


class PendulumCfg(FixedRobotCfg):
    class env(FixedRobotCfg.env):
        num_envs = 256
        num_actuators = 1
        episode_length_s = 10

    class terrain(FixedRobotCfg.terrain):
        pass

    class viewer:
        ref_env = 0
        pos = [10.0, 5.0, 10.0]  # [m]
        lookat = [0.0, 0.0, 0.0]  # [m]

    class init_state(FixedRobotCfg.init_state):
        default_joint_angles = {"theta": 0}  # -torch.pi / 2.0}

        # * default setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * initial conditions for reset_to_range
        dof_pos_range = {
            "theta": [-torch.pi, torch.pi],
        }
        dof_vel_range = {"theta": [-5, 5]}

    class control(FixedRobotCfg.control):
        actuated_joint_names = ["theta"]
        ctrl_frequency = 25
        desired_sim_frequency = 50
        stiffness = {"theta": 0.0}  # [N*m/rad]
        damping = {"theta": 0.0}  # [N*m*s/rad]

    class asset(FixedRobotCfg.asset):
        # * Things that differ
        file = "{GYM_ROOT_DIR}/resources/robots/" + "pendulum/urdf/pendulum.urdf"
        disable_gravity = False
        disable_motors = False  # all torques set to 0
        joint_damping = 0.1
        mass = 1.0
        length = 1.0

    class reward_settings(FixedRobotCfg.reward_settings):
        tracking_sigma = 0.25

    class scaling(FixedRobotCfg.scaling):
        dof_pos_obs = 1.0  # sin(theta), cos(theta)
        dof_vel = 5.0
        dof_pos = 2.0 * torch.pi
        # * Action scales
        tau_ff = 5.0  # one policy unit spans the URDF torque limit


class PendulumRunnerCfg(FixedRobotCfgPPO):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(FixedRobotCfgPPO.actor):
        frequency = 25
        hidden_dims = [128, 64, 32]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "tanh"
        # 2.5 Nm initial exploration: about 5% of samples reach the torque limit.
        init_noise_std = 0.5

        normalize_obs = False
        obs = [
            "dof_pos_obs",
            "dof_vel",
        ]

        actions = ["tau_ff"]
        disable_actions = False

        class noise:
            dof_pos_obs = 0.0
            dof_vel = 0.0

    class critic:
        hidden_dims = [128, 64, 32]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "tanh"

        normalize_obs = False
        obs = [
            "dof_pos_obs",
            "dof_vel",
        ]

        class reward:
            class weights:
                theta = 0.1
                omega = 0.1
                equilibrium = 1.0
                energy = 0.5
                # Swing-up needs about 6 rad/s at the bottom; a large quadratic
                # velocity penalty instead teaches the policy to stop there.
                dof_vel = 0.001
                torques = 0.025

            class termination_weight:
                termination = 0.0

    class algorithm(FixedRobotCfgPPO.algorithm):
        gamma = 0.95
        lam = 0.98
        max_gradient_steps = 24
        # 16 control steps per environment (0.64 s), with a full-rollout batch.
        rollout_size = 4096
        batch_size = 4096
        clip_param = 0.2
        learning_rate = 1.0e-4
        lr_range = [1.0e-4, 1.0e-3]
        max_grad_norm = 1.0
        # Critic
        use_clipped_value_loss = True
        # Actor
        entropy_coef = 0.01
        schedule = "adaptive"  # could be adaptive, fixed
        desired_kl = 0.01

    class runner(FixedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pendulum"
        max_iterations = 200  # number of policy updates
        algorithm_class_name = "PPO2"
