from gym.envs.go2.go2_config import (
    Go2Cfg,
    Go2RunnerCfg,
    GO2_LEG_GROUPS,
)

BASE_HEIGHT_REF = 0.4


class Go2RefCfg(Go2Cfg):
    class init_state(Go2Cfg.init_state):
        ref_traj = (
            "{GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/trajectories/single_leg.csv"
        )
        reset_mode = "reset_to_basic"
        default_joint_angles = {
            "hip": 0.00,
            "thigh": 0.66,
            "calf": -1.34,
        }

        dof_pos_range = {
            "hip": [-0.01, 0.01],
            "thigh": [0.65, 0.67],
            "calf": [-1.37, -1.35],
        }  # broken?

    class reward_settings(Go2Cfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25
        switch_scale = 0.1

    class asset(Go2Cfg.asset):
        disable_gravity = False
        disable_motors = False
        fix_base_link = False

    class control(Go2Cfg.control):
        # * PD Drive parameters:
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 500

        gait_freq = 2.0
        reference_leg_groups = GO2_LEG_GROUPS
        gait_phase_offsets = {
            "FL_leg": 0.0,
            "FR_leg": 0.5,
            "RL_leg": 0.5,
            "RR_leg": 0.0,
        }

    class scaling(Go2Cfg.scaling):
        base_ang_vel = 0.3
        base_lin_vel = BASE_HEIGHT_REF
        dof_vel = 4 * [2.0, 2.0, 4.0]
        base_height = 0.3 / 2
        dof_pos = 4 * [0.2, 0.3, 0.3]
        dof_pos_obs = dof_pos
        dof_pos_target = 4 * [0.2, 0.3, 0.3]
        tau_ff = 4 * [18, 18, 28]
        commands = [3, 1, 3]


class Go2RefRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_ang_vel",
            "phase_obs",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_pos_obs_residual",
            "dof_vel",
            "dof_pos_target",
        ]
        normalize_obs = False
        actions = ["dof_pos_target"]
        add_noise = True
        disable_actions = False

        class noise:
            scale = 1.0
            dof_pos_obs = 0.01
            base_ang_vel = 0.01
            dof_pos = 0.005
            dof_vel = 0.005
            lin_vel = 0.05
            ang_vel = [0.3, 0.15, 0.4]
            gravity_vec = 0.1

    class critic(Go2RunnerCfg.critic):
        hidden_dims = [128, 64]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.01
                orientation = 1.0
                torques = 5.0e-6
                dof_vel = 0.0
                min_base_height = 1.5
                action_rate = 0.1
                action_rate2 = 0.01
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0
                swing_grf = 1.0
                stance_grf = 1.0

            class termination_weight:
                termination = 0.01

    class algorithm(Go2RunnerCfg.algorithm):
        pass

    class runner(Go2RunnerCfg.runner):
        run_name = ""
        experiment_name = "go2"
        max_iterations = 500
        algorithm_class_name = "PPO2"
