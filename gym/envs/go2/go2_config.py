from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

BASE_HEIGHT_REF = 0.4

GO2_DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

GO2_BODY_NAMES = [
    "base",
    "Head_upper",
    "Head_lower",
    "FL_hip",
    "FL_hip_rotor",
    "FL_thigh",
    "FL_thigh_rotor",
    "FL_calf",
    "FL_calflower",
    "FL_calflower1",
    "FL_calf_rotor",
    "FL_foot",
    "FR_hip",
    "FR_hip_rotor",
    "FR_thigh",
    "FR_thigh_rotor",
    "FR_calf",
    "FR_calflower",
    "FR_calflower1",
    "FR_calf_rotor",
    "FR_foot",
    "RL_hip",
    "RL_hip_rotor",
    "RL_thigh",
    "RL_thigh_rotor",
    "RL_calf",
    "RL_calflower",
    "RL_calflower1",
    "RL_calf_rotor",
    "RL_foot",
    "RR_hip",
    "RR_hip_rotor",
    "RR_thigh",
    "RR_thigh_rotor",
    "RR_calf",
    "RR_calflower",
    "RR_calflower1",
    "RR_calf_rotor",
    "RR_foot",
    "imu",
    "radar",
    "front_camera",
]

GO2_LEG_GROUPS = ["FL_leg", "FR_leg", "RL_leg", "RR_leg"]


class Go2Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actuators = 12
        episode_length_s = 3

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            "hip": 0.0,
            "thigh": 0.66,
            "calf": -1.36,
        }

        # * reset setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * default COM for basic initialization
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup
        dof_pos_range = {
            "hip": [-0.01, 0.01],
            "thigh": [0.65, 0.67],
            "calf": [-1.37, -1.35],
        }
        dof_vel_range = {"hip": [0.0, 0.0], "thigh": [0.0, 0.0], "calf": [0.0, 0.0]}
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.40, 0.40],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]
        root_vel_range = [
            [-0.5, 2.0],  # x
            [0.0, 0.0],  # y
            [-0.05, 0.05],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(LeggedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 500

    class commands:
        # * time before command are changed[s]
        resampling_time = 3.0

        class ranges:
            lin_vel_x = [-2.0, 3.0]  # min max [m/s]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 3  # max [rad/s]

    class push_robots:
        toggle = False
        interval_s = 1
        max_push_vel_xy = 0.5
        push_box_dims = [0.3, 0.1, 0.1]  # x,y,z [m]

    class asset(LeggedRobotCfg.asset):
        file = "{GYM_ROOT_DIR}/resources/robots/" + "go2/urdf/go2.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base"]
        end_effector_names = ["foot"]
        collapse_fixed_joints = False
        self_collisions = 1
        flip_visual_attachments = False
        disable_gravity = False
        disable_motors = False
        joint_damping = 0.01
        rotor_inertia = [0.002268, 0.002268, 0.005484] * 4

        class robot_layout:
            version = "go2_v1"
            dof_names = GO2_DOF_NAMES
            actuated_dof_names = GO2_DOF_NAMES
            body_names = GO2_BODY_NAMES
            dof_groups = {
                "FL_leg": GO2_DOF_NAMES[0:3],
                "FR_leg": GO2_DOF_NAMES[3:6],
                "RL_leg": GO2_DOF_NAMES[6:9],
                "RR_leg": GO2_DOF_NAMES[9:12],
                "abad": GO2_DOF_NAMES[0:12:3],
            }
            body_groups = {
                "feet": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            }

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(LeggedRobotCfg.scaling):
        base_ang_vel = 1.0
        base_lin_vel = BASE_HEIGHT_REF
        dof_vel = 4 * [5.0, 5.0, 8.0]
        base_height = 0.3 / 2
        dof_pos = 4 * [0.5, 0.5, 0.5]
        dof_pos_obs = dof_pos
        dof_pos_obs_residual = 4 * [0.25, 0.25, 0.25]
        dof_pos_target = 4 * [0.5, 0.5, 1.0]
        tau_ff = 4 * [18, 18, 28]
        commands = [3, 1, 3]

    class mjspec_attributes:
        njmax = 130

    class mjspec_option_attributes:
        ccd_iterations = 50

    class vsim_attributes:
        solver_iterations = 16


class Go2RunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(LeggedRobotRunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
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

    class critic(LeggedRobotRunnerCfg.critic):
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
                tracking_ang_vel = 2.0 * 2
                lin_vel_z = 0.0
                ang_vel_xy = 0.01
                orientation = 1.0 * 0.7
                torques = 5.0e-6 * 50
                dof_vel = 0.0
                min_base_height = 1.5 * 0.8
                action_rate = 0.1
                action_rate2 = 0.01 * 50
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0 + 0.07
                dof_near_home = 0.0 + 0.05

            class termination_weight:
                termination = 0.01

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        pass

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ""
        experiment_name = "go2"
        max_iterations = 500
        algorithm_class_name = "PPO2"
