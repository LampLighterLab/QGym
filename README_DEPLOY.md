# Deploying a custom policy

## Setup

To deploy a custom policy, you will need to get `unitree-sdk2py` and `cyclonedds`. First make sure the QGym virtual environment is correctly set up.

Install `cyclonedds` version 0.10.2 [adapted from these instructions](https://pypi.org/project/cyclonedds/):
- `git clone --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git`
- `cd cyclonedds && mkdir build install && cd build`
- `cmake .. -DCMAKE_INSTALL_PREFIX=../install`
- `cmake --build . --config RelWithDebInfo --target install`
- `cd ..`
- `export CYCLONEDDS_HOME=<path to your cyclonedds directory>/cyclonedds/install`
- Activate QGym's venv (in `QGym/`: `source ./venv/bin/activate`)
- `CC=clang uv pip install cyclonedds==0.10.2`

Install `unitree-sdk2py`:
[Clone the Unitree SDK2 Python repo and follow its instructions](https://github.com/unitreerobotics/unitree_sdk2_python):
- `git clone https://github.com/unitreerobotics/unitree_sdk2_python.git`
- `cd unitree_sdk2_python`
- `uv pip install -e .`
- If you get the error `Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH`:
    - `export CYCLONEDDS_HOME=<path to your cyclonedds directory>/cyclonedds/install`
    - `uv pip install -e .`

## Configure deployment

Settings can be configured in `go2_deploy/deploy_config.py`. Options include:
- Control frequency
- kp (Stiffness constant)
- kd (Damping constant)
- Observation vector: a list of available observations can be found in `obs_sizes`
- Command limits (ex. pushing the joystick all the way to the right commands x m/s)

## Deployment

If anything goes wrong, the emergency stop on the wireless controller is X or the enter key in the terminal.

- Make sure your trained policy is in `logs/`, and that the observation vector, scaling, and other settings in `deploy_config.py` are all correct.
- Connect the Go2 via Ethernet cable.
- Turn on the Go2 and wait until it is in the standing position.
- In the terminal, run `uv run go2_deploy/deploy.py eth0`, but replace `eth0` with the actual network configuration. If this is successful, there should be a terminal output every few seconds telling you about the keyboard controls, observation frequency, etc.
- There are 4 states the robot can be in:
    - `RECOVERY`: Initially when the script starts running, the robot is in this mode. You must enter this state from `EMERGENCY_STOP` otherwise. The default Unitree controller can be used to make the robot walk, do tricks, etc. The `F1` key on the remote control can be used to disable/enable the A, B, X, and Y buttons, so that the default Unitree button combinations can be used without accidentally turning on a different state.
    - `EMERGENCY_STOP`: The robot will immediately start damping, wait around 15 seconds, and then stand up again. The robot will automatically enter the `RECOVERY` state.
    - `INTERMEDIATE`: The robot will hold its current position. This is for testing to make sure the kp and kd values are good, etc.
    - `CUSTOM_CTRL`: The robot will use the policy you trained to move.
- To stop running the script, return to `RECOVERY`, lie the robot on the ground, and press `Ctrl+C` in the terminal.
- Logs of the robot data can be found in `logs/deploy/`.

## Remote controls

- Switch modes
    - Enter `EMERGENCY_STOP`: `X`
    - Enter `INTERMEDIATE`: `B`
    - Enter `CUSTOM_CTRL`: `A`
- Disable/enable ABXY buttons: `F1`
- Increase kp by 10%: `Up` (below the left joystick)
- Decrease kp by 10%: `Down`
- Increase kd by 10%: `Right`
- Decrease kd by 10%: `Left`

## Keyboard controls
Type the following letters in the terminal:

- Switch modes
    - Enter `EMERGENCY_STOP`: `<enter key>`
    - Enter `INTERMEDIATE`: `q`
    - Enter `CUSTOM_CTRL`: `c`
- Increase kp by 10%: `i`
- Decrease kp by 10%: `k`
- Increase kd by 10%: `l`
- Decrease kd by 10%: `j`

## Something went wrong

- `deploy.py` is printing errors in the terminal
    - Make sure the robot is either lying down, or standing up while in sport mode, and exit the script in the terminal.
- The robot is shaking violently when running a custom policy
    - In `deploy_config.py`, make sure that your observation vector is correct and the values in the `DeployScaling` class match what you used during training. Also make sure you are running the correct policy from `logs/`.