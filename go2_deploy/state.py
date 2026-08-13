from enum import Enum, auto


class State(Enum):
    # Robot is in low state mode, lowcmd_thread is running
    # Can only be accessed from INTERMEDIATE
    CUSTOM_CTRL = auto()

    # After emergency_stop() finishes processing, the robot is in low state mode
    # a new lowcmd_thread is created + not started, self.emergency_lowcmd_thread = None
    # Can be accessed from any other state, triggered automatically in
    # dangerous conditions
    EMERGENCY_STOP = auto()

    # After recovering, the robot is in sport mode, is standing up,
    # lowcmd_thread is not started
    # Can be accessed from any other state
    RECOVERY = auto()

    # Record current position and hold it in lowcmd mode
    # lowcmd_thread is running
    INTERMEDIATE = auto()
