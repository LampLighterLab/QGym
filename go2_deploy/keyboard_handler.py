from go2_deploy.utility.thread import RecurrentThread


class KeyboardHandler:
    def __init__(self, controller):
        self.controller = controller
        self.keyboard_thread = RecurrentThread(
            interval=0.005, target=self._process_input
        )
        self.keyboard_thread.Start()

    def _process_input(self):
        key = input()
        if key == "":
            self.controller._estop_flag = True
        elif key == "r":
            self.controller.switch_to_recovery()
        elif key == "q":
            self.controller.switch_to_intermediate()
        elif key == "c":
            self.controller.switch_to_custom_controller()
        elif key == "d":
            self.controller.switch_to_default_controller()
        elif key == "i":
            self.controller.kp_mult += 0.1
            print(f"kp increased to {self.controller.kp_mult * self.controller.cfg.kp}")
        elif key == "k":
            if self.controller.kp_mult >= 0.1:
                self.controller.kp_mult -= 0.1
            print(f"kp decreased to {self.controller.kp_mult * self.controller.cfg.kp}")
        else:
            print("Invalid keyboard input!")
