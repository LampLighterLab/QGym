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
            self.controller.request_emergency_stop()
        elif key == "r":
            self.controller.switch_to_recovery()
        elif key == "q":
            self.controller.switch_to_intermediate()
        elif key == "c":
            self.controller.switch_to_custom_controller()
        elif key == "i":
            self.controller.kp_mult += 0.1
            print(f"kp increased to {self.controller.kp_mult * self.controller.cfg.kp}")
        elif key == "k":
            if self.controller.kp_mult >= 0.1:
                self.controller.kp_mult -= 0.1
            print(f"kp decreased to {self.controller.kp_mult * self.controller.cfg.kp}")
        elif key == "l":
            self.controller.kd_mult += 0.1
            print(f"kd increased to {self.controller.kd_mult * self.controller.cfg.kd}")
        elif key == "j":
            if self.controller.kd_mult >= 0.1:
                self.controller.kd_mult -= 0.1
            print(f"kd decreased to {self.controller.kd_mult * self.controller.cfg.kd}")
        else:
            print("Invalid keyboard input!")
