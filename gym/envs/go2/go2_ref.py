from gym.envs.go2.go2 import Go2


class Go2Ref(Go2):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)
