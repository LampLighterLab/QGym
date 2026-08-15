"""Backend-neutral sampling for physical and control domain parameters.

The task layer owns random sampling. Physics backends only apply explicit
values, which isolates DR from task RNG and keeps parameter semantics
independent of the selected engine.
"""

import zlib

import torch


DOMAIN_RANDOMIZATION_MODES = (
    "config",
    "off",
    "friction-only",
    "pd-only",
    "mass-only",
)

STARTUP_AXES = (
    "contact_friction_range",
    "link_mass_scale_range",
)
PD_GAIN_TARGETS = ("p_gains", "d_gains")


class DomainRandomizationCfg:
    class startup:
        # Sampled once for every environment during task construction.
        contact_friction_range = None
        # Mass and diagonal inertia are scaled together for each robot link.
        link_mass_scale_range = None

    class episode:
        # Multiplicative ranges for task-owned tensors that are resampled at
        # every episode reset. The task explicitly binds the allowed tensors.
        scale_ranges = {}


def get_domain_randomization_range(cfg, name: str):
    """Return a startup-axis or episodic tensor scale range."""
    if name == "contact_friction_range":
        return cfg.domain_randomization.startup.contact_friction_range
    if name == "link_mass_scale_range":
        return cfg.domain_randomization.startup.link_mass_scale_range
    return cfg.domain_randomization.episode.scale_ranges.get(name)


def set_domain_randomization_range(cfg, name: str, values) -> None:
    """Set a startup-axis or episodic tensor scale range."""
    if name == "contact_friction_range":
        cfg.domain_randomization.startup.contact_friction_range = values
        return
    if name == "link_mass_scale_range":
        cfg.domain_randomization.startup.link_mass_scale_range = values
        return
    ranges = dict(cfg.domain_randomization.episode.scale_ranges)
    if values is None:
        ranges.pop(name, None)
    else:
        ranges[name] = values
    cfg.domain_randomization.episode.scale_ranges = ranges


def apply_domain_randomization_override(cfg, mode: str) -> None:
    """Select a deliberate subset of the configured randomization axes."""
    if mode == "config":
        return
    if mode not in DOMAIN_RANDOMIZATION_MODES:
        raise ValueError(f"unknown domain-randomization mode {mode!r}")

    keep_startup = {
        "off": set(),
        "friction-only": {"contact_friction_range"},
        "pd-only": set(),
        "mass-only": {"link_mass_scale_range"},
    }[mode]
    keep_episode = set(PD_GAIN_TARGETS) if mode == "pd-only" else set()
    configured_startup = {
        name: get_domain_randomization_range(cfg, name) for name in STARTUP_AXES
    }
    configured_episode = dict(cfg.domain_randomization.episode.scale_ranges)
    missing = sorted(
        name
        for name in (*keep_startup, *keep_episode)
        if (
            configured_startup.get(name) is None
            and configured_episode.get(name) is None
        )
    )
    if missing:
        raise ValueError(
            f"{mode} domain randomization requires configured ranges for {missing}"
        )
    for name in configured_startup:
        if name not in keep_startup:
            set_domain_randomization_range(cfg, name, None)
    cfg.domain_randomization.episode.scale_ranges = {
        name: values
        for name, values in configured_episode.items()
        if name in keep_episode
    }


def _configured_range(cfg, name: str) -> tuple[float, float] | None:
    values = get_domain_randomization_range(cfg, name)
    if values is None:
        return None
    low, high = map(float, values)
    return low, high


def contact_friction_range(cfg) -> tuple[float, float] | None:
    """Return and validate the configured effective sliding-friction range.

    ``None`` disables contact-friction randomization. The portable coefficient
    maps to MuJoCo sliding friction and to both VSim static and dynamic
    friction.
    """
    values = _configured_range(cfg, "contact_friction_range")
    if values is None:
        return None
    low, high = values
    if not 0.0 <= low <= high:
        raise ValueError(
            f"contact-friction range must satisfy 0 <= low <= high, got [{low}, {high}]"
        )
    return low, high


def scale_range(cfg, name: str) -> tuple[float, float] | None:
    """Return a positive multiplicative range containing the nominal scale."""
    values = _configured_range(cfg, name)
    if values is None:
        return None
    low, high = values
    if not 0.0 < low <= 1.0 <= high:
        raise ValueError(
            f"{name.replace('_', '-')} must satisfy 0 < low <= 1 <= high, "
            f"got [{low}, {high}]"
        )
    return low, high


def link_mass_scale_range(cfg) -> tuple[float, float] | None:
    """Return the multiplicative link mass-and-inertia scale range."""
    return scale_range(cfg, "link_mass_scale_range")


class DomainRandomizer:
    """Sample enabled domain parameters and apply them through ``SimBackend``."""

    def __init__(self, cfg, backend, device: str) -> None:
        self._backend = backend
        self._device = device
        self._contact_friction_range = contact_friction_range(cfg)
        self._link_mass_scale_range = link_mass_scale_range(cfg)
        self._episode_scale_ranges = {}
        for name in cfg.domain_randomization.episode.scale_ranges:
            values = scale_range(cfg, name)
            if values is None:
                raise ValueError(
                    f"episodic DR target {name!r} has no range; omit it to disable it"
                )
            self._episode_scale_ranges[name] = values

        ranges = {
            "contact_friction": self._contact_friction_range,
            "link_mass": self._link_mass_scale_range,
        }
        self._generators = {}
        if any(values is not None for values in ranges.values()) or any(
            values is not None for values in self._episode_scale_ranges.values()
        ):
            seed = cfg.seed
            if seed < 0:
                raise ValueError(
                    "enabled domain randomization requires a resolved "
                    "non-negative cfg.seed"
                )
            for offset, (name, values) in enumerate(ranges.items()):
                if values is not None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed + offset)
                    self._generators[name] = generator
            for name, values in self._episode_scale_ranges.items():
                if values is not None:
                    generator = torch.Generator(device=device)
                    offset = zlib.crc32(f"episode:{name}".encode())
                    generator.manual_seed(seed + offset)
                    self._generators[name] = generator

        self.link_mass_scale = None
        self._episode_targets = {}
        self._episode_nominals = {}
        self._episode_scales = {}

    def bind_link_masses(self) -> None:
        """Allocate the persistent canonical scale tensor after backend setup."""
        if self._link_mass_scale_range is not None:
            self.link_mass_scale = torch.ones_like(self._backend.link_mass)

    def bind_episode_targets(self, targets: dict[str, torch.Tensor]) -> None:
        """Bind allowed task tensors and retain nominals for configured targets."""
        missing = self._episode_scale_ranges.keys() - targets.keys()
        if missing:
            raise ValueError(f"unbound episodic DR targets: {sorted(missing)}")
        num_envs = self._backend.contact_friction.shape[0]
        for name in self._episode_scale_ranges:
            target = targets[name]
            if not torch.is_floating_point(target):
                raise TypeError(f"episodic DR target {name!r} must be floating point")
            if target.ndim == 0 or target.shape[0] != num_envs:
                raise ValueError(
                    f"episodic DR target {name!r} must start with [{num_envs}], "
                    f"got {list(target.shape)}"
                )
            if target.device != torch.device(self._device):
                raise ValueError(
                    f"episodic DR target {name!r} is on {target.device}, "
                    f"expected {self._device}"
                )
            self._episode_targets[name] = target
            self._episode_nominals[name] = target.clone()
            self._episode_scales[name] = torch.ones_like(target)

    def episode_scale(self, name: str) -> torch.Tensor | None:
        """Return the current scale tensor, or ``None`` when not configured."""
        return self._episode_scales.get(name)

    def set_episode_scale(
        self,
        name: str,
        env_ids: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        """Apply explicit scales relative to the retained nominal tensor."""
        if name not in self._episode_targets:
            raise ValueError(f"episodic DR target {name!r} is not configured and bound")
        env_ids = self._env_ids(env_ids)
        target = self._episode_targets[name]
        scales = torch.as_tensor(scales, dtype=target.dtype, device=target.device)
        expected_shape = (env_ids.numel(), *target.shape[1:])
        if scales.shape != expected_shape:
            raise ValueError(
                f"episodic DR scale for {name!r} must have shape "
                f"{expected_shape}, got {tuple(scales.shape)}"
            )
        self._episode_scales[name][env_ids] = scales
        target[env_ids] = self._episode_nominals[name][env_ids] * scales

    def _sample(
        self,
        values: tuple[float, float],
        shape: tuple[int, ...],
        generator: torch.Generator,
    ) -> torch.Tensor:
        low, high = values
        return low + (high - low) * torch.rand(
            shape,
            generator=generator,
            device=self._device,
        )

    @property
    def contact_friction(self) -> torch.Tensor:
        """Current applied coefficient for every environment."""
        return self._backend.contact_friction

    def _env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(env_ids, dtype=torch.long, device=self._device).flatten()

    def randomize_startup(self, env_ids: torch.Tensor) -> None:
        """Sample physical parameters once, before the first episode reset."""
        env_ids = self._env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        if self._contact_friction_range is not None:
            values = self._sample(
                self._contact_friction_range,
                (env_ids.numel(),),
                self._generators["contact_friction"],
            )
            self._backend.set_contact_friction(env_ids, values)

        if self._link_mass_scale_range is not None:
            if self.link_mass_scale is None:
                raise RuntimeError("link-mass DR requires bound backend masses")
            scales = self._sample(
                self._link_mass_scale_range,
                (env_ids.numel(), self.link_mass_scale.shape[1]),
                self._generators["link_mass"],
            )
            self.link_mass_scale[env_ids] = scales
            self._backend.set_link_mass_scale(env_ids, scales)

    def randomize_episode(self, env_ids: torch.Tensor) -> None:
        """Resample configured task tensors for resetting environments."""
        env_ids = self._env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        for name, values in self._episode_scale_ranges.items():
            if name not in self._episode_targets:
                raise RuntimeError(f"episodic DR target {name!r} is not bound")
            target = self._episode_targets[name]
            scales = self._sample(
                values,
                (env_ids.numel(), *target.shape[1:]),
                self._generators[name],
            )
            self.set_episode_scale(name, env_ids, scales)
