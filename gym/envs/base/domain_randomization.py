"""Backend-neutral sampling for physical and control domain parameters.

The task layer owns random sampling. Physics backends only apply explicit
values, which isolates DR from task RNG and keeps parameter semantics
independent of the selected engine.
"""

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
EPISODE_AXES = (
    "stiffness_scale_range",
    "damping_scale_range",
)
AXIS_GROUPS = {
    **dict.fromkeys(STARTUP_AXES, "startup"),
    **dict.fromkeys(EPISODE_AXES, "episode"),
}


class DomainRandomizationCfg:
    class startup:
        # Sampled once for every environment during task construction.
        contact_friction_range = None
        # Mass and diagonal inertia are scaled together for each robot link.
        link_mass_scale_range = None

    class episode:
        # Resampled independently for environments at each episode reset.
        stiffness_scale_range = None
        damping_scale_range = None


def get_domain_randomization_range(cfg, name: str):
    """Return one axis range from its declared sampling-cadence group."""
    if name == "contact_friction_range":
        return cfg.domain_randomization.startup.contact_friction_range
    if name == "link_mass_scale_range":
        return cfg.domain_randomization.startup.link_mass_scale_range
    if name == "stiffness_scale_range":
        return cfg.domain_randomization.episode.stiffness_scale_range
    if name == "damping_scale_range":
        return cfg.domain_randomization.episode.damping_scale_range
    raise ValueError(f"unknown domain-randomization axis {name!r}")


def set_domain_randomization_range(cfg, name: str, values) -> None:
    """Set one axis range without duplicating cadence knowledge at call sites."""
    if name == "contact_friction_range":
        cfg.domain_randomization.startup.contact_friction_range = values
        return
    if name == "link_mass_scale_range":
        cfg.domain_randomization.startup.link_mass_scale_range = values
        return
    if name == "stiffness_scale_range":
        cfg.domain_randomization.episode.stiffness_scale_range = values
        return
    if name == "damping_scale_range":
        cfg.domain_randomization.episode.damping_scale_range = values
        return
    raise ValueError(f"unknown domain-randomization axis {name!r}")


def apply_domain_randomization_override(cfg, mode: str) -> None:
    """Select a deliberate subset of the configured randomization axes."""
    if mode == "config":
        return
    if mode not in DOMAIN_RANDOMIZATION_MODES:
        raise ValueError(f"unknown domain-randomization mode {mode!r}")

    keep = {
        "off": set(),
        "friction-only": {"contact_friction_range"},
        "pd-only": {"stiffness_scale_range", "damping_scale_range"},
        "mass-only": {"link_mass_scale_range"},
    }[mode]
    configured = {
        name: get_domain_randomization_range(cfg, name)
        for name in (*STARTUP_AXES, *EPISODE_AXES)
    }
    missing = sorted(name for name in keep if configured[name] is None)
    if missing:
        raise ValueError(
            f"{mode} domain randomization requires configured ranges for {missing}"
        )
    for name in configured:
        if name not in keep:
            set_domain_randomization_range(cfg, name, None)


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
        self._stiffness_scale_range = scale_range(cfg, "stiffness_scale_range")
        self._damping_scale_range = scale_range(cfg, "damping_scale_range")
        self._link_mass_scale_range = link_mass_scale_range(cfg)

        ranges = {
            "contact_friction": self._contact_friction_range,
            "stiffness": self._stiffness_scale_range,
            "damping": self._damping_scale_range,
            "link_mass": self._link_mass_scale_range,
        }
        self._generators = {}
        if any(values is not None for values in ranges.values()):
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

        self.stiffness_scale = None
        self.damping_scale = None
        self.link_mass_scale = None
        self._p_gains = None
        self._d_gains = None
        self._nominal_p_gains = None
        self._nominal_d_gains = None

    def bind_link_masses(self) -> None:
        """Allocate the persistent canonical scale tensor after backend setup."""
        if self._link_mass_scale_range is not None:
            self.link_mass_scale = torch.ones_like(self._backend.link_mass)

    def bind_control_gains(
        self,
        p_gains: torch.Tensor,
        d_gains: torch.Tensor,
        nominal_p_gains: torch.Tensor,
        nominal_d_gains: torch.Tensor,
    ) -> None:
        """Bind the task's persistent per-environment PD gain tensors."""
        self._p_gains = p_gains
        self._d_gains = d_gains
        self._nominal_p_gains = nominal_p_gains
        self._nominal_d_gains = nominal_d_gains
        if self._stiffness_scale_range is not None:
            self.stiffness_scale = torch.ones_like(p_gains)
        if self._damping_scale_range is not None:
            self.damping_scale = torch.ones_like(d_gains)

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
        """Resample inexpensive control parameters for resetting environments."""
        env_ids = self._env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        if self._stiffness_scale_range is not None:
            if self._p_gains is None:
                raise RuntimeError("stiffness DR requires bound control gains")
            scales = self._sample(
                self._stiffness_scale_range,
                (env_ids.numel(), self._p_gains.shape[1]),
                self._generators["stiffness"],
            )
            self.stiffness_scale[env_ids] = scales
            self._p_gains[env_ids] = self._nominal_p_gains * scales

        if self._damping_scale_range is not None:
            if self._d_gains is None:
                raise RuntimeError("damping DR requires bound control gains")
            scales = self._sample(
                self._damping_scale_range,
                (env_ids.numel(), self._d_gains.shape[1]),
                self._generators["damping"],
            )
            self.damping_scale[env_ids] = scales
            self._d_gains[env_ids] = self._nominal_d_gains * scales
