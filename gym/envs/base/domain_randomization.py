"""Backend-neutral sampling for physical and control domain parameters.

The task layer owns random sampling. Physics backends only apply explicit
values, which isolates DR from task RNG and keeps parameter semantics
independent of the selected engine.
"""

import torch


def contact_friction_range(cfg) -> tuple[float, float] | None:
    """Return and validate the configured effective sliding-friction range.

    ``None`` disables contact-friction randomization. The portable coefficient
    maps to MuJoCo sliding friction and to both VSim static and dynamic
    friction.
    """
    settings = getattr(cfg, "domain_randomization", None)
    if settings is None:
        return None
    values = settings.contact_friction_range
    if values is None:
        return None
    low, high = map(float, values)
    if not 0.0 <= low <= high:
        raise ValueError(
            f"contact-friction range must satisfy 0 <= low <= high, got [{low}, {high}]"
        )
    return low, high


class DomainRandomizer:
    """Sample enabled domain parameters and apply them through ``SimBackend``."""

    def __init__(self, cfg, backend, device: str) -> None:
        self._backend = backend
        self._device = device
        self._contact_friction_range = contact_friction_range(cfg)

        self._generator = None
        if self._contact_friction_range is not None:
            seed = getattr(cfg, "seed", None)
            if seed is None or seed < 0:
                raise ValueError(
                    "enabled domain randomization requires a resolved "
                    "non-negative cfg.seed"
                )
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(seed)

    @property
    def contact_friction(self) -> torch.Tensor:
        """Current applied coefficient for every environment."""
        return self._backend.contact_friction

    def randomize(self, env_ids: torch.Tensor) -> None:
        """Resample enabled parameters for exactly ``env_ids``."""
        if self._contact_friction_range is None:
            return
        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self._device
        ).flatten()
        if env_ids.numel() == 0:
            return
        low, high = self._contact_friction_range
        values = low + (high - low) * torch.rand(
            env_ids.numel(), generator=self._generator, device=self._device
        )
        self._backend.set_contact_friction(env_ids, values)
