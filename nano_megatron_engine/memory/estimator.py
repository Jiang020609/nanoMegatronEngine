"""Educational training memory estimator."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class MemoryEstimate:
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    total_bytes: int

    @property
    def total_megabytes(self) -> float:
        return self.total_bytes / (1024**2)


def estimate_training_memory(model: nn.Module, optimizer_name: str = "adamw") -> MemoryEstimate:
    """Estimate parameter, gradient, and optimizer-state memory.

    This is intentionally simple: it counts model parameters exactly, assumes
    gradients use the same dtype as parameters, and approximates optimizer
    state. Adam/AdamW keep two parameter-sized moment buffers; plain SGD keeps
    no extra state unless momentum is added.
    """

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    gradient_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    optimizer_slots = _optimizer_slots(optimizer_name)
    optimizer_state_bytes = parameter_bytes * optimizer_slots
    total_bytes = parameter_bytes + gradient_bytes + optimizer_state_bytes
    return MemoryEstimate(
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        total_bytes=total_bytes,
    )


def format_bytes(num_bytes: int) -> str:
    """Format bytes in human-readable binary units."""

    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def _optimizer_slots(optimizer_name: str) -> int:
    name = optimizer_name.lower()
    if name in {"adam", "adamw"}:
        return 2
    if name in {"sgd", "none"}:
        return 0
    return 1

