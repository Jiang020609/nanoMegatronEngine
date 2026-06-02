"""Lightweight named RNG state tracking utilities.

This module intentionally implements a small subset of the RNG-state mechanics
needed for distributed training experiments. It is process-local and does not
claim Megatron-LM RNG tracker parity.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class RNGState:
    """CPU and optional CUDA RNG state snapshot."""

    cpu: torch.Tensor
    cuda: torch.Tensor | None = None


class RNGStateTracker:
    """Track named CPU/CUDA RNG states for deterministic local experiments."""

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else _default_device()
        self._states: dict[str, RNGState] = {}

    def names(self) -> tuple[str, ...]:
        """Return names currently registered in insertion order."""

        return tuple(self._states)

    def add(self, name: str, seed: int) -> None:
        """Register ``name`` with RNG state produced by ``seed``.

        The current process RNG state is restored after registration, so adding
        a named stream does not consume randomness from the caller's active RNG
        stream.
        """

        name = _validate_name(name)
        if name in self._states:
            raise ValueError(f"RNG state {name!r} is already registered")
        if not isinstance(seed, int):
            raise TypeError(f"RNG seed for {name!r} must be an int, got {type(seed).__name__}")

        outer_state = capture_rng_state(self.device)
        try:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                _validate_cuda_device(self.device)
                torch.cuda.manual_seed(seed)
            self._states[name] = capture_rng_state(self.device)
        finally:
            restore_rng_state(outer_state, self.device)

    def get_state(self, name: str) -> RNGState:
        """Return a clone of a registered RNG state."""

        state = self._require_state(name)
        return _clone_state(state)

    def set_state(self, name: str, state: RNGState) -> None:
        """Replace a registered RNG state."""

        name = _validate_name(name)
        self._states[name] = _clone_state(state)

    @contextmanager
    def fork(self, name: str) -> Iterator[None]:
        """Temporarily use a named RNG stream and save its advanced state."""

        name = _validate_name(name)
        named_state = self._require_state(name)
        outer_state = capture_rng_state(self.device)
        try:
            restore_rng_state(named_state, self.device)
            yield
            self._states[name] = capture_rng_state(self.device)
        finally:
            restore_rng_state(outer_state, self.device)

    def reset(self) -> None:
        """Remove all named RNG states."""

        self._states.clear()

    def _require_state(self, name: str) -> RNGState:
        name = _validate_name(name)
        try:
            return self._states[name]
        except KeyError as exc:
            raise KeyError(f"RNG state {name!r} is not registered") from exc


def capture_rng_state(device: torch.device | str | None = None) -> RNGState:
    """Capture current CPU and optional CUDA RNG state."""

    resolved = torch.device(device) if device is not None else _default_device()
    cuda_state = None
    if resolved.type == "cuda":
        _validate_cuda_device(resolved)
        cuda_state = torch.cuda.get_rng_state(resolved)
    return RNGState(cpu=torch.get_rng_state(), cuda=cuda_state)


def restore_rng_state(state: RNGState, device: torch.device | str | None = None) -> None:
    """Restore CPU and optional CUDA RNG state."""

    if not isinstance(state, RNGState):
        raise TypeError(f"state must be an RNGState, got {type(state).__name__}")
    resolved = torch.device(device) if device is not None else _default_device()
    torch.set_rng_state(state.cpu)
    if resolved.type == "cuda":
        _validate_cuda_device(resolved)
        if state.cuda is None:
            raise ValueError("cannot restore CUDA RNG state from an RNGState without cuda state")
        torch.cuda.set_rng_state(state.cuda, resolved)


def _clone_state(state: RNGState) -> RNGState:
    return RNGState(cpu=state.cpu.clone(), cuda=None if state.cuda is None else state.cuda.clone())


def _default_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def _validate_cuda_device(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG tracking requires torch.cuda.is_available()")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is outside device_count={torch.cuda.device_count()}")


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"RNG state name must be a str, got {type(name).__name__}")
    if not name:
        raise ValueError("RNG state name must be non-empty")
    return name
