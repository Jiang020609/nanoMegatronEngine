"""Optional CPU/Gloo distributed collective wrappers."""

from __future__ import annotations

from datetime import timedelta
import os

import torch

from nano_megatron_engine.parallel.fake_tp import partition_range


def is_distributed_available() -> bool:
    """Return whether torch.distributed is available in this PyTorch build."""

    try:
        dist = _distributed_module()
    except RuntimeError:
        return False
    return bool(dist.is_available())


def is_distributed_initialized() -> bool:
    """Return whether a torch.distributed process group is initialized."""

    try:
        dist = _distributed_module()
    except RuntimeError:
        return False
    return bool(dist.is_available() and dist.is_initialized())


def init_distributed_from_env(backend: str = "gloo", timeout_seconds: int = 60) -> None:
    """Initialize torch.distributed using torchrun-style environment variables."""

    dist = _distributed_module()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if dist.is_initialized():
        return

    missing = [name for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT") if name not in os.environ]
    if missing:
        raise RuntimeError(
            "distributed initialization requires torchrun-style environment variables: "
            + ", ".join(missing)
        )

    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))


def get_rank(group: object | None = None) -> int:
    """Return the current distributed rank."""

    dist = _require_initialized()
    return int(dist.get_rank(group=group))


def get_world_size(group: object | None = None) -> int:
    """Return the current distributed world size."""

    dist = _require_initialized()
    return int(dist.get_world_size(group=group))


def distributed_all_reduce_sum(tensor: torch.Tensor, group: object | None = None) -> torch.Tensor:
    """All-reduce a tensor by summing across ranks and returning a new tensor."""

    dist = _require_initialized()
    output = tensor.clone()
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
    return output


def distributed_all_gather(tensor: torch.Tensor, dim: int = -1, group: object | None = None) -> torch.Tensor:
    """All-gather rank-local tensors, supporting uneven sizes along dim."""

    dist = _require_initialized()
    world_size = int(dist.get_world_size(group=group))
    dim = _normalize_dim(dim, tensor.ndim)

    local_shape = tuple(tensor.shape)
    shapes: list[tuple[int, ...] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(shapes, local_shape, group=group)
    gathered_shapes = [_validate_gather_shape(shape, local_shape, dim) for shape in shapes]

    max_dim_size = max(shape[dim] for shape in gathered_shapes)
    padded = _pad_to_dim_size(tensor, dim, max_dim_size).contiguous()
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)

    sliced = []
    for gathered_tensor, shape in zip(gathered, gathered_shapes):
        index = [slice(None)] * gathered_tensor.ndim
        index[dim] = slice(0, shape[dim])
        sliced.append(gathered_tensor[tuple(index)])
    return torch.cat(sliced, dim=dim)


def distributed_reduce_scatter_sum(
    tensor: torch.Tensor,
    dim: int = -1,
    group: object | None = None,
) -> torch.Tensor:
    """Reduce by sum, then return this rank's contiguous output partition."""

    reduced = distributed_all_reduce_sum(tensor, group=group)
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)
    dim = _normalize_dim(dim, reduced.ndim)
    start, end = partition_range(reduced.shape[dim], world_size, rank)
    index = [slice(None)] * reduced.ndim
    index[dim] = slice(start, end)
    return reduced[tuple(index)]


def _distributed_module():
    try:
        import torch.distributed as dist
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("torch.distributed could not be imported") from exc
    return dist


def _require_initialized():
    dist = _distributed_module()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized; call init_distributed_from_env first")
    return dist


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("tensor must have at least one dimension")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim={dim} is out of range for tensor with {ndim} dimensions")
    return dim


def _validate_gather_shape(shape: tuple[int, ...] | None, reference_shape: tuple[int, ...], dim: int) -> tuple[int, ...]:
    if shape is None:
        raise RuntimeError("failed to gather tensor shapes from all ranks")
    if len(shape) != len(reference_shape):
        raise ValueError("all-gather tensors must have the same rank on every rank")
    for axis, (actual, expected) in enumerate(zip(shape, reference_shape)):
        if axis != dim and actual != expected:
            raise ValueError(
                "all-gather tensors must match on non-gather dimensions, "
                f"got dim {axis}: {actual} != {expected}"
            )
    return shape


def _pad_to_dim_size(tensor: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current == dim_size:
        return tensor

    output_shape = list(tensor.shape)
    output_shape[dim] = dim_size
    output = torch.zeros(*output_shape, dtype=tensor.dtype, device=tensor.device)
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(0, current)
    output[tuple(index)] = tensor
    return output
