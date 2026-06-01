"""Optional rank-local distributed collective wrappers."""

from __future__ import annotations

from datetime import timedelta
import os

import torch

from nano_megatron_engine.parallel.fake_tp import partition_range

_SUPPORTED_BACKENDS = {
    "gloo": "cpu",
    "nccl": "cuda",
}
_REQUIRED_ENV_VARS = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


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

    backend = _normalize_backend(backend)
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")

    dist = _distributed_module()
    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available in this PyTorch build; "
            "distributed wrappers cannot be initialized with init_distributed_from_env"
    )
    _validate_backend_runtime(dist, backend)
    if dist.is_initialized():
        initialized_backend = _backend_name(dist.get_backend())
        if initialized_backend != backend:
            raise RuntimeError(
                "torch.distributed is already initialized with "
                f"backend={initialized_backend!r}; requested backend={backend!r}"
            )
        return

    missing = [name for name in _REQUIRED_ENV_VARS if name not in os.environ]
    if missing:
        raise RuntimeError(
            "torch.distributed initialization via init_distributed_from_env "
            "requires torchrun-style environment variables: "
            + ", ".join(missing)
        )

    _validate_env_rank_world_size()
    _validate_env_master_addr()
    _validate_env_master_port()
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))


def get_rank(group: object | None = None) -> int:
    """Return the current distributed rank."""

    dist = _require_initialized()
    return int(dist.get_rank(group=group))


def get_world_size(group: object | None = None) -> int:
    """Return the current distributed world size."""

    dist = _require_initialized()
    return int(dist.get_world_size(group=group))


def get_backend(group: object | None = None) -> str:
    """Return the current distributed backend name."""

    dist = _require_initialized()
    return _backend_name(dist.get_backend(group=group))


def get_expected_device_type(group: object | None = None) -> str:
    """Return the tensor device type expected by the current backend."""

    return _expected_device_type_for_backend(get_backend(group=group))


def validate_rank_local_tensor_device(
    tensor: torch.Tensor,
    op_name: str,
    group: object | None = None,
) -> None:
    """Validate that a rank-local tensor matches the initialized backend device."""

    _validate_rank_local_tensor(tensor, op_name, group=group)


def distributed_all_reduce_sum(tensor: torch.Tensor, group: object | None = None) -> torch.Tensor:
    """All-reduce a tensor by summing across ranks and returning a new tensor.

    When autograd is active, backward is the identity. This matches the
    Megatron-style row-parallel output mapping where forward produces a
    replicated summed tensor and each rank consumes the same output gradient.
    """

    _validate_rank_local_tensor(tensor, "distributed_all_reduce_sum", group=group)
    dist = _require_initialized()
    _validate_world_metadata(
        dist,
        tensor,
        group=group,
        match_shape=True,
        match_non_gather_dims=False,
        dim=None,
        op_name="distributed_all_reduce_sum",
    )
    if torch.is_grad_enabled() and tensor.requires_grad:
        return _AllReduceSumForwardIdentityBackward.apply(tensor, group)
    return _all_reduce_sum_impl(tensor, group=group)


def distributed_all_gather(tensor: torch.Tensor, dim: int = -1, group: object | None = None) -> torch.Tensor:
    """All-gather rank-local tensors, supporting uneven sizes along dim."""

    _validate_rank_local_tensor(tensor, "distributed_all_gather", group=group)
    dim = _normalize_dim(dim, tensor.ndim)
    dist = _require_initialized()
    world_size = int(dist.get_world_size(group=group))
    gathered_metadata = _validate_world_metadata(
        dist,
        tensor,
        group=group,
        match_shape=False,
        match_non_gather_dims=True,
        dim=dim,
        op_name="distributed_all_gather",
    )
    gathered_shapes = [metadata["shape"] for metadata in gathered_metadata]

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

    _validate_rank_local_tensor(tensor, "distributed_reduce_scatter_sum", group=group)
    dim = _normalize_dim(dim, tensor.ndim)
    reduced = distributed_all_reduce_sum(tensor, group=group)
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)
    start, end = partition_range(reduced.shape[dim], world_size, rank)
    index = [slice(None)] * reduced.ndim
    index[dim] = slice(start, end)
    return reduced[tuple(index)]


def _distributed_module():
    try:
        import torch.distributed as dist
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "torch.distributed could not be imported; distributed wrappers cannot be initialized "
            "with init_distributed_from_env"
        ) from exc
    return dist


def _require_initialized():
    dist = _distributed_module()
    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available in this PyTorch build; "
            "distributed wrappers cannot be initialized with init_distributed_from_env"
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized for distributed wrappers; "
            "call init_distributed_from_env first"
        )
    return dist


def _all_reduce_sum_impl(tensor: torch.Tensor, group: object | None = None) -> torch.Tensor:
    dist = _require_initialized()
    output = tensor.clone()
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
    return output


class _AllReduceSumForwardIdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: object | None) -> torch.Tensor:
        ctx.group = group
        return _all_reduce_sum_impl(tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _validate_rank_local_tensor(tensor: torch.Tensor, op_name: str, group: object | None = None) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{op_name} expects a torch.Tensor input, got {type(tensor).__name__}")
    if tensor.layout != torch.strided:
        raise ValueError(f"{op_name} expects a dense strided tensor, got layout={tensor.layout}")
    expected_device_type = _expected_device_type_for_rank_local_tensor(group=group)
    if tensor.device.type != expected_device_type:
        raise ValueError(
            f"{op_name} expects {expected_device_type} tensors for the initialized distributed backend, "
            f"got device={tensor.device}"
        )


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("tensor must have at least one dimension")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim={dim} is out of range for tensor with {ndim} dimensions")
    return dim


def _validate_world_metadata(
    dist,
    tensor: torch.Tensor,
    group: object | None,
    match_shape: bool,
    match_non_gather_dims: bool,
    dim: int | None,
    op_name: str,
) -> list[dict[str, object]]:
    world_size = int(dist.get_world_size(group=group))
    if world_size < 1:
        raise RuntimeError(f"torch.distributed returned invalid world_size={world_size}")
    backend = _backend_name(dist.get_backend(group=group))
    expected_device_type = _expected_device_type_for_backend(backend)

    local_metadata = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device_type": tensor.device.type,
    }
    gathered: list[dict[str, object] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_metadata, group=group)
    metadata = [_validate_metadata_object(item, op_name) for item in gathered]

    reference = metadata[0]
    reference_shape = reference["shape"]
    reference_dtype = reference["dtype"]
    if reference["device_type"] != expected_device_type:
        raise ValueError(
            f"{op_name} expects {expected_device_type} tensors for backend={backend}, "
            f"rank 0 has device_type={reference['device_type']}"
        )
    for rank, item in enumerate(metadata[1:], start=1):
        shape = item["shape"]
        if item["dtype"] != reference_dtype:
            raise ValueError(
                f"{op_name} requires matching tensor dtypes across ranks, "
                f"rank 0 has {reference_dtype} but rank {rank} has {item['dtype']}"
            )
        if item["device_type"] != expected_device_type:
            raise ValueError(
                f"{op_name} expects {expected_device_type} tensors for backend={backend}, "
                f"rank {rank} has device_type={item['device_type']}"
            )
        if len(shape) != len(reference_shape):
            raise ValueError(f"{op_name} tensors must have the same rank on every distributed rank")
        if match_shape and shape != reference_shape:
            raise ValueError(
                f"{op_name} requires matching tensor shapes across ranks, "
                f"rank 0 has {reference_shape} but rank {rank} has {shape}"
            )
        if match_non_gather_dims:
            assert dim is not None
            for axis, (actual, expected) in enumerate(zip(shape, reference_shape)):
                if axis != dim and actual != expected:
                    raise ValueError(
                        f"{op_name} tensors must match on non-gather dimensions, "
                        f"rank 0 dim {axis} is {expected} but rank {rank} dim {axis} is {actual}"
                    )
    return metadata


def _validate_metadata_object(item: dict[str, object] | None, op_name: str) -> dict[str, object]:
    if item is None:
        raise RuntimeError(f"{op_name} failed to gather tensor metadata from all distributed ranks")
    shape = item.get("shape")
    dtype = item.get("dtype")
    device_type = item.get("device_type")
    if not isinstance(shape, tuple) or not all(isinstance(size, int) for size in shape):
        raise RuntimeError(f"{op_name} gathered invalid tensor shape metadata")
    if not isinstance(dtype, str):
        raise RuntimeError(f"{op_name} gathered invalid tensor dtype metadata")
    if not isinstance(device_type, str):
        raise RuntimeError(f"{op_name} gathered invalid tensor device metadata")
    return item


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")
    backend_name = backend.lower()
    if backend_name not in _SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise ValueError(
            "nanoMegatronEngine distributed wrappers support backend='gloo' for CPU tensors "
            f"and backend='nccl' for CUDA tensors, got backend={backend!r}; supported: {supported}"
        )
    return backend_name


def _backend_name(backend: object) -> str:
    name = str(backend).lower()
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    return name


def _expected_device_type_for_backend(backend: str) -> str:
    return _SUPPORTED_BACKENDS[_normalize_backend(backend)]


def _expected_device_type_for_rank_local_tensor(group: object | None) -> str:
    if not is_distributed_initialized():
        return "cpu"
    return get_expected_device_type(group=group)


def _validate_backend_runtime(dist, backend: str) -> None:
    if backend == "gloo" and hasattr(dist, "is_gloo_available") and not dist.is_gloo_available():
        raise RuntimeError("backend='gloo' requires a PyTorch build with Gloo support")
    if backend == "nccl":
        if not hasattr(dist, "is_nccl_available") or not dist.is_nccl_available():
            raise RuntimeError("backend='nccl' requires a PyTorch build with NCCL support")
        if not torch.cuda.is_available():
            raise RuntimeError("backend='nccl' requires CUDA to be available")


def _validate_env_rank_world_size() -> tuple[int, int]:
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except ValueError as exc:
        raise ValueError("RANK and WORLD_SIZE must be integer environment variables") from exc
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be at least 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK={rank} must be in [0, WORLD_SIZE={world_size})")
    return rank, world_size


def _validate_env_master_addr() -> str:
    master_addr = os.environ["MASTER_ADDR"].strip()
    if not master_addr:
        raise ValueError("MASTER_ADDR must not be empty")
    return master_addr


def _validate_env_master_port() -> int:
    try:
        port = int(os.environ["MASTER_PORT"])
    except ValueError as exc:
        raise ValueError("MASTER_PORT must be an integer environment variable") from exc
    if port <= 0 or port > 65535:
        raise ValueError(f"MASTER_PORT must be in [1, 65535], got {port}")
    return port


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
