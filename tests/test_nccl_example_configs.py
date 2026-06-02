from examples.compare_distributed_gpt_gradients_nccl import (
    _dense_config as gradient_dense_config,
)
from examples.compare_distributed_gpt_gradients_nccl import (
    _distributed_config as gradient_distributed_config,
)
from examples.compare_distributed_gpt_nccl import (
    _dense_config as smoke_dense_config,
)
from examples.compare_distributed_gpt_nccl import (
    _distributed_checkpoint_config as smoke_checkpoint_config,
)
from examples.compare_distributed_gpt_nccl import (
    _distributed_config as smoke_distributed_config,
)


def test_cuda_nccl_smoke_configs_support_no_bias():
    dense = smoke_dense_config("small", bias=False)
    distributed = smoke_distributed_config(world_size=4, preset="small", bias=False)
    checkpointed = smoke_checkpoint_config(world_size=4, preset="small", bias=False)

    assert dense.bias is False
    assert distributed.bias is False
    assert checkpointed.bias is False
    assert distributed.tensor_parallel_size == 4
    assert checkpointed.use_activation_checkpointing is True


def test_cuda_nccl_gradient_configs_support_no_bias():
    dense = gradient_dense_config("small", bias=False)
    distributed = gradient_distributed_config(world_size=4, preset="small", bias=False)

    assert dense.bias is False
    assert distributed.bias is False
    assert distributed.tensor_parallel_size == 4
