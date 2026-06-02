import pytest
import torch

from examples.compare_rng_dropout import _run_check


def test_rng_tracker_dropout_replay_and_outer_restore_cpu():
    result = _run_check(device=torch.device("cpu"), seed=15001, dropout_seed=15002, dropout=0.25)

    assert result["replay_close"] is True
    assert result["stream_advances"] is True
    assert result["outer_rng_restored"] is True
    assert result["replay_max_abs_error"] == 0.0


def test_rng_tracker_dropout_rejects_invalid_probability():
    with pytest.raises(ValueError, match="dropout must be"):
        _run_check(device=torch.device("cpu"), seed=1, dropout_seed=2, dropout=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA dropout RNG test requires CUDA")
def test_rng_tracker_dropout_replay_and_outer_restore_cuda():
    device = torch.device("cuda", torch.cuda.current_device())
    result = _run_check(device=device, seed=15003, dropout_seed=15004, dropout=0.25)

    assert result["replay_close"] is True
    assert result["stream_advances"] is True
    assert result["outer_rng_restored"] is True
    assert result["replay_max_abs_error"] == 0.0
