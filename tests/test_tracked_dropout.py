import pytest
import torch

from nano_megatron_engine.parallel import RNGStateTracker, TrackedDropout, make_rank_local_rng_tracker


def test_tracked_dropout_replays_when_stream_state_is_reset():
    x = torch.ones(12, 12)
    tracker = RNGStateTracker(device="cpu")
    tracker.add("dropout", 1701)
    initial_state = tracker.get_state("dropout")
    dropout = TrackedDropout(0.35, rng_tracker=tracker, rng_name="dropout")
    dropout.train()

    first = dropout(x)
    second = dropout(x)
    tracker.set_state("dropout", initial_state)
    replay = dropout(x)

    torch.testing.assert_close(replay, first)
    assert not torch.equal(second, first)


def test_tracked_dropout_eval_does_not_advance_named_stream():
    x = torch.randn(4, 8)
    tracker = RNGStateTracker(device="cpu")
    tracker.add("dropout", 1702)
    before = tracker.get_state("dropout")
    dropout = TrackedDropout(0.5, rng_tracker=tracker, rng_name="dropout")
    dropout.eval()

    torch.testing.assert_close(dropout(x), x)

    after = tracker.get_state("dropout")
    assert torch.equal(after.cpu, before.cpu)


def test_make_rank_local_rng_tracker_offsets_seed_by_rank():
    x = torch.ones(16, 16)
    rank0 = make_rank_local_rng_tracker("dropout", seed=1703, rank=0, device="cpu")
    rank1 = make_rank_local_rng_tracker("dropout", seed=1703, rank=1, device="cpu")
    dropout0 = TrackedDropout(0.5, rng_tracker=rank0, rng_name="dropout")
    dropout1 = TrackedDropout(0.5, rng_tracker=rank1, rng_name="dropout")
    dropout0.train()
    dropout1.train()

    assert not torch.equal(dropout0(x), dropout1(x))


def test_tracked_dropout_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="tracked dropout probability"):
        TrackedDropout(1.0)
    with pytest.raises(TypeError, match="rng_tracker"):
        TrackedDropout(0.1, rng_tracker=object())
    with pytest.raises(ValueError, match="rng_name"):
        TrackedDropout(0.1, rng_name="")
    with pytest.raises(ValueError, match="rank-local RNG rank"):
        make_rank_local_rng_tracker("dropout", seed=1, rank=-1, device="cpu")
