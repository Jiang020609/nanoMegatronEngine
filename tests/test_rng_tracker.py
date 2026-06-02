import pytest
import torch

from nano_megatron_engine.parallel.rng import RNGStateTracker, capture_rng_state, restore_rng_state


def test_rng_tracker_add_does_not_advance_outer_cpu_rng():
    torch.manual_seed(1234)
    expected = torch.rand(5)

    torch.manual_seed(1234)
    tracker = RNGStateTracker(device="cpu")
    tracker.add("model-parallel", 5678)
    actual = torch.rand(5)

    torch.testing.assert_close(actual, expected)
    assert tracker.names() == ("model-parallel",)


def test_rng_tracker_fork_reproducible_and_advances_named_stream():
    tracker = RNGStateTracker(device="cpu")
    tracker.add("dropout", 2026)

    with tracker.fork("dropout"):
        first = torch.rand(4)
    with tracker.fork("dropout"):
        second = torch.rand(4)

    tracker.set_state("dropout", tracker.get_state("dropout"))
    replay_tracker = RNGStateTracker(device="cpu")
    replay_tracker.add("dropout", 2026)
    with replay_tracker.fork("dropout"):
        replay_first = torch.rand(4)
    with replay_tracker.fork("dropout"):
        replay_second = torch.rand(4)

    assert not torch.equal(first, second)
    torch.testing.assert_close(first, replay_first)
    torch.testing.assert_close(second, replay_second)


def test_rng_tracker_fork_restores_outer_cpu_rng():
    torch.manual_seed(777)
    tracker = RNGStateTracker(device="cpu")
    tracker.add("inner", 888)

    torch.manual_seed(777)
    before = torch.rand(3)
    with tracker.fork("inner"):
        _ = torch.rand(10)
    after = torch.rand(3)

    torch.manual_seed(777)
    expected_before = torch.rand(3)
    expected_after = torch.rand(3)
    torch.testing.assert_close(before, expected_before)
    torch.testing.assert_close(after, expected_after)


def test_capture_and_restore_cpu_rng_state():
    torch.manual_seed(901)
    state = capture_rng_state("cpu")
    expected = torch.rand(4)

    _ = torch.rand(9)
    restore_rng_state(state, "cpu")
    actual = torch.rand(4)

    torch.testing.assert_close(actual, expected)


def test_rng_tracker_rejects_invalid_names_and_duplicates():
    tracker = RNGStateTracker(device="cpu")
    with pytest.raises(ValueError, match="non-empty"):
        tracker.add("", 1)
    with pytest.raises(TypeError, match="must be a str"):
        tracker.add(123, 1)  # type: ignore[arg-type]

    tracker.add("x", 1)
    with pytest.raises(ValueError, match="already registered"):
        tracker.add("x", 2)
    with pytest.raises(KeyError, match="not registered"):
        with tracker.fork("missing"):
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG tracker test requires CUDA")
def test_rng_tracker_cuda_fork_restores_outer_rng():
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.manual_seed_all(123)
    tracker = RNGStateTracker(device=device)
    tracker.add("cuda-dropout", 456)

    torch.cuda.manual_seed_all(123)
    before = torch.rand(3, device=device)
    with tracker.fork("cuda-dropout"):
        _ = torch.rand(10, device=device)
    after = torch.rand(3, device=device)

    torch.cuda.manual_seed_all(123)
    expected_before = torch.rand(3, device=device)
    expected_after = torch.rand(3, device=device)
    torch.testing.assert_close(before.cpu(), expected_before.cpu())
    torch.testing.assert_close(after.cpu(), expected_after.cpu())
