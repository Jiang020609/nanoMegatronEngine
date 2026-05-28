import torch

from nano_megatron_engine.engine import split_batch


def test_split_tensor_batch_keeps_order_and_sizes():
    batch = torch.arange(10).view(5, 2)

    microbatches = split_batch(batch, micro_batch_size=2)

    assert [microbatch.shape[0] for microbatch in microbatches] == [2, 2, 1]
    assert torch.equal(torch.cat(microbatches, dim=0), batch)


def test_split_dict_batch_slices_tensor_values():
    batch = {
        "input_ids": torch.arange(12).view(4, 3),
        "labels": torch.arange(12, 24).view(4, 3),
        "name": "toy",
    }

    microbatches = split_batch(batch, micro_batch_size=3)

    assert len(microbatches) == 2
    assert microbatches[0]["name"] == "toy"
    assert microbatches[0]["input_ids"].shape == (3, 3)
    assert microbatches[1]["labels"].shape == (1, 3)
    assert torch.equal(torch.cat([microbatch["input_ids"] for microbatch in microbatches], dim=0), batch["input_ids"])

