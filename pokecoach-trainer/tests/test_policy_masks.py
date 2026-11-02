import torch
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.utils import act_len


def test_conditional_mask_blocks_duplicate_switch():
    mask = torch.ones((1, 2 * act_len))
    ally = torch.tensor([[2]])
    updated = MaskedActorCriticPolicy._update_mask(mask, ally)
    assert updated[0, act_len + 2] == 0
