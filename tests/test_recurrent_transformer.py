import torch
import pytest
param = pytest.mark.parametrize

from torch import nn
from recurrent_transformer.recurrent_transformer import Attention, RecurrentTransformer

@param('naive_recurrent', (False, True))
def test_recurrent_transformer(naive_recurrent):

    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        dim_head = 64,
        heads = 2,
        naive_recurrent = naive_recurrent
    )

    ids = torch.randint(0, 256, (2, 16))

    logits = model(ids)

    assert logits.shape == (2, 16, 256)

    loss = model(ids, return_loss = True)

    assert loss.numel() == 1

    loss.backward()
