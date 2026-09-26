import torch
import pytest
param = pytest.mark.parametrize

from recurrent_transformer.recurrent_transformer import Attention, RecurrentTransformer

@param('recurrent', (False, True))
@param('block_size', (1, 2, 4))
def test_recurrent_transformer(recurrent, block_size):

    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        dim_head = 64,
        heads = 2,
        recurrent = recurrent,
        block_size = block_size
    )

    ids = torch.randint(0, 256, (2, 16))

    logits = model(ids)

    assert logits.shape == (2, 16, 256)

    loss = model(ids, return_loss = True)

    assert loss.numel() == 1

    loss.backward()
