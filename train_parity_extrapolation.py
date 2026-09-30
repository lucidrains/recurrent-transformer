# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "recurrent-transformer-pytorch>=0.0.10",
# ]
# ///

import fire
import torch
from torch import nn
from torch.optim import AdamW

from recurrent_transformer import GatedTransition, RecurrentTransformer

# parity task

def generate_batch(batch_size, seq_len):
    tokens = torch.randint(0, 2, (batch_size, seq_len))
    return tokens, tokens.cumsum(dim = -1) % 2

def main(
    recurrent_mode = 'tiled',
    block_size = 1,
    state_transition = 'none',
    train_seq_len = 16,
    batch_size = 64,
    num_steps = 1000,
    lr = 3e-3,
    weight_decay = 1e-4,
    seed = 42
):
    assert recurrent_mode in ('naive', 'tiled')
    assert block_size >= 1
    assert state_transition in ('none', 'gated', 'gru')

    # seed

    torch.manual_seed(seed)

    # model

    dim = 32
    transition = None

    if state_transition == 'gru':
        transition = nn.GRU(dim, dim, batch_first = True)
    elif state_transition == 'gated':
        transition = GatedTransition(dim)

    model = RecurrentTransformer(
        num_tokens = 2,
        dim = dim,
        depth = 2,
        dim_head = 16,
        heads = 2,
        recurrent = True,
        recurrent_mode = recurrent_mode,
        block_size = block_size,
        gate_low_rank = 16,
        state_transition = transition
    )

    optimizer = AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)

    # train on short sequences

    print(f'training parity on seq len {train_seq_len} ({recurrent_mode} recurrent, block size {block_size}, state transition {state_transition})...\n')

    for step in range(1, num_steps + 1):
        tokens, labels = generate_batch(batch_size, train_seq_len)

        loss = model(tokens, labels = labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        optimizer.step()
        optimizer.zero_grad()

        if step % 250 == 0:
            print(f'step {step:4d} | loss: {loss.item():.4f}')

    # length extrapolation

    print('\nevaluating length extrapolation...\n')

    model.eval()

    header = f'{"seq len":>8} | {"token acc":>9} | {"seq acc":>7}'
    print(header)
    print('-' * len(header))

    with torch.no_grad():
        for seq_len in (16, 32, 64, 128):
            tokens, labels = generate_batch(100, seq_len)

            preds = model(tokens).argmax(dim = -1)
            token_acc = (preds == labels).float().mean().item()
            seq_acc = (preds == labels).all(dim = -1).float().mean().item()

            print(f'{seq_len:>8} | {token_acc:>9.2%} | {seq_acc:>7.2%}')

if __name__ == '__main__':
    fire.Fire(main)
