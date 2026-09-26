import torch
from torch.optim import AdamW

from recurrent_transformer import RecurrentTransformer

# seed

torch.manual_seed(42)

# model

model = RecurrentTransformer(
    num_tokens = 2,
    dim = 32,
    depth = 2,
    dim_head = 16,
    heads = 2,
    recurrent = True,
    gate_low_rank = 16
)

optimizer = AdamW(model.parameters(), lr = 3e-3, weight_decay = 1e-4)

# parity task

def generate_batch(batch_size, seq_len):
    tokens = torch.randint(0, 2, (batch_size, seq_len))
    return tokens, tokens.cumsum(dim = -1) % 2

# train on short sequences

train_seq_len, batch_size, num_steps = 16, 64, 1000

print(f'training parity on seq len {train_seq_len}...\n')

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

with torch.no_grad():
    for seq_len in (16, 32, 64, 128):
        tokens, labels = generate_batch(100, seq_len)

        preds = model(tokens).argmax(dim = -1)
        token_acc = (preds == labels).float().mean().item()
        seq_acc = (preds == labels).all(dim = -1).float().mean().item()

        print(f'seq len {seq_len:3d} | token acc {token_acc:6.2%} | seq acc {seq_acc:6.2%}')
