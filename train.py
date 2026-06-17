"""
Train the tiny GPT on the Tiny Shakespeare dataset (char-level).

Run:  uv run python train.py

Goal of this first pass: watch the loss DROP and see the samples go from random
noise to learned-gibberish that looks Shakespeare-ish.
"""

import os
import urllib.request

import torch

from model import GPT, GPTConfig

# ---------------- hyperparameters ----------------
block_size = 128          # context length (T)
batch_size = 32
max_iters = 3000
eval_interval = 250
eval_iters = 50
learning_rate = 3e-4
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1337)

# ---------------- data ----------------
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
data_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(data_path):
    print("downloading tiny shakespeare...")
    urllib.request.urlretrieve(DATA_URL, data_path)

with open(data_path, "r") as f:
    text = f.read()

# char-level tokenizer: each unique character is a token
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])  # targets shifted by 1
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------- model ----------------
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=128,
    n_layer=4,
    n_head=4,
    dropout=0.1,
)
model = GPT(config).to(device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{n_params:.2f}M parameters | vocab={vocab_size} | device={device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------- training loop ----------------
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------- sample ----------------
print("\n----- sample -----")
start = torch.zeros((1, 1), dtype=torch.long, device=device)  # the newline char (id 0)
print(decode(model.generate(start, max_new_tokens=500)[0].tolist()))

# ---------------- KV-cache benchmark ----------------
import time

def sync():
    if device == "mps":
        torch.mps.synchronize()   # GPU work is async — wait for it before stopping the clock

model.eval()  # turn OFF dropout for inference (else generation is stochastic → unfair comparison)
prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
N = 120  # keep prompt + N <= block_size (128): this toy cache has no eviction

torch.manual_seed(0)
t0 = time.time(); out_naive  = model.generate(prompt, N);         sync(); t_naive  = time.time() - t0
torch.manual_seed(0)
t0 = time.time(); out_cached = model.generate_cached(prompt, N);  sync(); t_cached = time.time() - t0

print(f"\n----- KV-cache benchmark ({N} new tokens) -----")
print(f"naive  generate         : {t_naive:.3f}s")
print(f"cached generate_cached  : {t_cached:.3f}s   ({t_naive / t_cached:.2f}x faster)")
print(f"identical output        : {torch.equal(out_naive, out_cached)}")

