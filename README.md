# nanogpt-served

A tiny GPT, built from scratch in PyTorch — and then **served** the way a real
inference stack serves: masked self-attention by hand, a KV cache, and a
**continuous-batching scheduler**. The point isn't the model (it's a 0.8M-param
char-level toy); it's learning the inference machinery by building each piece.

## Headline result — continuous batching

An iteration-level scheduler (`scheduler.py`), benchmarked against a static baseline
on the same 5 requests:

| mode | steps | useful / total slot-steps | utilization | served |
|---|---|---|---|---|
| static | 10 | 17 / 26 | 65% | 5 |
| **continuous** | **6** | 17 / 17 | **100%** | 5 |

Same work, same output — it just stops wasting GPU lanes (no idle lanes, no
head-of-line wait). That's throughput ↑ and time-to-first-token ↓, which is
vLLM's core idea in miniature: a `Scheduler` (policy) over a model-execution
layer (mechanism).

**Writeup:** [Continuous batching from scratch](https://balachidam21.github.io/blog/continuous-batching-from-scratch.html)

## What's in here

| file | what it is |
|---|---|
| `model.py` | char-level GPT — single→multi-head causal self-attention, plus a KV cache in `generate()` |
| `train.py` | trains on tiny Shakespeare (loss 4.34 → 1.68) |
| `scheduler.py` | continuous-batching scheduler: admit → decode → evict, re-decided every step |

## Run it

```bash
uv venv && uv sync        # or: pip install torch numpy
uv run train.py           # downloads tiny Shakespeare, trains the toy GPT
uv run scheduler.py       # static-vs-continuous batching benchmark
```

## What's next

`_decode` deliberately **recomputes** the full context every step — no KV cache —
because a naive single-sequence cache doesn't compose with a batch whose membership
and lengths change every step. That mismatch is exactly what **PagedAttention**
solves, and it's the next build.

---

Built to learn, in public. — [balachidam21.github.io](https://balachidam21.github.io)
