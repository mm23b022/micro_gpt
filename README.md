# MicroGPT — GPT-2 Style Language Model from Scratch

A minimal, well-documented implementation of a GPT-2 style decoder-only Transformer,
built from scratch in PyTorch. Inspired by Karpathy's [nanoGPT](https://github.com/karpathy/build-nanogpt),
scaled to train on consumer hardware (GTX 1650 4GB, ~10 min).

## Architecture

```
Input tokens
    │
    ▼
Token Embedding (vocab_size → n_embd)
    +
Position Embedding (block_size → n_embd)
    │
    ▼
┌─────────────────────────────┐
│  Transformer Block × 6      │
│                             │
│  LayerNorm                  │
│      ↓                      │
│  CausalSelfAttention (6h)   │
│      ↓  (+ residual)        │
│  LayerNorm                  │
│      ↓                      │
│  MLP (384→1536→384)         │
│      ↓  (+ residual)        │
└─────────────────────────────┘
    │
    ▼
Final LayerNorm
    │
    ▼
LM Head Linear (n_embd → vocab_size)
    │
    ▼
Logits → softmax → next token
```

## Model Config

| Hyperparameter | Value | Notes |
|---|---|---|
| `block_size` | 256 | Context length |
| `n_layer` | 6 | Transformer blocks |
| `n_head` | 6 | Attention heads |
| `n_embd` | 384 | Embedding dim |
| `dropout` | 0.2 | Regularization |
| **Parameters** | **~2M** | vs GPT-2's 124M |

## Key Implementation Details

- **Weight tying**: Token embedding and LM head share weights (saves ~25M params in GPT-2)
- **Causal masking**: Lower-triangular mask ensures tokens only attend to the past
- **Pre-LayerNorm**: Normalize before attention/MLP for stable training (vs original "post-LN")
- **AdamW + weight decay**: Applied to 2D params only (not biases, not LayerNorm)
- **Cosine LR schedule**: With 100-step linear warmup
- **Mixed precision (FP16)**: `torch.cuda.GradScaler` for VRAM efficiency
- **Gradient clipping**: `grad_clip=1.0` to prevent exploding gradients

## Training Results (Shakespeare, ~2M params)

| Metric | Value |
|---|---|
| Train loss | ~1.48 |
| Val loss | ~1.56 |
| Perplexity | ~4.76 |
| Bits/char | ~2.25 |
| Training time (GTX 1650) | ~10 min |

## Quick Start

```bash
# Install dependencies
pip install torch numpy

# Train (downloads Shakespeare automatically)
python micro_gpt.py

# Evaluate
python evaluate.py
```

## File Structure

```
micro-gpt/
├── micro_gpt.py      # Model definition + training loop
├── evaluate.py       # Perplexity, BPC, prompt completions
└── README.md
```

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners — GPT-2 (Radford et al., 2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Karpathy's nanoGPT](https://github.com/karpathy/build-nanogpt)
