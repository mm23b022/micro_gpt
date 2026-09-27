"""
MicroGPT: A Minimal GPT-2 Style Language Model
Built from scratch following Karpathy's nanoGPT architecture.
Scaled to run on consumer hardware (GTX 1650, 4GB VRAM).

Architecture: GPT-2 (decoder-only Transformer)
Model size: ~2M parameters (vs GPT-2's 124M)
Dataset: Shakespeare / tiny text corpus
"""

import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# ─────────────────────────────────────────────
# HYPERPARAMETERS (tuned for GTX 1650 4GB)
# ─────────────────────────────────────────────
class Config:
    # Model architecture
    block_size: int = 256          # context length (max tokens)
    vocab_size: int = 65           # character-level vocab (Shakespeare)
    n_layer: int = 6               # number of Transformer blocks
    n_head: int = 6                # number of attention heads
    n_embd: int = 384              # embedding dimension
    dropout: float = 0.2           # dropout probability

    # Training
    batch_size: int = 64           # sequences per batch
    max_iters: int = 5000          # training iterations
    eval_interval: int = 500       # evaluate every N iters
    eval_iters: int = 200          # iterations to estimate loss
    learning_rate: float = 3e-4   # AdamW learning rate
    weight_decay: float = 0.1
    grad_clip: float = 1.0         # gradient clipping

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Character-level tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Vocab size: {vocab_size}, Dataset: {len(data):,} tokens")
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")
    return train_data, val_data, encode, decode, vocab_size


def get_batch(split, train_data, val_data, cfg):
    """Sample a random batch of (x, y) pairs."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


# ─────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape  

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_size)
        att = (q @ k.transpose(-2, -1)) * scale

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  
        y = y.transpose(1, 2).contiguous().view(B, T, C)  
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   
        x = x + self.mlp(self.ln_2(x))    
        return x


class MicroGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  
                wpe=nn.Embedding(config.block_size, config.n_embd),  
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # CORRECTED: Dictionary bracket access for ModuleDict
        self.transformer["wte"].weight = self.lm_head.weight  

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence too long"

        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        
        # CORRECTED: Dictionary bracket access for ModuleDict
        tok_emb = self.transformer["wte"](idx)   
        pos_emb = self.transformer["wpe"](pos)   
        x = self.transformer["drop"](tok_emb + pos_emb)

        # CORRECTED: Dictionary bracket access for ModuleDict
        for block in self.transformer["h"]:
            x = block(x)

        # CORRECTED: Dictionary bracket access for ModuleDict
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    """Estimate loss on train/val sets (averaged over multiple batches)."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            # CORRECTED: pass cfg explicitly
            X, Y = get_batch(split, train_data, val_data, cfg)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def configure_optimizer(model, config: Config):
    decay_params = [p for n, p in model.named_parameters()
                    if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    return optimizer


def train(data_path: str = "input.txt"):
    cfg = Config()
    print(f"Device: {cfg.device}")
    
    # Load data
    train_data, val_data, encode, decode, vocab_size = load_data(data_path)
    cfg.vocab_size = vocab_size

    # Build model
    model = MicroGPT(cfg).to(cfg.device)

    # Comment out or remove the compilation step for local Windows testing
    # try:
    #     model = torch.compile(model)
    #     print("Using torch.compile")
    # except Exception:
    #     print("torch.compile not available, using eager mode")
    print("Using eager mode")

    scaler = torch.cuda.amp.GradScaler() if cfg.device == "cuda" else None
    optimizer = configure_optimizer(model, cfg)

    def get_lr(it):
        warmup_iters = 100
        lr_decay_iters = cfg.max_iters
        min_lr = cfg.learning_rate / 10
        if it < warmup_iters:
            return cfg.learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (cfg.learning_rate - min_lr)

    print(f"\nTraining MicroGPT for {cfg.max_iters} iterations...")
    t0 = time.time()
    best_val_loss = float("inf")

    for it in range(cfg.max_iters):
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % cfg.eval_interval == 0:
            # CORRECTED: pass cfg explicitly
            losses = estimate_loss(model, train_data, val_data, cfg)
            elapsed = time.time() - t0
            print(
                f"step {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), "micro_gpt_best.pt")

        # CORRECTED: pass cfg explicitly
        x, y = get_batch("train", train_data, val_data, cfg)

        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    print("\n--- Generated Text Sample ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    print(decode(generated[0].tolist()))

    return model, encode, decode


if __name__ == "__main__":
    import urllib.request, os
    if not os.path.exists("input.txt"):
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, "input.txt")
        print("Downloaded input.txt")

    model, encode, decode = train("input.txt")