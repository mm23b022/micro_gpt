"""
evaluate.py — Evaluation utilities for MicroGPT

Covers:
- Perplexity calculation
- Token-level accuracy
- Bits-per-character (BPC)
- Simple prompt completion test
"""

import torch
import math
from micro_gpt import MicroGPT, Config, load_data, get_batch


def compute_perplexity(model, val_data, cfg, num_batches=50):
    """
    Perplexity = exp(avg cross-entropy loss).
    Lower is better. A random model on 65-char vocab: PPL ≈ 65.
    A good char-level model on Shakespeare: PPL ≈ 3-6.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            # CORRECTED: Passed None for train_data and explicitly passed cfg
            x, y = get_batch("val", None, val_data, cfg)
            _, loss = model(x, y)
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    bpc = avg_loss / math.log(2)  # bits-per-character = loss / ln(2)
    return {"loss": avg_loss, "perplexity": perplexity, "bpc": bpc}


def evaluate_completions(model, encode, decode, prompts, cfg, max_new=200):
    """Test model on a set of prompts."""
    model.eval()
    for prompt in prompts:
        context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            out = model.generate(context, max_new_tokens=max_new, temperature=0.8, top_k=40)
        print(f"\nPrompt: {repr(prompt)}")
        print(f"Generated:\n{decode(out[0].tolist())}")
        print("─" * 60)


if __name__ == "__main__":
    cfg = Config()
    train_data, val_data, encode, decode, vocab_size = load_data("input.txt")
    cfg.vocab_size = vocab_size

    model = MicroGPT(cfg).to(cfg.device)
    # Ensure a trained model exists before running evaluation
    try:
        model.load_state_dict(torch.load("micro_gpt_best.pt", map_location=cfg.device))
    except FileNotFoundError:
        print("Error: 'micro_gpt_best.pt' not found. Please run micro_gpt.py to train the model first.")
        exit(1)
        
    model.eval()

    print("=== MicroGPT Evaluation ===\n")
    metrics = compute_perplexity(model, val_data, cfg)
    print(f"Val Loss:    {metrics['loss']:.4f}")
    print(f"Perplexity:  {metrics['perplexity']:.2f}")
    print(f"Bits/char:   {metrics['bpc']:.4f}")

    prompts = [
        "ROMEO:",
        "To be, or not to be",
        "The king said",
    ]
    evaluate_completions(model, encode, decode, prompts, cfg)