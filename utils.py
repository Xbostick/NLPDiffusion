import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import re
import os

def filter_words(input_file, output_file):
    seen = set()
    with open(input_file, 'r') as f_in:
        with open(output_file, 'w', encoding="utf-8") as f_out:
            for line in f_in:
                word = line.strip().lower()
                # Проверяем, что слово состоит только из латинских букв
                if re.fullmatch(r'[a-z]+', word):
                    if word not in seen:
                        seen.add(word)
                        f_out.write(word + '\n')



def make_linear_schedule(T: int, start: float = 0.0, end: float = 0.9):
    """Return alpha_t array of length T+1 (probability of keeping token at step t).
    alpha_0 = 1.0, alpha_T approx 1-end (so replacement prob increases to end).
    """
    alphas = [1.0]
    for t in range(1, T + 1):
        a = 1.0 - (t / T) * end
        alphas.append(max(0.0, a))
    return torch.tensor(alphas)


def sample_from_model(model: nn.Module, tokenizer: Tokenizer, seq_len: int, T: int, alphas: torch.Tensor, device: torch.device, sample_steps=None, xt_sample = None):
    model.eval()
    if sample_steps is None:
        sample_steps = list(range(T, 0, -1))
    batch = 1
    vocab_size = len(tokenizer.get_vocab())
    if not xt_sample:
        xt = torch.randint(0, vocab_size, (batch, seq_len), device=device, dtype=torch.long)
    else:
        xt = xt_sample[:seq_len]
    with torch.no_grad():
        for t in sample_steps:
            t_idx = torch.tensor([t], device=device, dtype=torch.long)
            logits = model(xt, t_idx)  # (1, seq_len, vocab)
            probs = F.softmax(logits, dim=-1)
            # Getting prediction on every step using random sampling due to logits
            xt = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(batch, seq_len)
    # decode first sequence
    ids = xt[0].tolist()
    # remove padding tokens
    if tokenizer.token_to_id('[PAD]') is not None:
        pad_id = tokenizer.token_to_id('[PAD]')
        ids = [i for i in ids if i != pad_id]
    text = tokenizer.decode(ids)
    return text, ids

def save_denoising_trace(model, tokenizer, seq_len, T, alphas, device, n_examples, out_dir, out_file):
    """Save step-by-step denoising trajectories for inspection."""
    vocab = tokenizer.get_vocab()
    traces = []

    for _ in range(n_examples):
        # start from full noise (random tokens)
        xt = torch.randint(0, len(vocab), (1, seq_len), device=device)
        trace = []

        for t in range(T, 0, -1):
            logits = model(xt, torch.tensor([t], device=device))
            probs = torch.softmax(logits, dim=-1)
            xt = torch.multinomial(probs[0], num_samples=1).squeeze(1).unsqueeze(0)  # resample tokens
            text = tokenizer.decode(xt[0].tolist())
            trace.append(f"t={t}: {text}")

        traces.append("\n".join(trace))

    with open(os.path.join(out_dir, out_file), "w") as f:
        for i, tr in enumerate(traces):
            f.write(f"\n=== Example {i+1} ===\n{tr}\n")

def save_epoch_samples(model, tokenizer, seq_len, T, alphas, device, epoch, out_dir, name, xt_pregen  ):
    """Generate 10 samples and append to a log file for this run."""
    samples = []
    for _ in range(10):
        text, _ = sample_from_model(model, tokenizer, seq_len, T, alphas, device, xt_sample = xt_pregen)
        samples.append(text)
    with open(os.path.join(out_dir, "epoch_samples.txt"), "a") as f:
        if epoch == 0:
            f.write(f"\n{name}\n")
        f.write(f"\n=== Epoch {epoch+1} ===\n")
        f.write("\n".join(samples) + "\n")

