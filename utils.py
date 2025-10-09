import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


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
