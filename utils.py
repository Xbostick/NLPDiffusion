import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import re
import os
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
    text = tokenizer.decode(ids, skip_special_tokens=False)
    return text, ids

def save_denoising_trace(model, tokenizer, sampler, seq_len, T, alphas, device, n_examples, out_dir, out_file):
    """Save step-by-step denoising trajectories for inspection."""
    vocab = tokenizer.get_vocab()
    traces = []

    for _ in range(n_examples):
        # start from full noise (random tokens)
        xt = torch.randint(0, len(vocab), (1, seq_len), device=device)
        trace = []

        for t in range(T, 0, -1):
            ids = sampler.reverse_diffusion(model,seq_len, T)
            # xt = torch.multinomial(probs[0], num_samples=1).squeeze(1).unsqueeze(0)  # resample tokens
            text = tokenizer.decode(ids, skip_special_tokens=False)
            trace.append(f"t={t}: {text}")

        traces.append("\n".join(trace))

    with open(os.path.join(out_dir, out_file), "w") as f:
        for i, tr in enumerate(traces):
            f.write(f"\n=== Example {i+1} ===\n{tr}\n")

def save_epoch_samples(model, tokenizer, sampler, seq_len, T, alphas, device, epoch, out_dir, name):
    """Generate 10 samples and append to a log file for this run."""
    samples = []
    for _ in range(10):
        ids = sampler.reverse_diffusion(model,seq_len, T)
        logger.info(ids)
        text = tokenizer.decode(ids, skip_special_tokens=False)

        # text, _ = sample_from_model(model, tokenizer, seq_len, T, alphas, device, xt_sample = xt_pregen)
        samples.append(text)
    with open(os.path.join(out_dir, "epoch_samples.txt"), "a") as f:
        if epoch == 0:
            f.write(f"\n{name}\n")
        f.write(f"\n=== Epoch {epoch+1} ===\n")
        f.write("\n".join(samples) + "\n")

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
def run_experiment( tokenizer, sampler, model, dataloader, out_dir, T, seq_len,vocab_size, epochs, alphas, batch_size=64, lr = 1e-5, device = "cuda"):
     # model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
   

    # track per-T loss (last epoch)
    loss_T_last = np.zeros(T)
    counts = np.zeros(T)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"T={T}, seq={seq_len}, epoch={epoch+1}")
        for x0 in pbar:
            x0 = x0.to(device)
            b = x0.size(0)
            t = torch.randint(1, T+1, (b,), device=device, dtype=torch.long)
            # corrupt input
            ground_truth, model_input = sampler.forward_diffusion(x0,t)
            # print(f"x0 = {x0[0]} xt_1 = {xt_1[0]} xt = {xt[0]} \n")
            logits = model(model_input, t)
            loss = criterion(logits.view(-1, vocab_size), ground_truth.view(-1))
            # print(logits.view(-1, vocab_size)[0], xt_1.view(-1)[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch == epochs - 1:  # record only last epoch
                for ti in t.tolist():
                    loss_T_last[ti-1] += loss.item()
                    counts[ti-1] += 1
        
        save_epoch_samples(model, tokenizer, sampler, seq_len, T, alphas, device, epoch, out_dir, 
                           name = f"len = {seq_len}, T = {T}")
        save_denoising_trace(model, tokenizer,sampler, seq_len, T, alphas, device, n_examples=10, out_dir=out_dir, out_file=f"len = {seq_len}, T = {T}_denoise.txt")


    avg_loss_T = loss_T_last / np.maximum(counts, 1)

    # save loss curve
    plt.figure()
    plt.plot(range(1, T+1), avg_loss_T, marker="o")
    plt.title(f"Loss(T) – seq_len={seq_len}, T={T}")
    plt.xlabel("t")
    plt.ylabel("Loss")
    plt.grid()
    plot_path = os.path.join(out_dir, f"loss_seq{seq_len}_T{T}.png")
    plt.savefig(plot_path)
    plt.close()

    # generate 50 samples
    samples = []
    for _ in range(50):
        text, _ = sample_from_model(model, tokenizer, seq_len, T, alphas, device)
        samples.append(text)
    with open(os.path.join(out_dir, f"samples_seq{seq_len}_T{T}.txt"), "w") as f:
        f.write("\n".join(samples))

    # save numeric loss(T)
    np.save(os.path.join(out_dir, f"loss_seq{seq_len}_T{T}.npy"), avg_loss_T)
    # trace_path = os.path.join(out_dir, "denoising_trace.txt")
    # save_denoising_trace(model, tokenizer, seq_len, T, alphas, device, n_examples=10, out_dir=out_dir, out_file=trace_pa)

    return avg_loss_T, samples, plot_path
