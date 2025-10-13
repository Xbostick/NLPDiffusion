# explore.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

from models import SimpleDiffusionTransformer, SimpleDiffusionMLP
from tokenizer import encode_words_to_ids, load_tokenizer, load_words_file, MASK_ID
from types_ import BPEWordDataset
from utils import make_linear_schedule, sample_from_model, save_denoising_trace, save_epoch_samples
from diffusion_sampling import MaskedDiffusionSampler

from torch.utils.data import DataLoader



XT_PREGEN = None
VOCAB_SIZE = 40
BATCH_SIZE = 128
MAX_SEQ_LEN = 30
N_EPOCHS = 100







def run_experiment(words_file, tokenizer_path, out_dir, T, seq_len, epochs=N_EPOCHS, batch_size=128, d_model=128, n_heads=4, n_layers=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    

    # load tokenizer & data
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer.get_vocab())
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    words = load_words_file(words_file,seq_len=seq_len)
    encoded = encode_words_to_ids(words, tokenizer, seq_len, pad_id)
    ds = BPEWordDataset(encoded)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # model
    model = SimpleDiffusionTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, seq_len=seq_len, T=T).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    alphas = make_linear_schedule(T).to(device)

    sampler = MaskedDiffusionSampler(T,alphas,"cuda", MASK_ID)

    # track per-T loss (last epoch)
    loss_T_last = np.zeros(T)
    counts = np.zeros(T)

    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"T={T}, seq={seq_len}, epoch={epoch+1}")
        for x0 in pbar:
            x0 = x0.to(device)
            b = x0.size(0)
            t = torch.randint(1, T+1, (b,), device=device, dtype=torch.long)
            # corrupt input
            xt_1,xt = sampler.forward_diffusion(x0,t)

            logits = model(xt, t)
            loss = criterion(logits.view(-1, vocab_size), xt_1.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch == epochs - 1:  # record only last epoch
                for ti in t.tolist():
                    loss_T_last[ti-1] += loss.item()
                    counts[ti-1] += 1
        
        save_epoch_samples(model, tokenizer, seq_len, T, alphas, device, epoch, out_dir, 
                           name = f"len = {seq_len}, T = {T}", xt_pregen = XT_PREGEN)
        save_denoising_trace(model, tokenizer, seq_len, T, alphas, device, n_examples=10, out_dir=out_dir, out_file=f"len = {seq_len}, T = {T}_denoise.txt")


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


def grid_search(words_file, tokenizer_path, out_dir="results", T_values=range(10, 101, 30), seq_values=range(5, 26, 10)):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    
    XT_PREGEN = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN), device="cuda", dtype=torch.long)

    for T, seq_len in product(T_values, seq_values):
        avg_loss_T, samples, plot_path = run_experiment(words_file, tokenizer_path, out_dir, T, seq_len)
        results[(T, seq_len)] = avg_loss_T

    # global comparison plot
    plt.figure(figsize=(10,6))
    for (T, seq_len), loss_curve in results.items():
        plt.plot(range(1, len(loss_curve)+1), loss_curve, label=f"T={T}, seq={seq_len}")
    plt.legend()
    plt.title("Comparison of Loss(T)")
    plt.xlabel("t")
    plt.ylabel("Loss")
    plt.grid()
    comp_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(comp_path)
    plt.close()
    print(f"Global comparison saved: {comp_path}")


if __name__ == "__main__":
    # Example run with reduced sweep for runtime
    grid_search("data/cleared_dict.txt", "models/cleared_tokenizer_bpe.json", out_dir="results",
                T_values=range(10,100,20), seq_values=range(5,30,5))
