"""
Diffusion model operating on the probability simplex (logistic-normal / logit-space).
This is a self-contained single-file example implementing:
- tokenizer placeholder (user should replace with real BPE tokenizer)
- a small transformer backbone
- diffusion schedule in logit-space (additive Gaussian on logits)
- training loop with warm-start (checkpoint load/save)
- sampling (ancestral / DDPM-like in logit-space)

Notes:
- "Simplex" here means operating over the probability simplex: vectors that sum to 1 and are positive.
  We represent discrete tokens as one-hot vectors, convert to logits (log of one-hot + small eps -> large values),
  add Gaussian noise in logit space and map back to simplex via softmax.
- This is a practical, easy-to-modify baseline. Replace tokenizer/embedding with your BPE pipeline.

Requirements: torch, tqdm

Author: generated for user
"""

import math
import os
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from types_ import BPEWordDataset
from tokenizer import encode_words_to_ids, load_tokenizer, load_words_file, train_bpe_tokenizer
from utils import sample_from_model

# ---------------------- Utility / Tokenizer placeholders ----------------------
# Replace this with your real BPE tokenizer / vocabulary
class DummyTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode(self, text):
        # placeholder: map chars to ints (very naive)
        ids = [min(ord(c) % (self.vocab_size - 2) + 2, self.vocab_size - 1) for c in text]
        return ids[:128]

    def decode(self, ids):
        return ''.join(chr((i - 2) % 256) for i in ids if i >= 2)


# ---------------------- Simple Dataset example ----------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer: DummyTokenizer, seq_len=64):
        self.texts = texts
        self.tok = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tok.encode(self.texts[idx])
        # pad/truncate
        ids = ids[: self.seq_len]
        ids = ids + [self.tok.pad_token_id] * (self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# ---------------------- Model backbone ----------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # predictor: map transformer outputs to logits for each token position
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids: (B, L)
        x = self.token_embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        # transformer expects (L, B, D)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        logits = self.to_logits(x)  # (B, L, V)
        return logits


# ---------------------- Diffusion in logit-space (simplex) ----------------------
# We operate in unconstrained logit space: z = log p (logits). One-hot tokens -> logits with large magnitude.
# Noise: additive Gaussian on logits. Mapping to simplex: softmax(logits).

class LogitDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-5, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)
        # for numeric stability
        self.eps = 1e-8

    def q_sample_logits(self, logits, t, noise=None):
        """
        Forward diffusion in logit-space: add Gaussian noise to logits scaled by alpha_cumprod.
        logits: (B, L, V)
        t: integer or tensor of timesteps
        """
        if noise is None:
            noise = torch.randn_like(logits)
        a = self.alphas_cumprod[t].view(-1, 1, 1)  # (B,1,1)
        # following DDPM parameterization: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        return torch.sqrt(a) * logits + torch.sqrt(1 - a) * noise, noise

    def predict_start_from_noise(self, x_t, t, noise_pred):
        a = self.alphas_cumprod[t].view(-1, 1, 1)
        return (x_t - torch.sqrt(1 - a) * noise_pred) / (torch.sqrt(a) + self.eps)

    def p_mean_variance(self, model, x_t, t, cond=None):
        # model predicts the noise in logit-space
        noise_pred = model(x_t, t, cond)
        x0_pred = self.predict_start_from_noise(x_t, t, noise_pred)
        # posterior mean/variance (DDPM closed form)
        betat = self.beta[t].view(-1, 1, 1)
        a_t = self.alpha[t].view(-1, 1, 1)
        a_prev = self.alphas_cumprod[t - 1].view(-1, 1, 1) if t > 0 else torch.ones_like(a_t)
        posterior_var = betat * (1 - a_prev) / (1 - self.alphas_cumprod[t])
        # mean formula simplified for vector case (works per DDPM derivation)
        coef1 = betat * torch.sqrt(a_prev) / (1 - self.alphas_cumprod[t])
        coef2 = (1 - a_prev) * torch.sqrt(a_t) / (1 - self.alphas_cumprod[t])
        posterior_mean = coef1 * x0_pred + coef2 * x_t
        return posterior_mean, posterior_var, noise_pred


# ---------------------- Wrapper model that conditions on timestep ----------------------
class DiffusionModel(nn.Module):
    def __init__(self, backbone: SimpleTransformer, timesteps=1000):
        super().__init__()
        self.backbone = backbone
        # timestep embedding to modulate transformer
        self.t_embed = nn.Sequential(
            nn.Linear(1, backbone.token_embed.embedding_dim),
            nn.SiLU(),
            nn.Linear(backbone.token_embed.embedding_dim, backbone.token_embed.embedding_dim),
        )
        # We will add the t-embedding to token embeddings before transformer
        self.timesteps = timesteps

    def forward(self, logits_noisy, t, cond=None):
        # logits_noisy: (B, L, V) in logit-space
        # We'll convert logits_noisy back to token embeddings by multiplying with token embedding matrix.
        # A simple and effective trick: project logits into embedding space using transpose of token embedding.
        # first: softmax to get pseudo-probabilities (not strictly necessary for the model input) - optional
        B, L, V = logits_noisy.shape
        # convert logits to token-embedding-space representation via expected embedding
        # p = softmax(logits)
        p = F.softmax(logits_noisy, dim=-1)  # (B,L,V)
        # token_embed weights: (V, D)
        w = self.backbone.token_embed.weight  # (V, D)
        # x_emb = p @ w -> (B, L, D) (expected embedding under p)
        x_emb = p.matmul(w)

        # add timestep embedding
        t = t.float().view(-1, 1) / float(self.timesteps)
        t_emb = self.t_embed(t)  # (B, D)
        x = x_emb + t_emb.unsqueeze(1)


        # pass through transformer's encoder stack directly using the transformer submodule
        x = x + self.backbone.pos_embed[:, :L, :]
        x = self.backbone.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)

        # predict noise in logit-space: map back from hidden to logits using to_logits
        noise_pred = self.backbone.to_logits(x)  # (B, L, V)
        return noise_pred


# ---------------------- Helpers: convert token ids <-> logits (one-hot / large logits) ----------------------
def ids_to_logits_onehot(ids, vocab_size, large=20.0):
    # ids: (B, L)
    B, L = ids.shape
    logits = torch.full((B, L, vocab_size), -large, device=ids.device)
    logits.scatter_(2, ids.unsqueeze(-1), large)
    return logits


def logits_to_ids(logits):
    # greedy decode from logits
    return torch.argmax(logits, dim=-1)


# ---------------------- Training / Sampling routines ----------------------

def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, device='cpu'):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device)


def train(
    model: DiffusionModel,
    diffusion: LogitDiffusion,
    dataloader: DataLoader,
    optimizer,
    device='cpu',
    epochs=10,
    ckpt_path: Optional[str] = None,
):
    model.to(device)
    diffusion.device = device
    mse = nn.MSELoss()

    start_epoch = 0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, device=device)
        if ckpt is not None:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['opt'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Warm-started from {ckpt_path}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        model.train()
        running_loss = 0.0
        for batch in pbar:
            batch = batch.to(device)  # (B, L)
            B, L = batch.shape
            # get clean logits from ids
            logits_0 = ids_to_logits_onehot(batch, model.backbone.vocab_size).to(device)

            # sample timesteps uniformly for each sample
            t = torch.randint(0, diffusion.timesteps, (B,), device=device)

            # add noise in logit-space
            x_t, noise = diffusion.q_sample_logits(logits_0, t)

            optimizer.zero_grad()
            noise_pred = model(x_t, t)

            loss = mse(noise_pred, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1e-12))

        # checkpoint
        if ckpt_path is not None:
            save_checkpoint({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}, ckpt_path)


@torch.no_grad()
def sample(model: DiffusionModel, diffusion: LogitDiffusion, seq_len=64, batch_size=4, device='cpu'):
    model.to(device)
    diffusion.device = device
    V = model.backbone.vocab_size
    # start from pure Gaussian noise in logit-space
    x_t = torch.randn(batch_size, seq_len, V, device=device)

    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_tensor)
        # predict x0
        x0_pred = diffusion.predict_start_from_noise(x_t, t_tensor, noise_pred)
        if t == 0:
            x_prev = x0_pred
        else:
            mean, var, _ = diffusion.p_mean_variance(lambda *args, **kwargs: noise_pred, x_t, t, None)
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(var).view(-1, 1, 1) * noise
        x_t = x_prev

    # map final logits to ids
    ids = logits_to_ids(x_t)
    return ids


# ---------------------- Example main: putting it together ----------------------
if __name__ == '__main__':
    # toy example — replace data with real corpus and tokenizer
    words_file, tokenizer_path = "data/cleared_dict.txt", "models/cleared_tokenizer_bpe.json"
    batch_size=128
    seq_len = 25

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer.get_vocab())
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    words = load_words_file(words_file,seq_len=seq_len)
    encoded = encode_words_to_ids(words, tokenizer, seq_len, pad_id)
    ds = BPEWordDataset(encoded)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    backbone = SimpleTransformer(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=5, seq_len=seq_len)
    model = DiffusionModel(backbone, timesteps=20)
    diffusion = LogitDiffusion(timesteps=20, beta_start=1e-4, beta_end=0.02, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    ckpt = 'diffusion_simplex_ckpt.pt'
    train(model, diffusion, dl, opt, device=device, epochs=50, ckpt_path=ckpt)

    samples = sample(model, diffusion, seq_len=15, batch_size=4, device=device)
    for s in samples:
        print(tokenizer.decode(s.cpu().numpy().tolist()))
