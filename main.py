import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from models import SimpleDiffusionTransformer
from tokenizer import encode_words_to_ids, load_tokenizer, load_words_file, train_bpe_tokenizer
from types_ import BPEWordDataset
from utils import make_linear_schedule, sample_from_model


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer setup
    if args.train_tokenizer:
        assert args.tokenizer_path is not None, "--tokenizer_path required when training tokenizer"
        train_bpe_tokenizer(args.words_file, args.vocab_size, args.tokenizer_path)

    assert args.tokenizer_path is not None and os.path.exists(args.tokenizer_path), "Tokenizer file must exist (use --train_tokenizer to create)."
    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = len(tokenizer.get_vocab())
    pad_id = tokenizer.token_to_id('[PAD]') if tokenizer.token_to_id('[PAD]') is not None else 0

    words = load_words_file(args.words_file)
    encoded = encode_words_to_ids(words, tokenizer, args.seq_len, pad_id)
    ds = BPEWordDataset(encoded)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = SimpleDiffusionTransformer(vocab_size=vocab_size,
                                       d_model=args.d_model,
                                       n_heads=args.n_heads,
                                       n_layers=args.n_layers,
                                       seq_len=args.seq_len,
                                       T=args.T).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # warm start
    global_step = 0
    if args.resume_path is not None and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optim_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optim_state_dict'])
        if 'step' in ckpt:
            global_step = ckpt['step']
        print(f"Resumed from {args.resume_path} at step {global_step}")

    alphas = make_linear_schedule(args.T, start=0.0, end=args.end_noise).to(device)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            x0 = batch.to(device)  # (batch, seq_len)
            b = x0.size(0)
            t = torch.randint(1, args.T + 1, (b,), device=device, dtype=torch.long)
            alpha_t = alphas[t].unsqueeze(1)  # (batch,1)

            # corrupt x0 -> xt via uniform replacement per token
            # (Меняем на случайные значения из random_tokens случайные токены из replace_mask)
            rand = torch.rand(x0.shape, device=device)
            replace_mask = rand >= alpha_t
            random_tokens = torch.randint(0, vocab_size, x0.shape, device=device, dtype=torch.long)
            xt = x0.clone()
            xt[replace_mask] = random_tokens[replace_mask]

            logits = model(xt, t)
            loss = criterion(logits.view(-1, vocab_size), x0.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            global_step += 1
            pbar.set_postfix({'loss': float(loss.item()), 'step': global_step})

            if global_step % args.save_every == 0:
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'tokenizer_path': args.tokenizer_path,
                    'args': vars(args),
                    'step': global_step
                }, os.path.join(args.out_dir, f'ckpt_step{global_step}.pt'))

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_path': args.tokenizer_path,
 'args': vars(args),
        'step': global_step
    }, os.path.join(args.out_dir, 'final.pt'))
    print("Training finished. Model saved.")

    print("Sampling examples:")
    for _ in range(10):
        text, ids = sample_from_model(model, tokenizer, args.seq_len, args.T, alphas, device)
        print(text)


# -----------------------
# CLI
# -----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--words_file', type=str, required=True, help='Path to file with one word per line')
    p.add_argument('--out_dir', type=str, default='out', help='Directory to save checkpoints')
    p.add_argument('--seq_len', type=int, default=8, help='Max token length per word (after BPE)')
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--T', type=int, default=200, help='Number of diffusion steps')
    p.add_argument('--end_noise', type=float, default=0.9, help='Final noise level (prob of replacement at T)')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max_steps', type=int, default=20000)
    p.add_argument('--save_every', type=int, default=2000)
    p.add_argument('--clip', type=float, default=1.0)

    # tokenizer options
    p.add_argument('--train_tokenizer', action='store_true', help='Train a BPE tokenizer from words_file')
    p.add_argument('--vocab_size', type=int, default=3000, help='If training tokenizer, target vocab size')
    p.add_argument('--tokenizer_path', type=str, default=None, help='Path to load/save BPE tokenizer (json)')

    # warm start
    p.add_argument('--resume_path', type=str, default=None, help='Checkpoint to resume from')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
