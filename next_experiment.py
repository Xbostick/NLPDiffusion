# explore.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from Models import SimpleDiffusionTransformer, SimplexDiffusionMLP
from Tokenizers import encode_words_to_ids, load_tokenizer, load_words_file, MASK_ID
from types_ import BPEWordDataset
from utils import make_linear_schedule
from Samplers import DiscreteDiffusionSampler_test_x0, DiscreteDiffusionSampler_test_xt_1,SimplexDiffusionSampler, ContinuousEmbeddingDiffusionSampler
from Experiments import DiscreteDiffusionExperiment, SimplexDiffusionExperiment

RESULT_PATH = Path('results')
TOKENIZER_PATH = "models/cleared_tokenizer_bpe.json"
WORDS_FILE = "data/cleared_dict.txt"
BATCH_SIZE = 128
T = 5
SEQ_LEN = 5
N_EPOCHS = 1

if __name__ == "__main__":
    # Example run with reduced sweep for runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULT_PATH, exist_ok=True)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = len(tokenizer.get_vocab())
    alphas = make_linear_schedule(T).to(device)
    
    words = load_words_file(WORDS_FILE,seq_len=SEQ_LEN)
    encoded = encode_words_to_ids(words, tokenizer, SEQ_LEN, 1)
    ds = BPEWordDataset(encoded)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # # X0 experiment
    model = SimpleDiffusionTransformer(vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=3, seq_len=SEQ_LEN, T=T).to(device)
    sampler = DiscreteDiffusionSampler_test_x0(T,alphas,"cuda", vocab_size)
    exp = DiscreteDiffusionExperiment(model,tokenizer, sampler,dl, RESULT_PATH / "test_x0", T, SEQ_LEN,vocab_size, epochs= N_EPOCHS, alphas=alphas)
    exp.run()

    # # xt-1 experiment
    model = SimpleDiffusionTransformer(vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=3, seq_len=SEQ_LEN, T=T).to(device)
    sampler = DiscreteDiffusionSampler_test_xt_1(T,alphas,"cuda", vocab_size)
    exp = DiscreteDiffusionExperiment(model,tokenizer, sampler,dl, RESULT_PATH / "test_x1", T, SEQ_LEN,vocab_size, epochs= N_EPOCHS, alphas=alphas)
    exp.run()

    # Simplex try 1
    model = SimplexDiffusionMLP(vocab_size=vocab_size, seq_len=SEQ_LEN, T=T).to(device)
    sampler = SimplexDiffusionSampler(T,alphas,"cuda", vocab_size)
    exp = SimplexDiffusionExperiment(model,tokenizer, sampler,dl, RESULT_PATH / "simplex", T, SEQ_LEN,vocab_size, epochs= N_EPOCHS, alphas=alphas)
    exp.run()

    # # Continious(embedings) diffusion try
    # model = SimplexDiffusionMLP(vocab_size=vocab_size, seq_len=SEQ_LEN, T=T).to(device)
    # sampler = ContinuousEmbeddingDiffusionSampler(T,alphas,"cuda", vocab_size)
    # exp = SimplexDiffusionExperiment(model,tokenizer, sampler,dl, RESULT_PATH / "simplex", T, SEQ_LEN,vocab_size, epochs= N_EPOCHS, alphas=alphas)
    # exp.run()