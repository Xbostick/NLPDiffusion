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

from Models import SimpleDiffusionTransformer, SimpleDiffusionMLP
from Tokenizers import encode_words_to_ids, load_tokenizer, load_words_file, MASK_ID
from types_ import BPEWordDataset
from utils import make_linear_schedule, sample_from_model, save_denoising_trace, save_epoch_samples, run_experiment
from Samplers import DiscreteDiffusionSampler_test_x0, DiscreteDiffusionSampler_test_xt_1
from Experiments import DiscreteDiffusionExperiment

RESULT_PATH = Path('results')
TOKENIZER_PATH = "models/cleared_tokenizer_bpe.json"
WORDS_FILE = "data/cleared_dict.txt"
BATCH_SIZE = 256
T = 10
SEQ_LEN = 5
N_EPOCHS = 2

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
    model = SimpleDiffusionTransformer(vocab_size=vocab_size, d_model=256, n_heads=8, n_layers=6, seq_len=SEQ_LEN, T=T).to(device)
    
    sampler = DiscreteDiffusionSampler_test_x0(T,alphas,"cuda", vocab_size)

    exp = DiscreteDiffusionExperiment(model,tokenizer, sampler,dl, RESULT_PATH / "test_x0", T, SEQ_LEN,vocab_size, epochs= N_EPOCHS, alphas=alphas)
    exp.run()

    exit()
    sampler = DiscreteDiffusionSampler_test_xt_1(T,alphas,"cuda", vocab_size)
    exp = DiscreteDiffusionExperiment(model,tokenizer, sampler,dl, RESULT_PATH / "test_x1", T, SEQ_LEN,vocab_size, epochs= N_EPOCHS, alphas=alphas)
    exp.run()