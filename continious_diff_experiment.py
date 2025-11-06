# explore.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from torch.utils.data import DataLoader

from Models import SimplexDiffusionMLP, SimpleDiffusionMLP
from Tokenizers import encode_words_to_ids, load_tokenizer, load_words_file, MASK_ID
from types_ import BPEWordDataset
from utils import make_linear_schedule, sample_from_model, save_denoising_trace, save_epoch_samples, run_experiment
from Samplers import ContinuousEmbeddingDiffusionSampler
from Experiments import ContiniousDiffusionExperiment

RESULT_PATH = 'results'
TOKENIZER_PATH = "models/cleared_tokenizer_bpe.json"
WORDS_FILE = "data/cleared_dict.txt"
BATCH_SIZE = 64
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
    
    
    model = SimpleDiffusionMLP(vocab_size=vocab_size, seq_len=SEQ_LEN, T=T).to(device)
    sampler = ContinuousEmbeddingDiffusionSampler(T, alphas, device, embedding_layer=model.token_emb)

    exp = ContiniousDiffusionExperiment(model,tokenizer, sampler,dl, RESULT_PATH, T, SEQ_LEN,vocab_size, epochs= N_EPOCHS, alphas=alphas)
    exp.run()

    