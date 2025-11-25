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

