from torch import tensor, long, LongTensor
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List

class BPEWordDataset(Dataset):
    def __init__(self, encoded_ids: List[List[int]]):
        self.encoded = encoded_ids

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return tensor(self.encoded[idx], dtype=long)
    
class TimeEmbedding(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.emb = nn.Embedding(T + 1, dim)

    def forward(self, t: LongTensor):
        return self.emb(t)