import torch
import torch.nn as nn
from types_ import TimeEmbedding

class SimpleDiffusionTransformer(nn.Module):    
    """Diffusion Transformerm module
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, seq_len=8, T=200):
        super().__init__()

        # Model embeddings sizes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        # nn.Embeddings convert discrete value to vector 
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.time_emb = TimeEmbedding(T, d_model)

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # взято из ориг. кода, тк с такой начальной развесовкой модель стабильнее сходиться
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb.emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.LongTensor, t: torch.LongTensor):
        batch, seq_len = x.shape
        assert seq_len == self.seq_len
        tok = self.token_emb(x) 

        # Position embedding for attention
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        
        pos = self.pos_emb(pos_ids)
        time = self.time_emb(t)
        time = time.unsqueeze(1).expand(-1, seq_len, -1)

        # Input transformer layer
        h = tok + pos + time
        h = self.transformer(h)
        logits = self.out_proj(h)
        return logits
    

class SimpleDiffusionMLP(nn.Module):
    """
    Simplified diffusion model without Transformer — only token/time MLPs.
    Each token position is processed independently with local MLPs.
    """
    def __init__(self, vocab_size, d_model=256, seq_len=8, T=200, padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.time_emb = TimeEmbedding(T, d_model)

        # Simple 2-layer MLP
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.act = nn.GELU()

        # Output projection
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb.emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.LongTensor, t: torch.LongTensor):
        """
        x: (batch, seq_len)
        t: (batch,)
        """
        print(x.shape)
        batch, seq_len,_ = x.shape
        tok = self.token_emb(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        pos = self.pos_emb(pos_ids)
        time = self.time_emb(t).unsqueeze(1).expand(-1, seq_len, -1)

        h = tok + pos + time

        # Pass through local MLP per token position
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.act(h)

        logits = self.out_proj(h)
        return logits
    
    # model_simplex.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.emb = nn.Embedding(T + 1, dim)

    def forward(self, t):
        return self.emb(t)

class SimplexDiffusionMLP(nn.Module):
    """
    Simplex diffusion: works on continuous simplex-valued inputs (probability vectors)
    Predicts noise ε ~ N(0, I) added to simplex representation
    """
    def __init__(self, vocab_size, d_model=256, seq_len=8, T=200):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        self.token_proj = nn.Linear(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.time_emb = TimeEmbedding(T, d_model)

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )

        self.out_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_t, t):
        """
        x_t: (batch, seq_len, vocab_size)
        t: (batch,)
        """
        b, seq, v = x_t.shape
        x = self.token_proj(x_t)
        pos_ids = torch.arange(seq, device=x.device).unsqueeze(0).expand(b, -1)
        pos = self.pos_emb(pos_ids)
        time = self.time_emb(t).unsqueeze(1).expand(-1, seq, -1)

        h = x + pos + time
        h = self.net(h)
        eps_pred = self.out_proj(h)
        return eps_pred
