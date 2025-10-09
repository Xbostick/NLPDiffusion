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