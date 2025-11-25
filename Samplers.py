
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class BaseDiffusionSampler(ABC):
    """
    Abstract base class for diffusion samplers.
    Defines the interface for forward and reverse diffusion processes.
    """
    def __init__(self, T: int, alphas: torch.Tensor, device: torch.device):
        self.T = T
        self.alphas = alphas.to(device)
        self.device = device

    @abstractmethod
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Given clean data x0 and timestep t, return (x_t, noise_info)
        noise_info is whatever auxiliary data needed for training loss.
        """
        pass

    @abstractmethod
    def reverse_diffusion(self, model, seq_len: int, T: int):
        """
        Given clean data x0 and timestep t, return (x_t, noise_info)
        noise_info is whatever auxiliary data needed for training loss.
        """
        pass

    
    
class BaseDiscreteDiffusionSampler(BaseDiffusionSampler):
    @torch.no_grad()
    def reverse_diffusion(self, model, seq_len: int, T: int):
        trace = []
        x_t = torch.randint(0, self.vocab_size, [1,seq_len], device=self.device)
        for t in reversed(range(1, T + 1)):
            t_tensor = torch.tensor([t], device=self.device)
            logits = model(x_t, t_tensor)
            probs = F.softmax(logits, dim=-1)
            x_t = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(1, seq_len)
            trace.append(x_t)
        return trace



class DiscreteDiffusionSampler_test_x0(BaseDiscreteDiffusionSampler):
    def __init__(self, T, alphas, device, vocab_size):
        super().__init__(T, alphas, device)
        self.vocab_size = vocab_size

    def forward_diffusion(self, x0, t):
        x0 = x0.to(self.device)
        b = x0.size(0)
        alpha_t = self.alphas[t].unsqueeze(1)

        # corrupt input
        rand = torch.rand(x0.shape, device=self.device)
        replace_mask = rand >= alpha_t
        random_tokens = torch.randint(0, self.vocab_size, x0.shape, device=self.device)
        xt = x0.clone()
        xt[replace_mask] = random_tokens[replace_mask]
        return x0,xt
    
    
class DiscreteDiffusionSampler_test_xt_1(BaseDiscreteDiffusionSampler):
    def __init__(self, T, alphas, device, vocab_size):
        super().__init__(T, alphas, device)
        self.vocab_size = vocab_size

    def forward_diffusion(self, x0, t):
        x0 = x0.to(self.device)
        b = x0.size(0)
        alpha_t_1 = self.alphas[t-1].unsqueeze(1)

        # corrupt input
        rand = torch.rand(x0.shape, device=self.device)
        replace_mask = rand >= alpha_t_1
        random_tokens = torch.randint(0, self.vocab_size, x0.shape, device=self.device)
        xt_1 = x0.clone()
        xt_1[replace_mask] = random_tokens[replace_mask]

        alpha_t = self.alphas[t].unsqueeze(1)

        # corrupt input
        rand = torch.rand(x0.shape, device=self.device)
        replace_mask = rand >= alpha_t
        random_tokens = torch.randint(0, self.vocab_size, x0.shape, device=self.device)
        xt = xt_1.clone()
        xt[replace_mask] = random_tokens[replace_mask]
        while torch.all(torch.all(xt == xt_1, dim = 1)):
            rand = torch.rand(x0.shape, device=self.device)
            replace_mask = rand >= alpha_t
            random_tokens = torch.randint(0, self.vocab_size, x0.shape, device=self.device)
            xt = xt_1.clone()
            xt[replace_mask] = random_tokens[replace_mask]


        return x0,xt
    




class MaskedDiffusionSampler(BaseDiffusionSampler):
    def __init__(self, T, alphas, device, mask_id, vocab_size):
        super().__init__(T, alphas, device)
        self.mask_id = mask_id
        self.vocab_size = vocab_size

    def forward_diffusion(self, x0: torch.LongTensor, t: torch.LongTensor):
        """
        Masking tokens from xt-1 to xt with MASK token according to (1 - alpha_t)
        """
        b, seq = x0.shape
        alpha_t_1 = self.alphas[t].unsqueeze(1).to(self.device)
        rand = torch.rand(b, seq, device=self.device)
        replace_mask_t_1 = rand >= alpha_t_1
        x_t_1 = x0.clone()
        x_t_1[replace_mask_t_1] = self.mask_id

        alpha_t = self.alphas[t].unsqueeze(1).to(self.device)
        rand = torch.rand(b, seq, device=self.device)
        replace_mask_t = rand >= alpha_t
        while torch.all(torch.all(replace_mask_t_1 == replace_mask_t, dim = 1)):
            rand = torch.rand(b, seq, device=self.device)
            replace_mask_t = rand >= alpha_t
        x_t = x_t_1.clone()
        x_t[replace_mask_t] = self.mask_id
        
        return x_t_1, x_t

    @torch.no_grad()
    def reverse_diffusion(self, model, seq_len: int, T: int):
        """
        Reverse sampling: categorical denoising.
        """
        x_t = torch.tensor([[self.mask_id] * seq_len], device=self.device)
        for t in reversed(range(1, T + 1)):
            t_tensor = torch.tensor([t], device=self.device)
            logits = model(x_t, t_tensor)
            probs = F.softmax(logits, dim=-1)
            x_t = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(1, seq_len)
        return x_t.cpu().detach().numpy()[0]

class SimplexDiffusionSampler(BaseDiffusionSampler):
    def __init__(self, T, alphas, device, vocab_size):
        super().__init__(T, alphas, device)
        self.vocab_size = vocab_size
    
  

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Continuous forward process on simplex data:
        x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * noise
        """
        x0 = F.one_hot(x0, num_classes=self.vocab_size).to(torch.float)
        b = x0.size(0)
        a_t = self.alphas[t].view(b, 1, 1).to(self.device)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * noise
        
        return x_t, noise

    @torch.no_grad()
    def reverse_diffusion(self, model, seq_len: int, T):
        """
        DDPM-like reverse diffusion on simplex.
        """
        trace = []
        x_t = torch.randn(1, seq_len, self.vocab_size, device=self.device)
        for t in reversed(range(1,T + 1)):
            t_tensor = torch.tensor([t], device=self.device)
            eps = model(x_t, t_tensor)
            a_t = self.alphas[t]
            a_prev = self.alphas[t - 1] if t > 1 else torch.tensor(1.0, device=self.device)
            beta_t = 1 - a_t / a_prev
            x_t = (1 / torch.sqrt(a_prev)) * (x_t - (beta_t / torch.sqrt(1 - a_t)) * eps)
            if t > 1:
                x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)
            probs = F.softmax(x_t[0], dim=-1)
            tokens = torch.argmax(probs, dim=-1).unsqueeze(0)
            trace.append(tokens)
           
        return trace

class ContinuousEmbeddingDiffusionSampler(BaseDiffusionSampler):
    """
    Continuous diffusion over embedding space.
    Each token is embedded to a vector, noise is added to embeddings,
    and the model learns to predict the noise ε.
    """
    def __init__(self, T, alphas, device, embedding_layer):
        """
        embedding_layer: nn.Embedding or pretrained embedding matrix
        """
        super().__init__(T, alphas, device)
        self.embedding_layer = embedding_layer

    def forward_diffusion(self, x0_idx: torch.LongTensor, t: torch.LongTensor):
        """
        x0_idx: (batch, seq_len) token indices
        Returns:
          x_t: (batch, seq_len, emb_dim) noised embeddings
          noise: added Gaussian noise for training target
        """
        b = x0_idx.size(0)
        x0 = self.embedding_layer(x0_idx).detach()  # (b, seq, emb_dim)
        a_t = self.alphas[t].view(b, 1, 1).to(self.device)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * noise
        return x_t, noise

    @torch.no_grad()
    def reverse_diffusion(self, model, seq_len: int, vocab_size: int):
        """
        Start from Gaussian noise in embedding space and denoise step-by-step.
        At the end, map back to tokens via cosine similarity with embedding matrix.
        """
        emb_dim = self.embedding_layer.embedding_dim
        x_t = torch.randn(1, seq_len, emb_dim, device=self.device)

        emb_matrix = self.embedding_layer.weight.detach()  # (vocab_size, emb_dim)
        emb_norm = F.normalize(emb_matrix, dim=-1)

        for t in reversed(range(1, self.T + 1)):
            t_tensor = torch.tensor([t], device=self.device)
            eps = model(x_t, t_tensor)
            a_t = self.alphas[t]
            a_prev = self.alphas[t - 1] if t > 1 else torch.tensor(1.0, device=self.device)
            beta_t = 1 - a_t / a_prev

            x_t = (1 / torch.sqrt(a_prev)) * (x_t - (beta_t / torch.sqrt(1 - a_t)) * eps)
            if t > 1:
                x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)

        # Map final embeddings to nearest tokens
        x_norm = F.normalize(x_t[0], dim=-1)
        sims = torch.matmul(x_norm, emb_norm.T)  # (seq_len, vocab_size)
        tokens = torch.argmax(sims, dim=-1)
        return tokens
