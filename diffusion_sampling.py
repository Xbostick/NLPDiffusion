
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


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
    def reverse_diffusion(self, model, seq_len: int):
        """
        Sample new data by iteratively denoising from x_T to x_0 using the model.
        Returns final data tensor or sampled tokens.
        """
        pass

class DiscreteDiffusionSampler(BaseDiffusionSampler):
    def __init__(self, T, alphas, device, vocab_size):
        super().__init__(T, alphas, device)
        self.vocab_size = vocab_size

    def forward_diffusion(self, x0: torch.LongTensor, t: torch.LongTensor):
        """
        Replace tokens in x0 with random tokens according to (1 - alpha_t)
        """
        b, seq = x0.shape
        alpha_t = self.alphas[t].unsqueeze(1).to(self.device)
        rand = torch.rand(b, seq, device=self.device)
        replace_mask = rand >= alpha_t
        random_tokens = torch.randint(0, self.vocab_size, (b, seq), device=self.device)
        x_t = x0.clone()
        x_t[replace_mask] = random_tokens[replace_mask]
        return x_t, replace_mask

    @torch.no_grad()
    def reverse_diffusion(self, model, seq_len: int):
        """
        Reverse sampling: categorical denoising.
        """
        x_t = torch.randint(0, self.vocab_size, (1, seq_len), device=self.device)
        for t in reversed(range(1, self.T + 1)):
            t_tensor = torch.tensor([t], device=self.device)
            logits = model(x_t, t_tensor)
            probs = F.softmax(logits, dim=-1)
            x_t = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(1, seq_len)
        return x_t

class SimplexDiffusionSampler(BaseDiffusionSampler):
    def __init__(self, T, alphas, device, vocab_size):
        super().__init__(T, alphas, device)
        self.vocab_size = vocab_size

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Continuous forward process on simplex data:
        x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * noise
        """
        b = x0.size(0)
        a_t = self.alphas[t].view(b, 1, 1).to(self.device)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * noise
        return x_t, noise

    @torch.no_grad()
    def reverse_diffusion(self, model, seq_len: int):
        """
        DDPM-like reverse diffusion on simplex.
        """
        x_t = torch.randn(1, seq_len, self.vocab_size, device=self.device)
        for t in reversed(range(1, self.T + 1)):
            t_tensor = torch.tensor([t], device=self.device)
            eps = model(x_t, t_tensor)
            a_t = self.alphas[t]
            a_prev = self.alphas[t - 1] if t > 1 else torch.tensor(1.0, device=self.device)
            beta_t = 1 - a_t / a_prev
            x_t = (1 / torch.sqrt(a_prev)) * (x_t - (beta_t / torch.sqrt(1 - a_t)) * eps)
            if t > 1:
                x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)
        probs = F.softmax(x_t[0], dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        return tokens
