import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import torch.nn.functional as F


from utils import make_linear_schedule, sample_from_model, save_denoising_trace, save_epoch_samples

class BaseExperiment(ABC):
    def __init__(self, model, tokenizer, sampler, dataloader, out_dir,
                 T, seq_len, vocab_size, epochs, alphas,
                 batch_size=64, lr=1e-5, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.dataloader = dataloader
        self.out_dir = out_dir
        self.T = T
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.alphas = alphas
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        self.optimizer = self.configure_optimizer()
        self.criterion = self.configure_loss()

        self.loss_T_last = np.zeros(T)
        self.loss_epochs = np.zeros(epochs)
        self.counts = np.zeros(T)

        os.makedirs(out_dir, exist_ok=True)

    # === Configurable parts ===
    def configure_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def configure_loss(self):
        return nn.CrossEntropyLoss()

    # === Main experiment loop ===
    def run(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)

            # generate samples and denoising trace each epoch
            self.save_epoch_results(epoch)

        # summarize results
        avg_loss_T = self.loss_T_last / np.maximum(self.counts, 1)
        self.save_loss_T_plot(avg_loss_T)
        self.save_samples(avg_loss_T)
        self.save_loss_plot(self.loss_epochs)
        return avg_loss_T

    # === Core methods to override ===
    @abstractmethod
    def forward_pass(self, x0, t):
        """Defines how model processes a batch. Returns logits, ground_truth."""
        pass

    @abstractmethod
    def count_loss(self, prediction, target):
        pass

    def train_one_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.dataloader, desc=f"T={self.T}, seq={self.seq_len}, epoch={epoch+1}")
        epoch_loss = 0

        for i,x0 in enumerate(pbar):
            x0 = x0.to(self.device)
            b = x0.size(0)
            t = torch.randint(1, self.T + 1, (b,), device=self.device, dtype=torch.long)

            logits, ground_truth = self.forward_pass(x0, t)
            
            loss = self.count_loss(logits, ground_truth)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                epoch_loss += loss
                if epoch == self.epochs - 1:
                    for ti in t.tolist():
                        self.loss_T_last[ti - 1] += loss.item()
                        self.counts[ti - 1] += 1

        self.loss_epochs[epoch] = epoch_loss/i

    # === Saving / visualization ===
    def save_loss_T_plot(self, avg_loss_T):
        plt.figure()
        plt.plot(range(1, self.T + 1), avg_loss_T, marker="o")
        plt.title(f"Loss(T) – seq_len={self.seq_len}, T={self.T}")
        plt.xlabel("t")
        plt.ylabel("Loss")
        plt.grid()
        plot_path = os.path.join(self.out_dir, f"loss_seq{self.seq_len}_T{self.T}_by_T.png")
        plt.savefig(plot_path)
        plt.close()
        np.save(os.path.join(self.out_dir, f"loss_seq{self.seq_len}_T{self.T}.npy"), avg_loss_T)

    def save_loss_plot(self, losses):

        # Проверка на тип
        if not isinstance(losses, np.ndarray):
            losses = np.array(losses)
        
        epochs = np.arange(1, len(losses) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, f"loss_seq{self.seq_len}_{self.T}_by_epochs.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
    

    @abstractmethod
    def save_epoch_results(self, epoch):
        """Hook for generating samples or diagnostics per epoch."""
        pass

    @abstractmethod
    def save_samples(self, avg_loss_T):
        """Final sample generation."""
        pass


class MaskedDiffusionExperiment(BaseExperiment):
    def forward_pass(self, x0, t):
        # corruption step
        ground_truth, model_input = self.sampler.forward_diffusion(x0, t)
        logits = self.model(model_input, t)
        return logits, ground_truth

    def save_epoch_results(self, epoch):
        save_epoch_samples(
            self.model, self.tokenizer, self.sampler,
            self.seq_len, self.T, self.alphas,
            self.device, epoch, self.out_dir,
            name=f"len={self.seq_len}, T={self.T}"
        )

        save_denoising_trace(
            self.model, self.tokenizer, self.sampler,
            self.seq_len, self.T, self.alphas,
            self.device, n_examples=10,
            out_dir=self.out_dir,
            out_file=f"len={self.seq_len},T={self.T}_denoise.txt"
        )

    def save_samples(self, avg_loss_T):
        samples = []
        for _ in range(50):
                    
            self.model.eval()
            if sample_steps is None:
                sample_steps = list(range(self.T, 0, -1))
            batch = 1
            vocab_size = len(self.tokenizer.get_vocab())
            xt = torch.randint(0, vocab_size, (batch, self.seq_len), device=self.device, dtype=torch.long)
            with torch.no_grad():
                for t in sample_steps:
                    t_idx = torch.tensor([t], device=self.device, dtype=torch.long)
                    logits = self.model(xt, t_idx)  # (1, seq_len, vocab)
                    probs = F.softmax(logits, dim=-1)
                    # Getting prediction on every step using random sampling due to logits
                    xt = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(batch, self.seq_len)
            # decode first sequence
            ids = xt[0].tolist() 
            # remove padding tokens
            if self.tokenizer.token_to_id('[PAD]') is not None:
                pad_id = self.tokenizer.token_to_id('[PAD]')
                ids = [i for i in ids if i != pad_id]
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            
            samples.append(text)

        with open(os.path.join(self.out_dir, f"samples_seq{self.seq_len}_T{self.T}.txt"), "w") as f:
            f.write("\n".join(samples))

class DiscreteDiffusionExperiment(BaseExperiment):
    def forward_pass(self, x0, t):
        # corruption step
        ground_truth, model_input = self.sampler.forward_diffusion(x0, t)
        logits = self.model(model_input, t)
        # ground_truth = F.one_hot(ground_truth, num_classes=self.vocab_size).to(torch.float)

        return logits, ground_truth
    
    def count_loss(self, prediction, target):
        loss = self.criterion(prediction.view(-1, self.vocab_size), target.view(-1))
        return loss

    def save_epoch_results(self, epoch):
        save_epoch_samples(
            self.model, self.tokenizer, self.sampler,
            self.seq_len, self.T, self.alphas,
            self.device, epoch, self.out_dir,
            name=f"len={self.seq_len}, T={self.T}"
        )

        save_denoising_trace(
            self.model, self.tokenizer, self.sampler,
            self.seq_len, self.T, self.alphas,
            self.device, n_examples=10,
            out_dir=self.out_dir,
            out_file=f"len={self.seq_len},T={self.T}_denoise.txt"
        )

    def save_samples(self, avg_loss_T):
        samples = []
        for _ in range(50):
                    
            self.model.eval()
            sample_steps = list(range(self.T, 0, -1))
            batch = 1
            vocab_size = len(self.tokenizer.get_vocab())
            xt = torch.randint(0, vocab_size, (batch, self.seq_len), device=self.device, dtype=torch.long)
            with torch.no_grad():
                for t in sample_steps:
                    t_idx = torch.tensor([t], device=self.device, dtype=torch.long)
                    logits = self.model(xt, t_idx)  # (1, seq_len, vocab)
                    probs = F.softmax(logits, dim=-1)
                    # Getting prediction on every step using random sampling due to logits
                    xt = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(batch, self.seq_len)
            # decode first sequence
            ids = xt[0].tolist() 
            # remove padding tokens
            if self.tokenizer.token_to_id('[PAD]') is not None:
                pad_id = self.tokenizer.token_to_id('[PAD]')
                ids = [i for i in ids if i != pad_id]
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            
            samples.append(text)
        with open(os.path.join(self.out_dir, f"samples_seq{self.seq_len}_T{self.T}.txt"), "w") as f:
            f.write("\n".join(samples))


class SimplexDiffusionExperiment(BaseExperiment):
    def forward_pass(self, x0, t):
        # corruption step
        model_input,ground_truth   = self.sampler.forward_diffusion(x0, t)
        logits = self.model(model_input, t)
        return logits, ground_truth

    def save_epoch_results(self, epoch):
        save_epoch_samples(
            self.model, self.tokenizer, self.sampler,
            self.seq_len, self.T, self.alphas,
            self.device, epoch, self.out_dir,
            name=f"len={self.seq_len}, T={self.T}"
        )

        save_denoising_trace(
            self.model, self.tokenizer, self.sampler,
            self.seq_len, self.T, self.alphas,
            self.device, n_examples=10,
            out_dir=self.out_dir,
            out_file=f"len={self.seq_len},T={self.T}_denoise.txt"
        )

    def save_samples(self, avg_loss_T):
        samples = []
        for _ in range(50):
            
            tokens = self.sampler.reverse_diffusion(self.model, self.seq_len, self.T)
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            samples.append(text)
        with open(os.path.join(self.out_dir, f"samples_seq{self.seq_len}_T{self.T}.txt"), "w") as f:
            f.write("\n".join(samples))

    def count_loss(self, prediction, target):
        # print(prediction.shape, target.shape)
        loss = self.criterion(prediction, target)
        return loss



class ContiniousDiffusionExperiment(BaseExperiment):
    def forward_pass(self, x0, t):
        # corruption step
        model_input,ground_truth   = self.sampler.forward_diffusion(x0, t)
        #TODO is_embedded=True
        logits = self.model(model_input, t)
        return logits, ground_truth
    #TODO make save_epoch_samples and save_denoising_trace as ABC methods
    def save_epoch_results(self, epoch):
        save_epoch_samples(
            self.model, self.tokenizer, self.sampler,
            self.seq_len, self.T, self.alphas,
            self.device, epoch, self.out_dir,
            name=f"len={self.seq_len}, T={self.T}"
        )

        save_denoising_trace(
            self.model, self.tokenizer, self.sampler,
            self.seq_len, self.T, self.alphas,
            self.device, n_examples=10,
            out_dir=self.out_dir,
            out_file=f"len={self.seq_len},T={self.T}_denoise.txt"
        )

    def save_samples(self, avg_loss_T):
        samples = []
        for _ in range(50):
            
            tokens = self.sampler.reverse_diffusion(self.model, self.seq_len, self.T)
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            samples.append(text)
        with open(os.path.join(self.out_dir, f"samples_seq{self.seq_len}_T{self.T}.txt"), "w") as f:
            f.write("\n".join(samples))

    def count_loss(self, prediction, target):
        # print(prediction.shape, target.shape)
        loss = self.criterion(prediction, target)
        return loss