import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import torch.nn.functional as F
import shutil
from pathlib import Path


class BaseExperiment(ABC):
    def __init__(self, model, tokenizer, sampler, dataloader, out_dir, 
                 T, seq_len, vocab_size, epochs, alphas,
                 batch_size=64, lr=1e-4, device="cuda", clear_output = True,
                 denoise_trace_count = 10, samples_count = 10, final_samples_count = 50):
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
       

        self.loss_T_last = np.zeros(T)
        self.loss_epochs = np.zeros(epochs)
        self.counts = np.zeros(T)
        self.weight_decay = 1e-2

        self.denoise_trace_count = denoise_trace_count
        self.samples_count = samples_count
        self.final_samples_count = final_samples_count
        
        self.criterion = self.configure_loss()
        self.optimizer = self.configure_optimizer()

        if clear_output:
            dirpath = Path(out_dir)
            if dirpath.exists() and dirpath.is_dir():
                shutil.rmtree(dirpath)
        dirpath.mkdir(exist_ok= True)

    # === Configurable parts ===
    def configure_optimizer(self, optimizer = torch.optim.AdamW):
        return optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def configure_loss(self, loss_fuction = nn.CrossEntropyLoss):
        return loss_fuction()

    # === Main experiment loop ===
    def run(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)

            # generate samples and denoising trace each epoch
            self.save_epoch_results(epoch)

        # summarize results
        avg_loss_T = self.loss_T_last / np.maximum(self.counts, 1)
        self.save_loss_T_plot(avg_loss_T)
        self.save_samples()
        self.save_loss_plot(self.loss_epochs)
        return avg_loss_T


    def forward_pass(self, x0, t):
        # corruption step
        ground_truth, model_input = self.sampler.forward_diffusion(x0, t)
        logits = self.model(model_input, t)
        return logits, ground_truth

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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

    def save_epoch_results(self, epoch):
        """denoising trace"""
        traces = []
        for i in range(self.denoise_trace_count):
            trace = self.sampler.reverse_diffusion(self.model, self.seq_len, self.T)
            print(trace)
            trace_decoded = [self.tokenizer.decode(x_t.cpu().detach().numpy()[0], skip_special_tokens=False) for x_t in trace]
            traces.append("\n".join(trace_decoded))
        with open(os.path.join(self.out_dir, f"len={self.seq_len},T={self.T}_denoise.txt"), "w") as f:
            for i, tr in enumerate(traces):
                f.write(f"\n=== Example {i+1} ===\n{tr}\n")

        """ sampling """
        samples = [self.sampler.reverse_diffusion(self.model, self.seq_len, self.T)[-1] for _ in range(self.samples_count)]
        samples_decoded = [self.tokenizer.decode(x_t.cpu().detach().numpy()[0], skip_special_tokens=False) for x_t in samples]

        with open(os.path.join(self.out_dir, "epoch_samples.txt"), "a") as f:
            if epoch == 0:
                f.write(f"\n{f"len={self.seq_len}, T={self.T}"}\n")
            f.write(f"\n=== Epoch {epoch+1} ===\n")
            f.write("\n".join(samples_decoded) + "\n")


    def save_samples(self):
        """Final sample generation."""

        samples = [self.sampler.reverse_diffusion(self.model, self.seq_len, self.T)[-1] for _ in range(self.final_samples_count)]
        samples_decoded = [self.tokenizer.decode(x_t.cpu().detach().numpy()[0], skip_special_tokens=False) for x_t in samples]

        with open(os.path.join(self.out_dir, f"samples_seq{self.seq_len}_T{self.T}.txt"), "w") as f:
            f.write("\n".join(samples_decoded))


class MaskedDiffusionExperiment(BaseExperiment):
    
    pass

class DiscreteDiffusionExperiment(BaseExperiment):    
    def count_loss(self, prediction, target):
        loss = self.criterion(prediction.view(-1, self.vocab_size), target.view(-1))
        return loss

    


class SimplexDiffusionExperiment(BaseExperiment):
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

    def count_loss(self, prediction, target):
        # print(prediction.shape, target.shape)
        loss = self.criterion(prediction, target)
        return loss