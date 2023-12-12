import math
import numpy as np
import torch

from torch import Tensor


class Embedder:
    def __init__(
        self, num_freqs, embed_type="nape", input_type="tensor", input_dims=1
    ):
        self.num_freqs = num_freqs
        self.max_freq = max(32, num_freqs)
        self.embed_type = embed_type
        self.input_type = input_type
        self.input_dims = input_dims
        self.eps = 0.01
        if input_type == "tensor":
            self.embed_fns = [torch.sin, torch.cos]
            self.embed = self.embed_tensor
        else:
            self.embed_fns = [np.sin, np.cos]
            self.embed = self.embed_array
        self.create_embedding_fn()

    def __call__(self, x):
        return self.embed(x)

    def create_embedding_fn(self):
        max_freq = self.max_freq
        N_freqs = self.num_freqs

        if self.embed_type == "nape":
            freq_bands = (
                (self.eps + torch.linspace(1, max_freq, N_freqs)) * math.pi / (max_freq + 1)
            )

        elif self.embed_type == "nerf":
            freq_bands = (
                torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
            ) * math.pi

        elif self.embed_type == "trans":
            dim = self.num_freqs
            freq_bands = torch.tensor([1 / (10000 ** (j / dim)) for j in range(dim)])

        self.freq_bands = freq_bands
        self.out_dim = self.input_dims * len(self.embed_fns) * len(freq_bands)

    def embed_tensor(self, inputs: Tensor):
        self.freq_bands = self.freq_bands.to(inputs.device)
        return torch.cat([fn(self.freq_bands * inputs) for fn in self.embed_fns], -1)

    def embed_array(self, inputs):
        return np.concatenate([fn(self.freq_bands * inputs) for fn in self.embed_fns])
