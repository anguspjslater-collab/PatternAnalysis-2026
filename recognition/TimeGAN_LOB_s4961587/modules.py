"""Model components for the COMP3710 TimeGAN LOB project. PyTorch only (spec: no NumPy).

Baseline: RNN-GAN, i.e. a GRU generator and a GRU discriminator trained with a plain
adversarial loss. It is TimeGAN without the embedder, recovery and supervisor networks,
so comparing the two isolates what TimeGAN's extra components contribute.
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Noise sequence (batch, seq_len, z_dim) -> synthetic window (batch, seq_len, n_features).

    A GRU reads one noise vector per step and carries a hidden state forward, so each
    generated step can depend on the steps before it.
    """
    def __init__(self, z_dim: int = 40, hidden: int = 64, n_features: int = 40, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(z_dim, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)      # hidden state -> 40 normalised features

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(z)                            # (batch, seq_len, hidden)
        return self.out(h)                            # (batch, seq_len, n_features)


class Discriminator(nn.Module):
    """Window (batch, seq_len, n_features) -> one real/fake logit per step (batch, seq_len, 1).

    Scoring every step (as in TimeGAN) gives the generator feedback on the whole
    sequence, not just its last step. Outputs logits; the loss applies the sigmoid.
    """
    def __init__(self, n_features: int = 40, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.out(h)