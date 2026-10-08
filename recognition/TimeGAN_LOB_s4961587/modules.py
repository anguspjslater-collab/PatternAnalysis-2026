"""Model components for the COMP3710 TimeGAN LOB project. PyTorch only (spec: no NumPy).

Baseline: RNN-GAN, i.e. a GRU generator and a GRU discriminator trained with a plain
adversarial loss. It is TimeGAN without the embedder, recovery and supervisor networks,
so comparing the two isolates what TimeGAN's extra components contribute.
"""
import torch
import torch.nn as nn

# -- BASELINE MODELS ------------------------------------------------------------
class RNNGenerator(nn.Module):
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


class RNNDiscriminator(nn.Module):
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

# -- TimeGAN MODELS ------------------------------------------------------------
class TGBlock(nn.Module):
    """GRU stack followed by a per-step linear head: the building block of all five TimeGAN networks.

    (batch, T, in_dim) -> (batch, T, out_dim). out_act is applied to the head's output
    (torch.sigmoid for latent-space networks, None for linear outputs).
    Architecture follows Yoon et al., 2019 (GRU cells, sigmoid-bounded latents).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 24, layers: int = 3, out_act=None):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)
        self.out_act = out_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        y = self.out(h)
        return self.out_act(y) if self.out_act is not None else y

class TGGenerator(TGBlock):
    """TimeGAN generator: noise (batch, T, z_dim) -> fake latent sequence (batch, T, hidden) in [0, 1]."""
    def __init__(self, z_dim: int = 40, hidden: int = 24, layers: int = 3):
        super().__init__(in_dim=z_dim, out_dim=hidden, hidden=hidden, layers=layers,
                         out_act=torch.sigmoid)

class TGEmbedder(TGBlock):
    """TimeGAN embedder: real window (batch, T, n_features) -> real latent sequence (batch, T, hidden) in [0, 1]."""
    def __init__(self, n_features: int = 40, hidden: int = 24, layers: int = 3):
        super().__init__(in_dim=n_features, out_dim=hidden, hidden=hidden, layers=layers, out_act=torch.sigmoid)