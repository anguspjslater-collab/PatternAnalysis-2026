"""Train the RNN-GAN baseline on LOBSTER windows. Saves a checkpoint and a loss plot.

Usage:  python train.py --data_dir ~/Desktop/LOBSTER --epochs 5        (Mac smoke test)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_data
from modules import *


def train_rnngan():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)                                   # reproducible runs
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_loader, val_w, _, stats = get_data(args.data_dir)       # test windows untouched until the end

    G, D = RNNGenerator().to(device), RNNDiscriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))   # standard GAN Adam settings
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()                                   # sigmoid + BCE, numerically stable

    hist = {"D": [], "G": []}
    for epoch in range(args.epochs):
        for (real,) in train_loader:
            real = real.to(device)
            z = torch.randn(real.shape[0], real.shape[1], 40, device=device)

            # 1. Discriminator: real -> 1, fake -> 0. detach() stops this step updating G.
            fake = G(z)
            d_real, d_fake = D(real), D(fake.detach())
            loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # 2. Generator: make D score fakes as real
            d_fake = D(fake)
            loss_G = bce(d_fake, torch.ones_like(d_fake))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        hist["D"].append(loss_D.item()); hist["G"].append(loss_G.item())
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  loss_D {loss_D.item():.3f}  loss_G {loss_G.item():.3f}", flush=True)

    # Save weights + normalisation stats (needed to turn generated windows back into books)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "stats": stats, "args": vars(args)},
               "checkpoints/rnngan.pt")

    Path("figures").mkdir(exist_ok=True)
    plt.plot(hist["D"], label="Discriminator"); plt.plot(hist["G"], label="Generator")
    plt.xlabel("Epoch"); plt.ylabel("BCE loss"); plt.title("RNN-GAN training losses"); plt.legend()
    plt.savefig("figures/rnngan_losses.png", dpi=150)

def train_embedding(E, R, train_loader, val_w, device, steps: int = 2000, lr: float = 1e-3):
    """TimeGAN phase 1: train embedder + recovery as an autoencoder on reconstruction MSE.

    Builds a latent space that keeps the information in all features before any GAN training.
    Counts optimiser steps rather than epochs, as the original implementation does.
    Returns per-step training losses and the final validation reconstruction MSE.
    """
    opt = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=lr)   # one optimiser, both nets
    hist, step = [], 0
    while step < steps:
        for (x,) in train_loader:
            x = x.to(device)
            x_tilde = R(E(x))                                  # window -> latent -> window
            loss = F.mse_loss(x_tilde, x)       
            opt.zero_grad(); loss.backward(); opt.step()
            hist.append(loss.item()); step += 1
            if step % 250 == 0:
                print(f"phase 1  step {step:5d}  recon MSE {loss.item():.4f}", flush=True)
            if step >= steps:
                break

    with torch.no_grad():                                      # bottleneck test on unseen data
        v = val_w.to(device)
        val_mse = F.mse_loss(R(E(v)), v).item()
        
    return hist, val_mse



if __name__ == "__main__":
    # baseline_main()
    pass