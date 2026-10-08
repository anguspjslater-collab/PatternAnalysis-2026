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

def train_supervisor(E, S, train_loader, val_w, device, steps: int = 2000, lr: float = 1e-3):
    """TimeGAN phase 2: train the supervisor to predict the next real latent step.

    The embedder (trained in phase 1) is frozen and only produces real latents H. Loss is
    MSE(S(H)[:, :-1], H[:, 1:]): the guess made at step t vs the real step t+1.
    Returns per-step training losses, plus the validation loss next to a persistence baseline
    (predict 'next step = this step'), which the supervisor must beat to have learned dynamics.
    """
    opt = torch.optim.Adam(S.parameters(), lr=lr)              # supervisor only
    hist, step = [], 0
    while step < steps:
        for (x,) in train_loader:
            with torch.no_grad():                              # embedder frozen in this phase
                h = E(x.to(device))
            loss = F.mse_loss(S(h)[:, :-1], h[:, 1:])          # guesses at 1..T-1 vs real steps 2..T
            opt.zero_grad(); loss.backward(); opt.step()
            hist.append(loss.item()); step += 1
            if step % 250 == 0:
                print(f"phase 2  step {step:5d}  supervised MSE {loss.item():.5f}", flush=True)
            if step >= steps:
                break

    with torch.no_grad():
        h = E(val_w.to(device))
        val_loss = F.mse_loss(S(h)[:, :-1], h[:, 1:]).item()
        persistence = F.mse_loss(h[:, :-1], h[:, 1:]).item()  # 'next step = this step'

    return hist, val_loss, persistence

def moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Match per-(step, feature) std and mean across the batch (original TimeGAN 'G_loss_V')."""
    return (x_hat.std(0) - x.std(0)).abs().mean() + (x_hat.mean(0) - x.mean(0)).abs().mean()


def train_joint(E, R, G, S, D, train_loader, device, steps: int = 2000, lr: float = 1e-3,
                z_dim: int = 40, gamma: float = 1.0):
    """TimeGAN phase 3: joint adversarial, supervised, moment and reconstruction training.

    Follows the original implementation (Yoon et al., 2019): per step, two generator and two
    embedder updates, then one discriminator update, skipped when D already wins (loss < 0.15).
    Loss weights (100, 100, 10, 0.1) and uniform noise also follow the original code.
    """
    bce = nn.BCEWithLogitsLoss()
    opt_gs = torch.optim.Adam(list(G.parameters()) + list(S.parameters()), lr=lr)
    opt_er = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=lr)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr)
    noise = lambda x: torch.rand(x.shape[0], x.shape[1], z_dim, device=device)   # uniform [0, 1), as original
    hist = {"G_adv": [], "G_sup": [], "G_mom": [], "E_rec": [], "D": []}
    step = 0
    while step < steps:
        for (x,) in train_loader:
            x = x.to(device)
            for _ in range(2):
                # A. generator + supervisor
                with torch.no_grad():
                    h = E(x)                                  # real latents: a target here, E not updated
                e_hat = G(noise(x))                           # raw fake latents
                h_hat = S(e_hat)                              # rolled forward by the learned dynamics
                x_hat = R(h_hat)                              # synthetic window
                d_hat, d_e = D(h_hat), D(e_hat)
                loss_adv = bce(d_hat, torch.ones_like(d_hat)) + gamma * bce(d_e, torch.ones_like(d_e))
                loss_sup = F.mse_loss(S(h)[:, :-1], h[:, 1:])
                loss_mom = moment_loss(x_hat, x)
                loss_g = loss_adv + 100 * torch.sqrt(loss_sup) + 100 * loss_mom
                opt_gs.zero_grad(); loss_g.backward(); opt_gs.step()

                # B. embedder + recovery keep refining the latent space
                h = E(x)
                loss_rec = F.mse_loss(R(h), x)
                loss_e = 10 * torch.sqrt(loss_rec) + 0.1 * F.mse_loss(S(h)[:, :-1], h[:, 1:])
                opt_er.zero_grad(); loss_e.backward(); opt_er.step()

            # C. discriminator, on detached inputs so only D learns here
            with torch.no_grad():
                h = E(x)
                e_hat = G(noise(x))
                h_hat = S(e_hat)
            d_real, d_fake, d_fake_e = D(h), D(h_hat), D(e_hat)
            loss_d = (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
                      + gamma * bce(d_fake_e, torch.zeros_like(d_fake_e)))
            if loss_d.item() > 0.15:                          # don't let D run away from G
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            for k, v in zip(hist, [loss_adv, loss_sup, loss_mom, loss_rec, loss_d]):
                hist[k].append(v.item())
            step += 1
            if step % 100 == 0:
                print(f"phase 3  step {step:5d}  D {loss_d.item():.3f}  G_adv {loss_adv.item():.3f}  "
                      f"G_sup {loss_sup.item():.5f}  G_mom {loss_mom.item():.3f}  E_rec {loss_rec.item():.3f}", flush=True)
            if step >= steps:
                break
    return hist



if __name__ == "__main__":
    # baseline_main()
    pass