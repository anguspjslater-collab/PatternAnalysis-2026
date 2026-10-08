"""Train the RNN-GAN baseline or TimeGAN on LOBSTER windows (COMP3710 TimeGAN project).

Usage:
    python train.py --data_dir ~/Desktop/LOBSTER --model timegan --steps 300 --eval_every 100   (Mac smoke test)
    python train.py --data_dir ~/data/LOBSTER --model timegan --steps 10000 --seed 0            (Rangpur)
    python train.py --data_dir ~/data/LOBSTER --model rnngan  --steps 10000 --seed 0

Every --eval_every steps the generator is scored on the VALIDATION windows (KL on spread and
returns plus invariant breakage). The best-scoring checkpoint is kept, so both models are selected
the same way. The test split is never touched here; predict.py uses it once at the end.
Outputs: checkpoints/{run}.pt, figures/{run}_losses.png, results/{run}.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                          # save figures without a display (Rangpur)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from dataset import get_data
from modules import (RNNGenerator, RNNDiscriminator, rnngan_generate,
                     TGEmbedder, TGRecovery, TGSupervisor, TGGenerator, TGDiscriminator, tg_generate)


# ----------------------------------------------------------------------------- baseline

def train_rnngan(G, D, train_loader, device, steps: int, lr: float = 2e-4, z_dim: int = 40,
                 eval_fn=None, eval_every: int = 250) -> dict:
    """RNN-GAN baseline: plain adversarial training (non-saturating BCE), one D and one G update per step."""
    bce = nn.BCEWithLogitsLoss()                                # sigmoid + BCE, numerically stable
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))   # standard GAN Adam settings
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    hist, step = {"D": [], "G": []}, 0
    while step < steps:
        for (real,) in train_loader:
            real = real.to(device)
            fake = G(torch.randn(real.shape[0], real.shape[1], z_dim, device=device))

            # Discriminator: real -> 1, fake -> 0. detach() stops this update reaching G.
            d_real, d_fake = D(real), D(fake.detach())
            loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Generator: make D score fakes as real
            d_fake = D(fake)
            loss_G = bce(d_fake, torch.ones_like(d_fake))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            hist["D"].append(loss_D.item()); hist["G"].append(loss_G.item())
            step += 1
            if step % 100 == 0:
                print(f"rnngan   step {step:5d}  D {loss_D.item():.3f}  G {loss_G.item():.3f}", flush=True)
            if eval_fn and step % eval_every == 0:
                eval_fn(step)
            if step >= steps:
                break
    return hist


# ----------------------------------------------------------------------------- TimeGAN phases

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
            x_tilde = R(E(x))                                   # window -> latent -> window
            loss = F.mse_loss(x_tilde, x)
            opt.zero_grad(); loss.backward(); opt.step()
            hist.append(loss.item()); step += 1
            if step % 250 == 0:
                print(f"phase 1  step {step:5d}  recon MSE {loss.item():.4f}", flush=True)
            if step >= steps:
                break

    with torch.no_grad():                                       # bottleneck test on unseen data
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
    opt = torch.optim.Adam(S.parameters(), lr=lr)               # supervisor only
    hist, step = [], 0
    while step < steps:
        for (x,) in train_loader:
            with torch.no_grad():                               # embedder frozen in this phase
                h = E(x.to(device))
            loss = F.mse_loss(S(h)[:, :-1], h[:, 1:])           # guesses at 1..T-1 vs real steps 2..T
            opt.zero_grad(); loss.backward(); opt.step()
            hist.append(loss.item()); step += 1
            if step % 250 == 0:
                print(f"phase 2  step {step:5d}  supervised MSE {loss.item():.5f}", flush=True)
            if step >= steps:
                break

    with torch.no_grad():
        h = E(val_w.to(device))
        val_loss = F.mse_loss(S(h)[:, :-1], h[:, 1:]).item()
        persistence = F.mse_loss(h[:, :-1], h[:, 1:]).item()   # 'next step = this step'
    return hist, val_loss, persistence


def moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Match per-(step, feature) std and mean across the batch (original TimeGAN 'G_loss_V')."""
    return (x_hat.std(0) - x.std(0)).abs().mean() + (x_hat.mean(0) - x.mean(0)).abs().mean()


def train_joint(E, R, G, S, D, train_loader, device, steps: int = 2000, lr: float = 1e-3,
                z_dim: int = 40, gamma: float = 1.0, eval_fn=None, eval_every: int = 250) -> dict:
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
                    h = E(x)                                    # real latents: a target here, E not updated
                e_hat = G(noise(x))                             # raw fake latents
                h_hat = S(e_hat)                                # rolled forward by the learned dynamics
                x_hat = R(h_hat)                                # synthetic window
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
            if loss_d.item() > 0.15:                            # don't let D run away from G
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            for k, v in zip(hist, [loss_adv, loss_sup, loss_mom, loss_rec, loss_d]):
                hist[k].append(v.item())
            step += 1
            if step % 100 == 0:
                print(f"phase 3  step {step:5d}  D {loss_d.item():.3f}  G_adv {loss_adv.item():.3f}  "
                      f"G_sup {loss_sup.item():.5f}  G_mom {loss_mom.item():.3f}  E_rec {loss_rec.item():.3f}", flush=True)
            if eval_fn and step % eval_every == 0:
                eval_fn(step)
            if step >= steps:
                break
    return hist


# ----------------------------------------------------------------------------- selection and plots

def selection_score(val_w, fake_w, stats):
    """Validation score used to pick the best checkpoint (lower is better).

    Spec metrics (KL on spread and on returns) plus the fraction of books breaking an invariant.
    SSIM is skipped here for speed; the full report is produced by predict.py.
    """
    m = utils.evaluate(val_w, fake_w, stats, ssim=False)["fake"]
    score = m["KL spread vs real"] + m["KL returns vs real"] + m["Any invariant broken (%)"] / 100
    return score, m


def plot_losses(hists: dict, run: str, path: Path):
    """One panel per training stage, each loss curve labelled."""
    fig, axes = plt.subplots(1, len(hists), figsize=(5 * len(hists), 3.5), squeeze=False)
    for ax, (title, h) in zip(axes[0], hists.items()):
        for name, values in (h.items() if isinstance(h, dict) else [("loss", h)]):
            ax.plot(values, lw=0.8, label=name)
        ax.set(title=title, xlabel="Step", yscale="log")
        ax.legend(fontsize=8)
    fig.suptitle(run); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="folder containing the LOBSTER files (searched recursively)")
    p.add_argument("--model", choices=["rnngan", "timegan"], default="timegan")
    p.add_argument("--steps", type=int, default=2000,
                   help="TimeGAN: steps per phase. RNN-GAN: total steps (3x this, to match TimeGAN's total)")
    p.add_argument("--eval_every", type=int, default=250, help="score on validation every N steps")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 TimeGAN (paper), 2e-4 RNN-GAN")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default=None, help="name for outputs; default {model}_s{seed}")
    args = p.parse_args()

    run = args.run or f"{args.model}_s{args.seed}"
    lr = args.lr or (1e-3 if args.model == "timegan" else 2e-4)
    torch.manual_seed(args.seed)                                # reproducible runs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))           # set by torchrun; 0 for plain python
    device = (f"cuda:{local_rank}" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for d in ("checkpoints", "figures", "results"):
        Path(d).mkdir(exist_ok=True)
    print(f"run {run}  model {args.model}  device {device}  steps {args.steps}  lr {lr}", flush=True)

    train_loader, val_w, _, stats = get_data(args.data_dir)    # test split untouched until predict.py
    best = {"score": float("inf")}
    summary = {"run": run, "args": vars(args), "lr": lr}
    t0 = time.time()

    def save_if_best(step, nets, generate):
        """Score the current generator on validation; overwrite the checkpoint if it's the best so far."""
        torch.manual_seed(1234)                                 # same noise at every evaluation
        fake = generate().cpu()
        score, m = selection_score(val_w, fake, stats)
        flag = ""
        if score < best["score"]:
            best.update(score=score, step=step, metrics=m)
            torch.save({**{k: n.state_dict() for k, n in nets.items()},
                        "model": args.model, "stats": stats, "args": vars(args)}, f"checkpoints/{run}.pt")
            flag = "  <- best, saved"
        print(f"  eval step {step:5d}  score {score:.3f}  KL spread {m['KL spread vs real']:.3f}  "
              f"KL ret {m['KL returns vs real']:.3f}  invariants broken {m['Any invariant broken (%)']:.1f}%{flag}",
              flush=True)

    if args.model == "timegan":
        E, R, S = TGEmbedder().to(device), TGRecovery().to(device), TGSupervisor().to(device)
        G, D = TGGenerator().to(device), TGDiscriminator().to(device)
        nets = {"E": E, "R": R, "S": S, "G": G, "D": D}
        h1, val_rec = train_embedding(E, R, train_loader, val_w, device, args.steps, lr)
        print(f"phase 1 done: val recon MSE {val_rec:.4f}", flush=True)
        h2, val_sup, persistence = train_supervisor(E, S, train_loader, val_w, device, args.steps, lr)
        print(f"phase 2 done: val supervised MSE {val_sup:.5f} vs persistence {persistence:.5f}", flush=True)
        gen = lambda: tg_generate(G, S, R, len(val_w), device=device)
        h3 = train_joint(E, R, G, S, D, train_loader, device, args.steps, lr,
                         eval_fn=lambda s: save_if_best(s, nets, gen), eval_every=args.eval_every)
        summary.update(val_recon_mse=val_rec, val_supervised_mse=val_sup, persistence_mse=persistence)
        plot_losses({"Phase 1: reconstruction": h1, "Phase 2: supervised": h2, "Phase 3: joint": h3},
                    run, Path(f"figures/{run}_losses.png"))
    else:
        G, D = RNNGenerator().to(device), RNNDiscriminator().to(device)
        nets = {"G": G, "D": D}
        gen = lambda: rnngan_generate(G, len(val_w), device=device)
        h = train_rnngan(G, D, train_loader, device, 3 * args.steps, lr,
                         eval_fn=lambda s: save_if_best(s, nets, gen), eval_every=args.eval_every)
        plot_losses({"RNN-GAN": h}, run, Path(f"figures/{run}_losses.png"))

    # Quick-look figures from the best checkpoint, on VALIDATION (test stays untouched)
    ckpt = torch.load(f"checkpoints/{run}.pt", weights_only=False)
    for k, n in nets.items():
        n.load_state_dict(ckpt[k])                              # restore the best weights, not the last ones
    torch.manual_seed(1234)
    fake = gen().cpu()
    utils.plot_paths(val_w, fake, stats, label=run, path=f"figures/{run}_val_paths.png")
    utils.plot_distributions(val_w, fake, stats, label=run, path=f"figures/{run}_val_dists.png")
    utils.plot_book_comparison(val_w, fake, stats, label=run, path=f"figures/{run}_val_book.png")
    plt.close("all")
    
    summary.update(
        train_seconds=round(time.time() - t0, 1),
        params={k: sum(p.numel() for p in n.parameters()) for k, n in nets.items()},
        peak_vram_mb=(round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.startswith("cuda") else None),
        best_step=best.get("step"), best_score=best["score"],
        best_val_metrics={k: float(v) for k, v in best.get("metrics", {}).items()},
    )
    Path(f"results/{run}.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {summary['train_seconds']}s  best step {summary['best_step']}  best score {best['score']:.3f}  "
          f"params {sum(summary['params'].values()):,}  peak VRAM {summary['peak_vram_mb']} MB", flush=True)


if __name__ == "__main__":
    main()