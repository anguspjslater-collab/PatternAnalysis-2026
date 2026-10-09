"""
predict.py: final, one-time evaluation on the held-out TEST hour (14:45-15:45).

Loads saved checkpoints (no training), generates synthetic order-book windows and produces
everything the report's results section needs (spec section 2.7):

  1. Metrics on the test set, mean +/- std over 3 seeds per model:
     KL divergence on spread and returns (target <= 0.1), invariant audit (crossed books,
     ladder order, negative volume), depth-heatmap SSIM (target > 0.6), zero returns,
     volatility clustering, trend and diversity.
     Two reference columns: the real test statistics, and real VALIDATION windows scored as if
     they were a generator ("real val (ref)"). That row is the best score an ideal generator
     could expect, because the market itself drifts between hours.
  2. Resources: parameters, training time and peak VRAM (from each run's training JSON),
     generation throughput (1,000 windows).
  3. Figures: 4 uncurated real-vs-synthetic books and depth heatmaps for the main model
     (raw and tick-snapped), one book per other model, return paths and distributions.

Test data is used here and nowhere else (all tuning and checkpoint selection used validation).
Figures always use seed 0 of each model and randomly chosen windows, so nothing is cherry-picked.
Noise is drawn on the CPU from a fixed generator, so the samples are identical on any machine
(CUDA and CPU random streams differ, even with the same seed).

Usage:
    python predict.py --data_dir ~/Desktop/LOBSTER
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # write figures to files, no windows
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

import utils
from dataset import get_data
from modules import RNNGenerator, TGGenerator, TGSupervisor, TGRecovery, ConstrainedHead

# Models compared on the test set. {seed} is filled with 0, 1, 2.
# The main model was chosen on VALIDATION (Trial 3); the others are the baselines and controls.
RUNS = {
    "TimeGAN + head (ae10k)":    "Trial_3_results/checkpoints/timegan_con_ae10k_s{seed}.pt",
    "TimeGAN + head (trial 2)":  "Trial_2_results/checkpoints/timegan_con_s{seed}.pt",
    "TimeGAN, no head":          "Trial_2_results/checkpoints/timegan_s{seed}.pt",
    "RNN-GAN + head (baseline)": "Trial_2_results/checkpoints/rnngan_con_s{seed}.pt",
    "RNN-GAN, no head":          "Trial_2_results/checkpoints/rnngan_s{seed}.pt",
}
MAIN = "TimeGAN + head (ae10k)"
SEEDS = (0, 1, 2)
Z_DIM = 40                    # noise dimension used in training for both models
N_BOOKS = 4                   # spec asks for 3-5 real-vs-synthetic heatmaps
TARGETS = {                   # spec section 2.7 targets: (metric, direction, threshold)
    "KL spread vs real": ("<=", 0.1),
    "KL returns vs real": ("<=", 0.1),
    "Heatmap SSIM vs real (best match)": (">", 0.6),
    "Any invariant broken (%)": ("<=", 1.0),
}


def slug(name: str) -> str:
    """Model name -> safe file-name fragment."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def constrain(net: nn.Module, stats: dict) -> nn.Module:
    """Append the ConstrainedHead exactly as train.py did (same positive and volume columns)."""
    cols = list(stats["mean"].index)
    pos = [cols.index("spread")] + [i for i, c in enumerate(cols) if "gap" in c]
    vol = [i for i, c in enumerate(cols) if "logv" in c]
    head = ConstrainedHead(torch.tensor(stats["mean"].to_numpy()), torch.tensor(stats["std"].to_numpy()), pos, vol)
    return nn.Sequential(net, head)


def load_generator(path: Path, device: str):
    """Rebuild the generator side of a saved model and load its weights.

    TimeGAN generates with Recovery(Supervisor(Generator(z))); the RNN-GAN with Generator(z).
    The embedder and discriminators are not needed to generate, so they are not loaded.
    Returns (sample_fn, training args, number of generator-side parameters).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)   # checkpoints were saved on a GPU
    a, stats = ck["args"], ck["stats"]
    n_feat = len(stats["mean"])
    if ck["model"] == "timegan":
        nets = {"G": TGGenerator(), "S": TGSupervisor(), "R": TGRecovery(n_features=n_feat)}
        if a.get("constrained"):
            nets["R"] = constrain(nets["R"], stats)
    else:
        nets = {"G": RNNGenerator(n_features=n_feat)}
        if a.get("constrained"):
            nets["G"] = constrain(nets["G"], stats)
    for k, net in nets.items():
        net.load_state_dict(ck[k])
        net.to(device).eval()
    n_params = sum(p.numel() for net in nets.values() for p in net.parameters())

    @torch.no_grad()
    def sample(n: int, seq_len: int, seed: int = 1234) -> torch.Tensor:
        """n normalised windows (n, seq_len, features), from noise drawn on the CPU."""
        g = torch.Generator().manual_seed(seed)
        if ck["model"] == "timegan":
            z = torch.rand(n, seq_len, Z_DIM, generator=g).to(device)      # uniform noise, as in training
            return nets["R"](nets["S"](nets["G"](z))).cpu()
        z = torch.randn(n, seq_len, Z_DIM, generator=g).to(device)         # Gaussian noise, as in training
        return nets["G"](z).cpu()

    return sample, a, n_params


def generation_seconds(sample, seq_len: int, n: int = 1000) -> float:
    """Wall-clock time to generate n windows, after one warm-up call."""
    sample(64, seq_len)
    t = time.perf_counter()
    sample(n, seq_len)              # .cpu() inside sample waits for the GPU to finish
    return time.perf_counter() - t


def training_info(ckpt_path: Path) -> dict:
    """Training time and peak VRAM from the JSON that train.py wrote next to the checkpoint."""
    js = ckpt_path.parent.parent / "results" / f"{ckpt_path.stem}.json"
    if not js.exists():
        return {}
    d = json.loads(js.read_text())
    return {"train time (s)": d.get("train_seconds"), "peak VRAM (MB)": d.get("peak_vram_mb")}


def main():
    p = argparse.ArgumentParser(description="Final test-set evaluation of saved generators.")
    p.add_argument("--data_dir", required=True, help="folder containing the LOBSTER sample files")
    p.add_argument("--out", default="predict_results", help="where tables and figures are written")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    out = Path(args.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    _, val_w, test_w, stats = get_data(args.data_dir)      # 24-step, Level-10 windows: the task the spec defines
    seq_len = test_w.shape[1]
    print(f"code {commit}  device {args.device}")
    print(f"test windows {tuple(test_w.shape)} (14:45-15:45, used only in this script); "
          f"reference: validation windows {tuple(val_w.shape)}\n")

    rows, resources, samples = [], [], {}
    real_row = ref_row = None
    for name, pattern in RUNS.items():
        for seed in SEEDS:
            path = Path(pattern.format(seed=seed))
            if not path.exists():
                print(f"  skipped (checkpoint missing): {path}")
                continue
            sample, a, n_params = load_generator(path, args.device)
            if a.get("seq_len", 24) != seq_len or a.get("level", 10) != 10:
                raise ValueError(f"{path} was trained on a different window length or depth")
            fake = sample(len(test_w), seq_len)
            m = utils.evaluate(test_w, fake, stats, ref_w=val_w)          # includes heatmap SSIM
            rows.append({"model": name, "seed": seed, **{k: float(v) for k, v in m["fake"].items()}})
            real_row, ref_row = m["real"], m["ref"]                       # identical for every model
            if seed == 0:
                samples[name] = fake
                resources.append({"model": name, "parameters (generator side)": n_params,
                                  **training_info(path),
                                  "generate 1,000 windows (s)": round(generation_seconds(sample, seq_len), 4)})
                if name == MAIN:
                    print(f"Full report, {name}, seed 0 (test set):")
                    utils.print_report(m, names=("real (test)", "generated", "real val (ref)"))
                    print()
            print(f"  evaluated {name:28s} seed {seed}")

    # ---------------------------------------------------------------- tables
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_metrics_per_seed.csv", index=False)
    metrics = [c for c in df.columns if c not in ("model", "seed")]
    grouped = df.groupby("model", sort=False)[metrics]
    table = (grouped.mean().round(3).astype(str) + " ± " + grouped.std().round(3).astype(str)).T
    table["real test"] = pd.Series({k: round(float(v), 3) for k, v in real_row.items()})
    table["real val (ref)"] = pd.Series({k: round(float(v), 3) for k, v in ref_row.items()})
    table.to_csv(out / "test_metrics_summary.csv")
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 16):
        print("\nTEST SET, mean ± std over 3 seeds (real val (ref) = best score an ideal generator could expect):")
        print(table.fillna("-"))

    res = pd.DataFrame(resources).set_index("model")
    res.to_csv(out / "resources.csv")
    print("\nResources (seed 0; training numbers from the training JSONs, generation on", args.device + "):")
    print(res.to_string())

    print(f"\nSpec targets, {MAIN} (mean of 3 seeds) vs the real-vs-real reference:")
    main_mean = grouped.mean().loc[MAIN]
    for metric, (op, thr) in TARGETS.items():
        v = main_mean[metric]
        ok = v <= thr if op == "<=" else v > thr
        ref = ref_row.get(metric, float("nan"))
        print(f"  {metric:36s} {v:7.3f}  target {op} {thr:<4}  {'PASS' if ok else 'MISS'}   (real vs real: {float(ref):.3f})")

    # ---------------------------------------------------------------- figures
    g = torch.Generator().manual_seed(0)                   # random, reproducible window choice (no cherry-picking)
    real_idx = torch.randperm(len(test_w), generator=g)[:N_BOOKS].tolist()
    fake_idx = torch.randperm(len(test_w), generator=g)[:N_BOOKS].tolist()
    main_fake = samples[MAIN]
    main_snapped = utils.snap_windows(main_fake, stats)
    for k, (i, j) in enumerate(zip(real_idx, fake_idx)):
        utils.plot_book_comparison(test_w, main_fake, stats, i=i, j=j, label=MAIN,
                                   path=fig_dir / f"book_{slug(MAIN)}_{k}.png")
        utils.plot_book_comparison(test_w, main_snapped, stats, i=i, j=j, label=f"{MAIN}, snapped",
                                   path=fig_dir / f"book_{slug(MAIN)}_{k}_snapped.png")
        plt.close("all")
    for name, fake in samples.items():
        s = slug(name)
        utils.plot_paths(test_w, fake, stats, label=name, path=fig_dir / f"paths_{s}.png")
        utils.plot_distributions(test_w, fake, stats, label=name, path=fig_dir / f"dists_{s}.png")
        if name != MAIN:                                   # same real window as the main model's book 0
            utils.plot_book_comparison(test_w, fake, stats, i=real_idx[0], j=fake_idx[0], label=name,
                                       path=fig_dir / f"book_{s}_0.png")
        plt.close("all")

    (out / "run_info.json").write_text(json.dumps(
        {"commit": commit, "device": args.device, "seeds": SEEDS, "runs": RUNS, "main": MAIN,
         "book_windows": {"real": real_idx, "generated": fake_idx}}, indent=2))
    print(f"\nwrote tables and {len(list(fig_dir.glob('*.png')))} figures to {out}/")


if __name__ == "__main__":
    main()