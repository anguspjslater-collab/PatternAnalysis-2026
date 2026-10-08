"""Evaluation and visualisation for generated LOB windows (COMP3710 TimeGAN project).

Every function takes windows of shape (N, 24, 40) in NORMALISED units (as produced by
dataset.get_data and by the generators) plus the training `stats`, and converts them to
real units internally. Every model (baseline, TimeGAN, constrained or raw) is therefore
scored by identical code.

Metrics follow spec §2.7: invariant violations, KL divergence on spread and returns,
depth-heatmap SSIM. Added on top: zero-return share, KL on non-zero returns, volatility
clustering, drift and diversity, so a degenerate generator cannot pass on KL alone.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

TICK = 100                                    # one cent in LOBSTER units (USD x 10,000)
SPREAD_EDGES = np.arange(-0.5, 60.5, 1.0)     # one bin per whole tick; spreads <= 0 land in the first bin


# ----------------------------------------------------------------------------- conversion

def unnormalise(w: torch.Tensor, stats: dict) -> torch.Tensor:
    """Undo the train-statistics z-score: normalised windows -> real units."""
    mean = torch.tensor(stats["mean"].to_numpy(), dtype=torch.float32)
    std = torch.tensor(stats["std"].to_numpy(), dtype=torch.float32)
    return w * std + mean


def feature_index(stats: dict) -> dict:
    """Column positions of each feature group, read from the stats index (no hard-coding)."""
    cols = list(stats["mean"].index)
    levels = sum(c.startswith("ask_logv") for c in cols)
    return {
        "ret": cols.index("ret"),
        "spread": cols.index("spread"),
        "ask_gap": [cols.index(f"ask_gap{i}") for i in range(1, levels)],
        "bid_gap": [cols.index(f"bid_gap{i}") for i in range(1, levels)],
        "ask_v": [cols.index(f"ask_logv{i}") for i in range(1, levels + 1)],
        "bid_v": [cols.index(f"bid_logv{i}") for i in range(1, levels + 1)],
    }


def to_book(x: torch.Tensor, ix: dict, mid0: float = 2_220_000.0):
    """Real-unit features (..., T, 40) -> ask/bid prices (..., T, levels) in LOBSTER units, plus share sizes.

    mid is rebuilt from cumulative log returns starting at mid0 (an anchor price; the model has
    no notion of price level). Best ask/bid = mid +/- spread/2, outer levels add the gaps.
    Nothing is rounded or clipped, so violations stay visible.
    """
    mid = mid0 * torch.exp(torch.cumsum(x[..., ix["ret"]], dim=-1))
    half = x[..., ix["spread"]] * TICK / 2
    zero = torch.zeros(*x.shape[:-1], 1)
    asks = (mid + half)[..., None] + TICK * torch.cat([zero, torch.cumsum(x[..., ix["ask_gap"]], -1)], -1)
    bids = (mid - half)[..., None] - TICK * torch.cat([zero, torch.cumsum(x[..., ix["bid_gap"]], -1)], -1)
    return asks, bids, torch.expm1(x[..., ix["ask_v"]]), torch.expm1(x[..., ix["bid_v"]])


def heatmaps(x: torch.Tensor, ix: dict) -> torch.Tensor:
    """Depth heatmaps (N, 2*levels, T): rows Ask_max ... Ask1, Bid1 ... Bid_max; values log(1+size)."""
    rows = ix["ask_v"][::-1] + ix["bid_v"]
    return x[..., rows].transpose(-1, -2)


# ----------------------------------------------------------------------------- metrics

def invariant_rates(x: torch.Tensor, ix: dict) -> dict:
    """% of books (one per window step) breaking each invariant, measured BEFORE any rounding/clipping."""
    gaps = x[..., ix["ask_gap"] + ix["bid_gap"]]
    crossed = x[..., ix["spread"]] <= 0                        # best bid >= best ask
    disorder = (gaps <= 0).any(-1)                             # a level at or past its neighbour
    sub_tick = (gaps < 1 - 1e-3).any(-1)                       # tolerance: float round-trip turns 1.0 into 0.99999                              # gap smaller than the 1-tick minimum
    neg_vol = (x[..., ix["ask_v"] + ix["bid_v"]] < 0).any(-1)  # log(1+size) < 0  <=>  size < 0
    pct = lambda m: 100 * m.float().mean().item()
    return {"Crossed books": pct(crossed), "Ladder out of order": pct(disorder),
            "Gap < 1 tick": pct(sub_tick), "Negative volume": pct(neg_vol),
            "Any invariant broken": pct(crossed | disorder | neg_vol)}


def _ret_edges(width_bps: float, span_bps: float = 10.0) -> np.ndarray:
    """Return bins centred on multiples of width_bps (about half a tick), covering +/- span_bps."""
    k = math.ceil(span_bps / width_bps)
    return (np.arange(-k, k + 2) - 0.5) * width_bps


def _kl(p_samples: np.ndarray, q_samples: np.ndarray, edges: np.ndarray, eps: float = 1e-6) -> float:
    """KL(P || Q) between two sample sets on shared bins. Outliers clip into the edge bins; eps avoids log(0)."""
    lo, hi = edges[0] + 1e-9, edges[-1] - 1e-9
    p, _ = np.histogram(np.clip(p_samples, lo, hi), edges)
    q, _ = np.histogram(np.clip(q_samples, lo, hi), edges)
    p, q = p / p.sum() + eps, q / q.sum() + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def abs_return_acf(r_bps: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Autocorrelation of |returns| at lags 1..max_lag, pooled within windows (volatility clustering)."""
    x = np.abs(r_bps) - np.abs(r_bps).mean()
    var = (x ** 2).mean()
    return np.array([(x[:, :-k] * x[:, k:]).mean() / var for k in range(1, max_lag + 1)])


def _ssim(a: torch.Tensor, b: torch.Tensor, data_range: float, k: int = 7) -> torch.Tensor:
    """Mean SSIM per pair of (P, H, W) images using a k x k uniform window."""
    a, b = a.unsqueeze(1), b.unsqueeze(1)
    pool = lambda t: F.avg_pool2d(t, k, stride=1)
    mu_a, mu_b = pool(a), pool(b)
    var_a, var_b = pool(a * a) - mu_a ** 2, pool(b * b) - mu_b ** 2
    cov = pool(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.mean(dim=(1, 2, 3))


def best_match_ssim(fake_h: torch.Tensor, real_h: torch.Tensor, n: int = 150, seed: int = 0) -> float:
    """For n random generated heatmaps, SSIM to the most similar of ALL real heatmaps; averaged.

    Generation is unpaired (no 'true' partner window), so each sample is compared with its best match.
    Only the generated side is subsampled, for speed; every real window is a candidate match.
    """
    g = torch.Generator().manual_seed(seed)
    f = fake_h[torch.randperm(len(fake_h), generator=g)[:n]]
    rng = float(real_h.max() - real_h.min())
    best = [_ssim(fi.unsqueeze(0).expand(len(real_h), -1, -1), real_h, rng).max() for fi in f]
    return float(torch.stack(best).mean())


def summarise(x: torch.Tensor, ix: dict, ret_bin_bps: float) -> dict:
    """Descriptive statistics of one set of real-unit windows."""
    r = x[..., ix["ret"]].numpy() * 1e4                         # returns in basis points, (N, T)
    s = x[..., ix["spread"]].numpy()
    return {
        "Spread mean (ticks)": s.mean(),
        "Spread std (ticks)": s.std(),
        "5s return std (bps)": r.std(),
        "Mean 5s return (bps)": r.mean(),
        "Windows trending up (%)": 100 * (r.sum(1) > 0).mean(),
        "Zero returns (%)": 100 * (np.abs(r) < ret_bin_bps / 2).mean(),
        "Diversity: std of window-mean spread": s.mean(1).std(),
        "Vol clustering: ACF |ret| lag 1": abs_return_acf(r)[0],
    }


def evaluate(real_w, fake_w, stats, ref_w=None, ret_bin_bps: float = 0.225) -> dict:
    """Score generated windows against real ones. All inputs are normalised (N, 24, 40) tensors.

    real_w: the comparison period (validation during development, test for final results).
    ref_w:  optional second real period (e.g. train). Scored like a generator, it gives the
            real-vs-real level to expect, separating model error from market drift.
    ret_bin_bps: return bin width, about half a tick at AMZN's price (0.005 / 222 = 0.225 bps).
    """
    ix = feature_index(stats)
    sets = {"real": unnormalise(real_w, stats), "fake": unnormalise(fake_w, stats)}
    if ref_w is not None:
        sets["ref"] = unnormalise(ref_w, stats)

    out = {k: summarise(v, ix, ret_bin_bps) for k, v in sets.items()}
    for k, v in sets.items():
        out[k].update({f"{name} (%)": val for name, val in invariant_rates(v, ix).items()})

    edges = _ret_edges(ret_bin_bps)
    real_r = sets["real"][..., ix["ret"]].numpy().ravel() * 1e4
    real_s = sets["real"][..., ix["spread"]].numpy().ravel()
    real_h = heatmaps(sets["real"], ix)
    for k in [k for k in sets if k != "real"]:
        r = sets[k][..., ix["ret"]].numpy().ravel() * 1e4
        s = sets[k][..., ix["spread"]].numpy().ravel()
        nz = np.abs(r) >= ret_bin_bps / 2
        out[k]["KL spread vs real"] = _kl(real_s, s, SPREAD_EDGES)
        out[k]["KL returns vs real"] = _kl(real_r, r, edges)
        out[k]["KL non-zero returns vs real"] = (
            _kl(real_r[np.abs(real_r) >= ret_bin_bps / 2], r[nz], edges) if nz.any() else float("nan"))
        out[k]["Heatmap SSIM vs real (best match)"] = best_match_ssim(heatmaps(sets[k], ix), real_h)
    return out


NOTES = {
    "Mean 5s return (bps)": "expect ~0 (no drift)",
    "Windows trending up (%)": "expect ~50",
    "Diversity: std of window-mean spread": "fake << real = mode collapse",
    "Vol clustering: ACF |ret| lag 1": "> 0 = clustering",
    "Crossed books (%)": "real is 0",
    "Ladder out of order (%)": "real is 0",
    "Gap < 1 tick (%)": "real is 0",
    "Negative volume (%)": "real is 0",
    "Any invariant broken (%)": "AC1: <= 1",
    "KL spread vs real": "target <= 0.1",
    "KL returns vs real": "target <= 0.1",
    "KL non-zero returns vs real": "size of real moves",
    "Heatmap SSIM vs real (best match)": "target > 0.6",
}


def print_report(out: dict, names=("real (val)", "generated", "ref (train)")) -> None:
    """Print the evaluate() results as a labelled table, one row per metric."""
    keys = [k for k in ("real", "fake", "ref") if k in out]
    print(f"{'Metric':<40}" + "".join(f"{n:>13}" for n in names[:len(keys)]) + "   note")
    print("-" * (40 + 13 * len(keys) + 25))
    for metric in out["fake"]:
        vals = "".join(f"{out[k][metric]:>13.3f}" if metric in out[k] else f"{'-':>13}" for k in keys)
        print(f"{metric:<40}{vals}   {NOTES.get(metric, '')}")


# ----------------------------------------------------------------------------- visualisation

def plot_book_comparison(real_w, fake_w, stats, i: int = 0, j: int = 0, mid0: float = 2_220_000.0,
                         label: str = "Generated", path=None):
    """2x2 figure: price ladder (top) and depth heatmap (bottom) for real window i vs generated window j."""
    ix = feature_index(stats)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, (name, w) in enumerate([("Real", unnormalise(real_w[i], stats)),
                                     (label, unnormalise(fake_w[j], stats))]):
        asks, bids, _, _ = to_book(w, ix, mid0)
        for lvl in range(asks.shape[-1]):          # ladder lines should never touch or cross
            axes[0, col].plot(asks[:, lvl] / 1e4, color="tab:red", lw=0.8, alpha=1 - lvl * 0.08)
            axes[0, col].plot(bids[:, lvl] / 1e4, color="tab:blue", lw=0.8, alpha=1 - lvl * 0.08)
        axes[0, col].set(title=f"{name}: price ladder", ylabel="Price ($)")
        im = axes[1, col].imshow(heatmaps(w, ix), aspect="auto", cmap="viridis", vmin=0, vmax=9)
        axes[1, col].axhline(asks.shape[-1] - 0.5, color="red", lw=1)     # spread line
        axes[1, col].set(title=f"{name}: depth log(1+size)", xlabel="Step (5 s)", ylabel="Ask10 -> Bid10")
    fig.colorbar(im, ax=axes[1, :], shrink=0.8)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_paths(real_w, fake_w, stats, n: int = 30, seed: int = 0, label: str = "Generated", path=None):
    """Cumulative mid-return paths of n random (uncurated) real and generated windows."""
    ix = feature_index(stats)
    g = torch.Generator().manual_seed(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, (name, w) in zip(axes, [("Real", real_w), (label, fake_w)]):
        pick = torch.randperm(len(w), generator=g)[:n]
        paths = torch.cumsum(unnormalise(w[pick], stats)[..., ix["ret"]], 1) * 1e4
        ax.plot(paths.T.numpy(), lw=0.8, alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set(title=f"{name}: {n} random windows", xlabel="Step (5 s)")
    axes[0].set_ylabel("Cumulative return (bps)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_distributions(real_w, fake_w, stats, ret_bin_bps: float = 0.225, label: str = "Generated", path=None):
    """Spread histogram, return histogram (log scale) and |return| autocorrelation, real vs generated."""
    ix = feature_index(stats)
    r_real, r_fake = unnormalise(real_w, stats), unnormalise(fake_w, stats)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.5))
    for x, name, c in [(r_real, "Real", "C0"), (r_fake, label, "C1")]:
        a1.hist(x[..., ix["spread"]].numpy().ravel(), bins=SPREAD_EDGES, density=True,
                histtype="step", lw=1.5, color=c, label=name)
        a2.hist(x[..., ix["ret"]].numpy().ravel() * 1e4, bins=_ret_edges(ret_bin_bps), density=True,
                histtype="step", lw=1.5, color=c, label=name)
        a3.plot(range(1, 11), abs_return_acf(x[..., ix["ret"]].numpy() * 1e4), "o-", color=c, label=name)
    a1.set(xlabel="Spread (ticks)", ylabel="Density", title="Spread distribution")
    a2.set(xlabel="5 s mid return (bps)", title="Return distribution", yscale="log")
    a3.axhline(0, color="black", lw=0.5)
    a3.set(xlabel="Lag (5 s steps)", ylabel="ACF of |return|", title="Volatility clustering")
    for a in (a1, a2, a3):
        a.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig