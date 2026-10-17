"""
utils.py - evaluation metric and plotting helpers shared by train.py and predict.py.
"""
import matplotlib
matplotlib.use("Agg")  # Rangpur compute nodes have no display; save figures to file
import matplotlib.pyplot as plt
import torch

CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]


@torch.no_grad()
def dice_per_class(logits, target, n_classes=6, eps=1e-6):
    """Hard (argmax) Dice for each class: the number we report, unlike the soft Dice in the loss.

    Args:
        logits: (B, C, X, Y, Z) model output.
        target: (B, X, Y, Z) int64 labels.
        n_classes: number of classes.
        eps: smoothing; a class absent from both prediction and target scores 1.

    Returns:
        (C,) tensor of Dice scores, one per class, pooled over the batch
        (pass one volume at a time for per-volume Dice).
    """
    pred = logits.argmax(dim=1)  # most likely class at each voxel
    scores = []
    for c in range(n_classes):
        # Dice = 2|P ∩ T| / (|P| + |T|): 1 = perfect overlap, 0 = none
        p, t = pred == c, target == c
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        scores.append((2 * inter + eps) / (denom + eps))
    return torch.stack(scores)


def plot_history(history, path):
    """Plot train/val loss and per-class val Dice against epoch, and save to path.

    Args:
        history: dict with "train_loss", "val_loss" (lists) and "val_dice" (list of per-class lists).
        path: output .png path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set(xlabel="epoch", ylabel="Dice + CE loss", title="Loss")
    ax_loss.legend()
    for c, name in enumerate(CLASS_NAMES[1:], start=1):  # skip background
        ax_dice.plot(epochs, [d[c] for d in history["val_dice"]], label=name)
    ax_dice.set(xlabel="epoch", ylabel="Dice", ylim=(0, 1), title="Validation Dice per class")
    ax_dice.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)  # free the figure: called every epoch, so open figures would pile up in memory