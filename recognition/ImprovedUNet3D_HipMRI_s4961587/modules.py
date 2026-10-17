"""
modules.py - 3D U-Net segmentation model and loss for HipMRI prostate MRI.

UNet3D: baseline from Cicek et al., 2016, "3D U-Net: Learning Dense Volumetric
Segmentation from Sparse Annotation" (MICCAI). Four 2x down/up-sampling levels,
16-32-64-128 channels with a 256-channel bottleneck.
ImprovedUNet3D (Isensee et al., 2017) will be added here later.

Input:  (B, 1, X, Y, Z), each spatial size divisible by 16 (we use 128x128x64).
Output: (B, 6, X, Y, Z) raw logits; softmax is applied in the loss and at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two [3x3x3 conv -> InstanceNorm -> ReLU] layers: the basic U-Net block.

    Args:
        in_ch: input channels.
        out_ch: output channels.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # bias=False: the norm layer subtracts the mean, which would cancel a bias anyway
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """(B, in_ch, X, Y, Z) -> (B, out_ch, X, Y, Z); spatial size unchanged (padding=1)."""
        return self.block(x)


class UNet3D(nn.Module):
    """Baseline 3D U-Net (Cicek et al., 2016).

    Args:
        in_ch: input channels (1 for single-sequence MRI).
        n_classes: output classes (6: background + 5 structures).
        base: channels at the first level; doubled at each level down.
    """

    def __init__(self, in_ch=1, n_classes=6, base=16):
        super().__init__()
        c1, c2, c3, c4, c5 = (base * 2**i for i in range(5))  # 16, 32, 64, 128, 256

        self.pool = nn.MaxPool3d(2)
        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        # Each up step: transposed conv doubles the size and halves the channels,
        # then DoubleConv merges it with the skip (so its input is 2x channels).
        self.up4, self.dec4 = nn.ConvTranspose3d(c5, c4, 2, stride=2), DoubleConv(c5, c4)
        self.up3, self.dec3 = nn.ConvTranspose3d(c4, c3, 2, stride=2), DoubleConv(c4, c3)
        self.up2, self.dec2 = nn.ConvTranspose3d(c3, c2, 2, stride=2), DoubleConv(c3, c2)
        self.up1, self.dec1 = nn.ConvTranspose3d(c2, c1, 2, stride=2), DoubleConv(c2, c1)

        self.out = nn.Conv3d(c1, n_classes, kernel_size=1)  # per-voxel class scores

    def forward(self, x):
        """(B, 1, X, Y, Z) -> (B, n_classes, X, Y, Z) logits."""
        e1 = self.enc1(x)                       # (B, 16, 128, 128, 64)
        e2 = self.enc2(self.pool(e1))           # (B, 32,  64,  64, 32)
        e3 = self.enc3(self.pool(e2))           # (B, 64,  32,  32, 16)
        e4 = self.enc4(self.pool(e3))           # (B, 128, 16,  16,  8)
        b = self.bottleneck(self.pool(e4))      # (B, 256,  8,   8,  4)

        # Skip connections: concatenate along the channel dimension.
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class DiceCELoss(nn.Module):
    """Cross-entropy + (1 - soft Dice averaged over classes).

    Dice weights every class equally regardless of size, so the 0.1% prostate
    matters as much as the background; CE gives stable gradients early on.
    Soft Dice loss from Milletari et al., 2016, "V-Net" (3DV).

    Args:
        n_classes: number of classes.
        eps: smoothing term so empty classes don't divide by zero.
    """

    def __init__(self, n_classes=6, eps=1e-5):
        super().__init__()
        self.n_classes, self.eps = n_classes, eps

    def forward(self, logits, target):
        """logits: (B, C, X, Y, Z) float; target: (B, X, Y, Z) int64 labels. Returns a scalar loss."""
        logits = logits.float()  # compute the loss in fp32 even under bf16 autocast
        ce = F.cross_entropy(logits, target)

        probs = torch.softmax(logits, dim=1)                                   # class probabilities
        onehot = F.one_hot(target, self.n_classes).permute(0, 4, 1, 2, 3).float()  # (B,X,Y,Z,C) -> (B,C,X,Y,Z)
        dims = (0, 2, 3, 4)                                                        # sum over batch + space
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)                         # one soft Dice per class
        return ce + (1 - dice.mean())


if __name__ == "__main__":
    # Shape test on CPU: random input -> correct output shape, parameter count, loss runs.
    model = UNet3D()
    x = torch.randn(1, 1, 128, 128, 64)
    y = model(x)
    print("output", tuple(y.shape))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    target = torch.randint(0, 6, (1, 128, 128, 64))
    print(f"loss on random data: {DiceCELoss()(y, target).item():.3f}")