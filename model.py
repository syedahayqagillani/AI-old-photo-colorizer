"""model.py - AI Old Photo Colorizer

U-Net (ResNet18 encoder) jo LAB colour space me kaam karta hai.
  Input  : L channel  (grayscale image ki brightness)
  Output : a, b channels (colour information)
Final image = L (original) + predicted (a, b)  ->  RGB
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

IMG_SIZE = 256


# ----------------------------------------------------------------------
# 1) Colour helpers
# ----------------------------------------------------------------------
def _make_lut():
    """Gray pixel value (0-255) -> L value (0-1). Fast lookup table."""
    g = np.arange(256, dtype=np.float32) / 255.0
    rgb = np.stack([g, g, g], axis=-1).reshape(1, 256, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return (lab[0, :, 0] / 100.0).astype(np.float32)


GRAY_TO_L = _make_lut()


def gray_to_L(gray_uint8):
    """HxW uint8 gray -> HxW float32 L in [0,1]"""
    return GRAY_TO_L[gray_uint8]


def rgb_to_lab(rgb_uint8):
    """HxWx3 uint8 RGB -> (L in [0,1], ab in [-1,1])"""
    lab = cv2.cvtColor(rgb_uint8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return lab[..., 0] / 100.0, (lab[..., 1:] / 128.0).astype(np.float32)


def lab_to_rgb(L, ab):
    """L: HxW in [0,1], ab: HxWx2 in [-1,1] -> HxWx3 uint8 RGB"""
    lab = np.concatenate([(L * 100.0)[..., None], ab * 128.0], axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# 2) Dataset (training ke liye)
# ----------------------------------------------------------------------
class ColorDataset(torch.utils.data.Dataset):
    """Input = gray image ka L channel, Target = color image ke (a,b) channels."""

    def __init__(self, gray_arr, color_arr, idx, augment=False):
        self.g, self.c = gray_arr, color_arr
        self.idx = np.asarray(idx)
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        g, c = self.g[i], self.c[i]
        if self.augment and torch.rand(1).item() < 0.5:  # random horizontal flip
            g, c = g[:, ::-1], c[:, ::-1]
        L = GRAY_TO_L[np.ascontiguousarray(g)]
        _, ab = rgb_to_lab(np.ascontiguousarray(c))
        return (torch.from_numpy(L).unsqueeze(0),
                torch.from_numpy(ab).permute(2, 0, 1).contiguous())


# ----------------------------------------------------------------------
# 3) Model: U-Net with pretrained ResNet18 encoder
# ----------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)


class UNetResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # H/2,  64
        self.pool = base.maxpool
        self.layer1 = base.layer1  # H/4,  64
        self.layer2 = base.layer2  # H/8,  128
        self.layer3 = base.layer3  # H/16, 256
        self.layer4 = base.layer4  # H/32, 512
        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.dec0 = ConvBlock(64 + 3, 32)
        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, L):
        # L: (B,1,H,W) values in [0,1]
        x = L.repeat(1, 3, 1, 1)
        xn = (x - 0.45) / 0.225  # ImageNet style normalisation
        e0 = self.layer0(xn)
        e1 = self.layer1(self.pool(e0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        d = self.dec4(torch.cat([_up(e4, e3), e3], 1))
        d = self.dec3(torch.cat([_up(d, e2), e2], 1))
        d = self.dec2(torch.cat([_up(d, e1), e1], 1))
        d = self.dec1(torch.cat([_up(d, e0), e0], 1))
        d = self.dec0(torch.cat([_up(d, x), x], 1))
        return torch.tanh(self.out(d))  # ab in [-1,1]


# ----------------------------------------------------------------------
# 4) Load + Inference
# ----------------------------------------------------------------------
def load_model(path="weights/colorizer.pth", device="cpu"):
    model = UNetResNet18(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def colorize_image(model, pil_img, device="cpu", strength=1.0, max_side=1024):
    """Koi bhi PIL image (B&W ya color) -> colorized RGB numpy array (uint8)."""
    g = np.array(pil_img.convert("L"))
    h, w = g.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        g = cv2.resize(g, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = g.shape
    L_full = gray_to_L(g)  # original resolution brightness (sharp)
    g_small = cv2.resize(g, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(gray_to_L(g_small))[None, None].to(device)
    ab_small = model(x)[0].permute(1, 2, 0).cpu().numpy()
    ab = cv2.resize(ab_small, (w, h), interpolation=cv2.INTER_CUBIC) * strength
    ab = np.clip(ab, -1, 1)
    return lab_to_rgb(L_full, ab)
