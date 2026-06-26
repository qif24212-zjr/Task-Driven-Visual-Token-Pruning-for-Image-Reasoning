import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# =========================
# Predictor
# =========================
class ImportancePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):

        score = self.mlp(x)

        return score.squeeze(-1)


# =========================
# paths
# =========================

FEATURE_DIR = "/root/autodl-tmp/datasets/mini_coco/features"
LABEL_DIR = "/root/autodl-tmp/datasets/mini_coco/labels"

MODEL_PATH = "predictor.pth"


# =========================
# load model
# =========================

predictor = ImportancePredictor()

predictor.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

predictor.eval()

print("predictor loaded")


# =========================
# find one sample
# =========================

names = []

for f in os.listdir(LABEL_DIR):

    name = f.replace(".pt", "")

    feat_path = os.path.join(
        FEATURE_DIR,
        name + ".pt"
    )

    if os.path.exists(feat_path):
        names.append(name)

print("samples =", len(names))

name = names[0]

print("test image =", name)


# =========================
# load feature
# =========================

feature = torch.load(
    os.path.join(
        FEATURE_DIR,
        name + ".pt"
    )
)

feature = feature.squeeze(0).float()

print("feature =", feature.shape)


# =========================
# predictor
# =========================

with torch.no_grad():

    scores = predictor(feature)

    probs = torch.sigmoid(scores)

print("score shape =", probs.shape)


# =========================
# load GT label
# =========================

label = torch.load(
    os.path.join(
        LABEL_DIR,
        name + ".pt"
    )
)

print("label shape =", label.shape)


# =========================
# F1
# =========================

pred = (probs > 0.5).float()

tp = (pred * label).sum()

fp = (pred * (1 - label)).sum()

fn = ((1 - pred) * label).sum()

precision = tp / (tp + fp + 1e-6)

recall = tp / (tp + fn + 1e-6)

f1 = 2 * precision * recall / (
    precision + recall + 1e-6
)

acc = (pred == label).float().mean()

print()
print("ACC =", acc.item())
print("Precision =", precision.item())
print("Recall =", recall.item())
print("F1 =", f1.item())


# =========================
# Top-K Coverage
# =========================

k = 288

idx = torch.topk(
    probs,
    k
).indices

selected = torch.zeros_like(label)

selected[idx] = 1

coverage = (
    (selected * label).sum()
    /
    (label.sum() + 1e-6)
)

print("Coverage@50% =", coverage.item())


# =========================
# heatmap
# =========================

pred_map = probs.reshape(
    24,
    24
)

gt_map = label.reshape(
    24,
    24
)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.imshow(gt_map)

plt.title("GT Label")

plt.colorbar()


plt.subplot(1,2,2)

plt.imshow(pred_map)

plt.title("Predictor")

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "heatmap_compare.png",
    dpi=300
)

print("saved heatmap_compare.png")