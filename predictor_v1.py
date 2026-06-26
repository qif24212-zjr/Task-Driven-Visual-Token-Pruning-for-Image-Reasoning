import torch
import torch.nn as nn

class ImportancePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def forward(self,x):

        score = self.mlp(x)

        return score.squeeze(-1)


image_features = torch.load(
    "/root/autodl-tmp/LLaVA/image_features.pt"
).float()
print(image_features.dtype)
predictor = ImportancePredictor()
print(next(predictor.parameters()).dtype)
scores = predictor(image_features)

print("scores shape =", scores.shape)
k = 288

idx = torch.topk(
    scores,
    k,
    dim=1
).indices

print("topk shape =", idx.shape)
pruned_features = torch.gather(
    image_features,
    1,
    idx.unsqueeze(-1).expand(
        -1,
        -1,
        4096
    )
)

print("pruned shape =", pruned_features.shape)
torch.save(
    idx.cpu(),
    "topk_idx.pt"
)

print("saved idx")