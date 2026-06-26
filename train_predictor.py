import torch
import torch.nn as nn

from semantic.semantic_predictor import SemanticPredictor

data = torch.load(
    "semantic/train_data.pt"
)

visual = data["visual"]
semantic = data["semantic"]
mask = data["mask"]

print(visual.shape)
print(semantic.shape)
print(mask.shape)

model = SemanticPredictor()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

criterion = nn.BCEWithLogitsLoss()

EPOCHS = 20

for epoch in range(EPOCHS):

    score = model(
        visual,
        semantic,
        mask
    )

    loss = criterion(
        score,
        mask.float()
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        epoch,
        loss.item()
    )

torch.save(
    model.state_dict(),
    "semantic/semantic_predictor.pth"
)