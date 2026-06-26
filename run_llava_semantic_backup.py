import os
import torch

from ultralytics import YOLO

from semantic.question_encoder import QuestionEncoder
from semantic.bbox_to_mask import bbox_to_mask


# =========================
# Config
# =========================

QUESTION = "What color is the cat?"

IMAGE_PATH = (
    "/root/autodl-tmp/LLaVA/images/cat.png"
)

YOLO_PATH = (
    "/root/autodl-tmp/LLaVA/yolov8s-world.pt"
)

CACHE_DIR = (
    "semantic/cache"
)


# =========================
# Concept Extraction
# =========================

def extract_concepts(question):

    stop_words = {
        "what",
        "where",
        "when",
        "who",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "does",
        "do",
        "did",
        "many",
        "much",
        "color"
    }

    words = (
        question
        .lower()
        .replace("?", "")
        .split()
    )

    concepts = []

    for w in words:

        if w not in stop_words:
            concepts.append(w)

    return concepts


# =========================
# Save Semantic Feature
# =========================

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

encoder = QuestionEncoder()

semantic_feature = encoder.encode(
    QUESTION
)

torch.save(
    semantic_feature,
    f"{CACHE_DIR}/semantic.pt"
)

print(
    "semantic:",
    semantic_feature.shape
)


# =========================
# Concept
# =========================

concepts = extract_concepts(
    QUESTION
)

print(
    "concepts:",
    concepts
)


# =========================
# YOLO
# =========================

model = YOLO(
    YOLO_PATH
)

model.set_classes(
    concepts
)

results = model.predict(
    IMAGE_PATH,
    verbose=False
)

# =========================
# Build Mask
# =========================

final_mask = torch.zeros(
    576
)

for r in results:

    image_h = r.orig_shape[0]
    image_w = r.orig_shape[1]

    for box in r.boxes.xyxy:

        bbox = box.tolist()

        mask = bbox_to_mask(
            bbox,
            image_w,
            image_h,
            grid_size=24
        )

        final_mask = torch.maximum(
            final_mask,
            mask.float()
        )

print(
    "selected:",
    int(final_mask.sum())
)

torch.save(
    final_mask.unsqueeze(0),
    f"{CACHE_DIR}/mask.pt"
)

print(
    "mask:",
    final_mask.unsqueeze(0).shape
)

print(
    "saved:",
    f"{CACHE_DIR}/semantic.pt"
)

print(
    "saved:",
    f"{CACHE_DIR}/mask.pt"
)