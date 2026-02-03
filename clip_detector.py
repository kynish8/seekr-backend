import torch
import clip
from PIL import Image
import cv2
from collections import deque
from typing import Dict, List

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "ViT-B/16"

PROCESS_EVERY = 1.0
WINDOW_SIZE = 5
REQUIRED_FRAMES = 3

Z_THRESHOLD = 1.5
MIN_SIM = 0.20
MARGIN = 0.04



OBJECTS: Dict[str, Dict[str, List[str]]] = {
    "spoon": {
        "prompts": [
            "a metal spoon",
            "a spoon on a table",
            "a spoon held in a hand",
            "a spoon in a kitchen",
            "a shiny metal spoon",
            "a spoon next to a bowl",
        ],
        "negatives": [
            "a fork",
            "a knife",
            "chopsticks",
            "a metal rod",
        ],
    },

    "dog": {
        "prompts": [
            "a dog",
            "a puppy",
            "a dog sitting on grass",
            "a dog indoors",
        ],
        "negatives": [
            "a cat",
            "a stuffed animal",
        ],
    },
    "person": {
        "prompts": [
            # Generic person
            "a person",
            "a human",
            "a person indoors",

            # Webcam / laptop context
            "a person sitting at a desk",
            "a person in front of a computer",
            "a person using a laptop",
            "a person looking at a screen",

            # Face-focused
            "a human face",
            "a person facing the camera",
            "a face looking at the camera",
            "a person close to the camera",

            # Upper body / framing
            "a person from the shoulders up",
            "a person sitting indoors facing forward",
            "a person talking to a camera",

            # Lighting & realism
            "a realistic photo of a person",
            "a webcam photo of a person",
            "a low resolution webcam image of a person"
        ],
        "negatives": [
            # Not a real person
            "a photo of a screen",
            "a reflection in a mirror",
            "a mannequin",
            "a doll",
            "a statue",
            "a cardboard cutout",

            # Background-only
            "an empty chair",
            "a desk with no people",
            "a room with no people",
            "a computer screen",

            # Non-human faces
            "a cartoon character",
            "an animated character",
            "a drawing of a face",
            "an emoji face",
            "a robot face",

            # Pets / misc
            "a dog",
            "a cat",
            "a stuffed animal"
        ],
    }
}

GLOBAL_NULLS = [
    "background clutter",
    "a table surface",
    "an empty room",
]

class CLIPDetector:
    def __init__(self):
        self.model, self.preprocess = clip.load(MODEL_NAME, device=DEVICE)
        self.model.eval()
        torch.set_grad_enabled(False)

        self.object_embeddings = {}
        self.null_embeddings = {}

        self.history = deque(maxlen=WINDOW_SIZE)

        self._build_embeddings()

    def _embed_text(self, prompts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(prompts).to(DEVICE)
        features = self.model.encode_text(tokens)
        features = features.mean(dim=0, keepdim=True)
        features /= features.norm(dim=-1, keepdim=True)
        return features

    def _build_embeddings(self):
        for name, cfg in OBJECTS.items():
            self.object_embeddings[name] = self._embed_text(cfg["prompts"])

            nulls = cfg.get("negatives", []) + GLOBAL_NULLS
            self.null_embeddings[name] = self._embed_text(nulls)

    def _get_crops(self, img):
        h, w, _ = img.shape
        crops = []

        # full frame
        crops.append(img)

        # center crop (important for small objects)
        # crops.append(img[h//4:3*h//4, w//4:3*w//4])

        return crops


    def detect(self, frame_bgr):
        crops = self._get_crops(frame_bgr)

        detected_label = "none"
        detected_score = 0.0

        # ---- tuned thresholds ----
        MIN_SIM = 0.20
        MARGIN = 0.04

        for crop in crops:
            # crop is already RGB
            pil = Image.fromarray(crop)
            image_tensor = self.preprocess(pil).unsqueeze(0).to(DEVICE)

            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # ---- similarity scores ----
            scores = {
                name: (image_features @ emb.T).item()
                for name, emb in self.object_embeddings.items()
            }

            # ---- DEBUG (remove later) ----
            best = max(scores.items(), key=lambda x: x[1])

            # ---- margin-based decision ----
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_label, best_score = sorted_scores[0]
            second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

            if (
                best_score > MIN_SIM
                and (best_score - second_score) > MARGIN
            ):
                if best_score > detected_score:
                    detected_label = best_label
                    detected_score = best_score

        # ---- temporal smoothing ----
        self.history.append(detected_label)

        stable = (
            detected_label != "none"
            and self.history.count(detected_label) >= REQUIRED_FRAMES
        )

        if not stable:
            return {"label": "none", "score": 0.0}


        return {
            "label": detected_label,
            "score": round(detected_score, 3),
        }

