import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # bypass corporate SSL proxy

import numpy as np
import torch
import clip
from PIL import Image
from typing import List

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
MODEL_NAME = "ViT-B/16"

# Default detection thresholds (overridden per-object from object_bank)
DEFAULT_THRESHOLD = 0.22
DEFAULT_MARGIN = 0.04

from object_bank import GLOBAL_NULLS


class CLIPDetector:
    def __init__(self):
        print(f"[clip] loading {MODEL_NAME} on {DEVICE}...")
        self.model, self.preprocess = clip.load(MODEL_NAME, device=DEVICE)
        self.model.eval()

        # Active object state (set per-round)
        self._active_object_id: str | None = None
        self._active_pos_emb: torch.Tensor | None = None
        self._active_neg_emb: torch.Tensor | None = None
        self._active_threshold: float = DEFAULT_THRESHOLD
        self._active_margin: float = DEFAULT_MARGIN

        self._warmup()
        print("[clip] ready")

    def _warmup(self):
        """Run a dummy inference so the first real call isn't slow."""
        with torch.no_grad():
            dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            tensor = self.preprocess(dummy).unsqueeze(0).to(DEVICE)
            self.model.encode_image(tensor)

    def _embed_text(self, prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(DEVICE)
            features = self.model.encode_text(tokens)
            features = features.mean(dim=0, keepdim=True)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def set_active_object(self, object_id: str, obj_config: dict):
        """
        Pre-encode embeddings for the current round's object.
        Call this once at round:start — not on every frame.
        """
        self._active_object_id = object_id
        self._active_pos_emb = self._embed_text(obj_config["prompts"])
        nulls = obj_config.get("negatives", []) + GLOBAL_NULLS
        self._active_neg_emb = self._embed_text(nulls)
        self._active_threshold = obj_config.get("threshold", DEFAULT_THRESHOLD)
        self._active_margin = obj_config.get("margin", DEFAULT_MARGIN)
        print(f"[clip] active object set: {object_id}")

    def detect_for_active_object(self, frame_rgb: np.ndarray) -> dict:
        """
        Run CLIP inference for the currently active object.
        Called in a thread executor — torch.no_grad() is required here
        because set_grad_enabled(False) only applies to the thread that called it.
        """
        if self._active_pos_emb is None:
            return {"label": "none", "score": 0.0, "confidence": 0.0}

        with torch.no_grad():
            pil = Image.fromarray(frame_rgb)
            tensor = self.preprocess(pil).unsqueeze(0).to(DEVICE)

            image_features = self.model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            pos_score = (image_features @ self._active_pos_emb.T).item()
            neg_score = (image_features @ self._active_neg_emb.T).item()

        margin = pos_score - neg_score
        detected = pos_score > self._active_threshold and margin > self._active_margin

        raw_confidence = (pos_score - self._active_threshold) / 0.10
        confidence = round(min(max(raw_confidence, 0.0), 1.0), 3)

        return {
            "label": self._active_object_id if detected else "none",
            "score": round(pos_score, 3),
            "confidence": confidence,
        }
