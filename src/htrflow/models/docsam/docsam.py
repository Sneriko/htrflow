"""Reference HTRflow wrapper for loading a DocSAM `.pth` checkpoint.

Copy this file into an HTRflow checkout (e.g. `src/htrflow/models/docsam/docsam.py`)
and adjust imports if your local package paths differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

# HTRflow imports (from htrflow repo)
from htrflow.document import Region
from htrflow.models.base_model import BaseModel
from htrflow.utils.geometry import Bbox, Polygon

# DocSAM imports (from your training code/repo)
from htrflow.models.docsam.DocSAM import DocSAM


@dataclass
class DocSAMResult:
    bbox: list[float]
    polygon: list[list[float]] | None
    score: float
    label: str


def _extract_state_dict(blob: Any) -> dict[str, torch.Tensor]:
    """Accept common checkpoint layouts and return a plain state_dict."""
    if isinstance(blob, dict):
        if "state_dict" in blob and isinstance(blob["state_dict"], dict):
            return blob["state_dict"]
        if "model" in blob and isinstance(blob["model"], dict):
            return blob["model"]
        if all(isinstance(v, torch.Tensor) for v in blob.values()):
            return blob
    raise ValueError("Unsupported checkpoint format. Expected state_dict-like .pth")


def _load_docsam_weights(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load `.pth` like train.py/test.py (`strict=False` with shape/key matching)."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pre_dict = _extract_state_dict(blob)

    cur_dict = model.state_dict()
    matched_dict: dict[str, torch.Tensor] = {}
    for key, value in cur_dict.items():
        if key in pre_dict and pre_dict[key].shape == value.shape:
            matched_dict[key] = pre_dict[key]

    model.load_state_dict(matched_dict, strict=False)
    return model


def _masks_to_polygon(mask: np.ndarray, approx_eps_ratio: float = 0.005) -> list[list[float]] | None:
    """Convert a binary mask to a simplified polygon."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, approx_eps_ratio * peri, True)

    if approx is None or len(approx) < 4:
        return None

    return [[float(p[0][0]), float(p[0][1])] for p in approx]


class DocSAMModel(BaseModel):
    """HTRflow segmentation wrapper for DocSAM `.pth` checkpoints."""

    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "base",
        mask2former_path: str | None = None,
        sentence_path: str | None = None,
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.class_names = class_names or ["region"]
        self.confidence_threshold = confidence_threshold

        model = DocSAM(
            model_size=model_size,
            mask2former_path=mask2former_path,
            sentence_path=sentence_path,
        )
        model = _load_docsam_weights(model, checkpoint_path)
        model.eval()
        model.to(self.device)

        self.model = model
        self.metadata.update(
            {
                "model": "DocSAMModel",
                "checkpoint_path": checkpoint_path,
                "model_size": model_size,
                "mask2former_path": mask2former_path,
                "sentence_path": sentence_path,
            }
        )

    def _prepare_batch(self, image: np.ndarray) -> dict[str, Any]:
        """Build minimal batch dict expected by your DocSAM forward.

        Replace this with your production preprocessing pipeline so it matches training.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        pil = Image.fromarray(rgb)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        chw = np.transpose(arr, (2, 0, 1))
        pixel_values = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        pixel_mask = torch.ones((1, pixel_values.shape[-2], pixel_values.shape[-1]), dtype=torch.bool, device=self.device)

        # The training code uses many fields; keep placeholders for inference usage.
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "instance_masks": None,
            "instance_bboxes": None,
            "instance_labels": None,
            "semantic_masks": None,
            "class_names": [self.class_names],
            "coco_datas": None,
            "image_bboxes": None,
            "dataset_names": ["inference"],
            "image_names": ["image"],
        }

    def _decode_outputs(self, outputs: Any, image_shape: tuple[int, int]) -> list[DocSAMResult]:
        """Convert model raw output to normalized intermediate results.

        NOTE: Replace field names below to match your DocSAM output object.
        """
        # Expected placeholders:
        # masks: (N, H, W) bool/float
        # scores: (N,)
        # labels: (N,) class ids
        masks = getattr(outputs, "pred_masks", None)
        scores = getattr(outputs, "scores", None)
        labels = getattr(outputs, "labels", None)

        if masks is None or scores is None:
            return []

        if torch.is_tensor(masks):
            masks = masks.detach().float().cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.detach().float().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()

        decoded: list[DocSAMResult] = []
        for idx in range(len(scores)):
            score = float(scores[idx])
            if score < self.confidence_threshold:
                continue

            mask = (masks[idx] > 0.5).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())
            polygon = _masks_to_polygon(mask)

            label_idx = int(labels[idx]) if labels is not None else 0
            label_name = self.class_names[label_idx] if 0 <= label_idx < len(self.class_names) else str(label_idx)

            decoded.append(
                DocSAMResult(
                    bbox=[x1, y1, x2, y2],
                    polygon=polygon,
                    score=score,
                    label=label_name,
                )
            )

        return decoded

    @torch.inference_mode()
    def _predict(self, images: list[np.ndarray], use_polygons: bool = True, **kwargs):
        all_results: list[list[Region]] = []

        for image in images:
            batch = self._prepare_batch(image)
            outputs = self.model(batch)
            decoded = self._decode_outputs(outputs, image_shape=image.shape[:2])

            regions: list[Region] = []
            for det in decoded:
                if use_polygons and det.polygon is not None and len(det.polygon) >= 4:
                    shape = Polygon(det.polygon)
                else:
                    shape = Bbox(det.bbox)

                regions.append(
                    Region(
                        shape,
                        segmentation_confidence=det.score,
                        segmentation_label=det.label,
                    )
                )

            all_results.append(regions)

        return all_results
