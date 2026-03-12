# warp/trainer/training_data.py
# Manages the local training dataset:
#   - Stores annotations (bbox + slot + name + state) per screenshot
#   - Exports cropped icon images for ML training
#   - Maintains annotations.json as the source of truth
#   - Provides API for SyncWorker to read data for upload
#
# Storage layout (under warp/training_data/):
#   annotations.json          — all annotations, keyed by image filename
#   crops/                    — exported icon crops (named by hash)
#   crops/crop_index.json     — maps crop filename → item name + slot

from __future__ import annotations

import json
import hashlib
import shutil
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AnnotationState(str, Enum):
    CANDIDATE = "candidate"   # auto-detected, not yet reviewed
    PENDING   = "pending"     # drawn by user, not yet confirmed
    CONFIRMED = "confirmed"   # confirmed by user (has name + slot)
    SKIPPED   = "skipped"     # user chose to skip (unknown item)


@dataclass
class Annotation:
    """One bounding-box annotation for an icon in a screenshot."""
    bbox:   tuple           # (x, y, w, h) in original image pixels
    slot:   str  = ""       # SETS slot name, e.g. "Fore Weapons"
    name:   str  = ""       # SETS item name
    state:  AnnotationState = AnnotationState.PENDING
    ann_id: str  = ""       # unique ID (hash of image+bbox)

    def __post_init__(self):
        if not self.ann_id:
            self.ann_id = self._make_id()

    def _make_id(self) -> str:
        raw = f"{self.bbox[0]}_{self.bbox[1]}_{self.bbox[2]}_{self.bbox[3]}_{self.slot}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


class TrainingDataManager:
    """
    Manages all annotation data for the WARP CORE trainer.

    Thread safety: not thread-safe — call from Qt main thread only.
    """

    ANNOTATIONS_FILE = "annotations.json"
    CROPS_DIR        = "crops"
    CROP_INDEX_FILE  = "crops/crop_index.json"

    def __init__(self, data_dir: Path):
        self._dir       = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / self.CROPS_DIR).mkdir(exist_ok=True)

        self._annotations: dict[str, list[dict]] = {}   # filename → list of ann dicts
        self._crop_index:  dict[str, dict]       = {}   # crop_filename → metadata
        self._dirty = False

        self._load()

    # ---------------------------------------------------------------- annotation CRUD

    def get_annotations(self, image_path: Path) -> list[Annotation]:
        """Returns all annotations for the given image (as Annotation objects)."""
        key  = image_path.name
        dicts = self._annotations.get(key, [])
        return [self._dict_to_ann(d) for d in dicts]

    def has_annotations(self, image_path: Path) -> bool:
        key = image_path.name
        anns = self._annotations.get(key, [])
        return any(a.get("state") == AnnotationState.CONFIRMED for a in anns)

    def add_annotation(
        self,
        image_path: Path,
        bbox: tuple,
        slot: str = "",
        name: str = "",
        state: AnnotationState = AnnotationState.PENDING,
    ) -> Annotation:
        """Add or update annotation by ann_id (no duplicates)."""
        ann = Annotation(bbox=bbox, slot=slot, name=name, state=state)
        key = image_path.name
        if key not in self._annotations:
            self._annotations[key] = []
        # Update in-place if ann_id already exists
        for i, d in enumerate(self._annotations[key]):
            if d.get('ann_id') == ann.ann_id:
                self._annotations[key][i] = asdict(ann)
                self._dirty = True
                return ann
        # New annotation
        self._annotations[key].append(asdict(ann))
        self._dirty = True
        try:
            self._export_crop(image_path, ann)
        except Exception as e:
            logger.warning(f"Could not export crop for {image_path.name}: {e}")
        return ann

    def add_candidate(
        self,
        image_path: Path,
        slot: str,
        slot_index: int,
        bbox: tuple,
    ) -> bool:
        """
        Add an auto-detected candidate bbox if not already present.
        Returns True if added, False if duplicate.
        """
        key = image_path.name
        existing = self._annotations.get(key, [])

        # Check for duplicate (same slot + index)
        for d in existing:
            if d.get("slot") == slot and d.get("slot_index") == slot_index:
                return False

        ann = Annotation(bbox=bbox, slot=slot, name="", state=AnnotationState.CANDIDATE)
        d   = asdict(ann)
        d["slot_index"] = slot_index
        if key not in self._annotations:
            self._annotations[key] = []
        self._annotations[key].append(d)
        self._dirty = True
        return True

    def update_annotation(
        self, image_path: Path, ann: Annotation,
        bbox: tuple | None = None,
    ):
        """Update an existing annotation in-place (matched by ann_id).
        If bbox is provided, replaces ann.bbox before saving."""
        from dataclasses import replace as dc_replace
        if bbox is not None:
            ann = dc_replace(ann, bbox=bbox)
        key = image_path.name
        dicts = self._annotations.get(key, [])
        for i, d in enumerate(dicts):
            if d.get("ann_id") == ann.ann_id:
                dicts[i] = asdict(ann)
                self._dirty = True
                return

    def remove_annotation(self, image_path: Path, ann: Annotation):
        """Remove an annotation by ann_id."""
        key = image_path.name
        dicts = self._annotations.get(key, [])
        self._annotations[key] = [d for d in dicts if d.get("ann_id") != ann.ann_id]
        self._dirty = True

    # ---------------------------------------------------------------- persistence

    def save(self):
        """Write annotations.json to disk."""
        ann_path = self._dir / self.ANNOTATIONS_FILE
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(self._annotations, f, indent=2)
        idx_path = self._dir / self.CROP_INDEX_FILE
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(self._crop_index, f, indent=2)
        self._dirty = False
        logger.info(f"Training data saved to {self._dir}")

    def _load(self):
        ann_path = self._dir / self.ANNOTATIONS_FILE
        if ann_path.exists():
            try:
                with open(ann_path) as f:
                    self._annotations = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load annotations: {e}")

        idx_path = self._dir / self.CROP_INDEX_FILE
        if idx_path.exists():
            try:
                with open(idx_path) as f:
                    self._crop_index = json.load(f)
            except Exception:
                pass

    # ---------------------------------------------------------------- crop export

    def _export_crop(self, image_path: Path, ann: Annotation):
        """
        Crops the icon region from the original screenshot and saves it as PNG.
        Filename is derived from item name + slot (for easy dataset browsing).
        """
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return

        x, y, w, h = ann.bbox
        h_img, w_img = img.shape[:2]
        # Clamp to image bounds
        x  = max(0, x); y = max(0, y)
        x2 = min(w_img, x + w); y2 = min(h_img, y + h)
        crop = img[y:y2, x:x2]

        if crop.size == 0:
            return

        # Build filename: slot_name + ann_id
        safe_slot = ann.slot.replace(" ", "_").lower()
        safe_name = (ann.name or "unknown").replace(" ", "_").lower()[:40]
        fname     = f"{safe_slot}__{safe_name}__{ann.ann_id}.png"
        out_path  = self._dir / self.CROPS_DIR / fname

        cv2.imwrite(str(out_path), crop)

        # Update crop index
        self._crop_index[fname] = {
            "slot":   ann.slot,
            "name":   ann.name,
            "state":  ann.state,
            "source": image_path.name,
        }

    # ---------------------------------------------------------------- export for sync

    def get_confirmed_crops(self) -> list[dict]:
        """
        Returns list of confirmed crop metadata dicts for upload.
        Each dict: { path, slot, name, source }
        """
        result = []
        crops_dir = self._dir / self.CROPS_DIR
        for fname, meta in self._crop_index.items():
            if meta.get("state") == AnnotationState.CONFIRMED:
                p = crops_dir / fname
                if p.exists():
                    result.append({"path": str(p), **meta})
        return result

    def get_stats(self) -> dict:
        """Returns summary statistics for the dataset."""
        total      = sum(len(v) for v in self._annotations.values())
        confirmed  = sum(
            1 for v in self._annotations.values()
            for d in v if d.get("state") == AnnotationState.CONFIRMED
        )
        images     = len(self._annotations)
        slots: dict[str, int] = {}
        for v in self._annotations.values():
            for d in v:
                s = d.get("slot", "unknown")
                slots[s] = slots.get(s, 0) + 1
        return {
            "images":    images,
            "total":     total,
            "confirmed": confirmed,
            "slots":     slots,
        }

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _dict_to_ann(d: dict) -> Annotation:
        return Annotation(
            bbox=tuple(d.get("bbox", (0, 0, 0, 0))),
            slot=d.get("slot", ""),
            name=d.get("name", ""),
            state=AnnotationState(d.get("state", AnnotationState.PENDING)),
            ann_id=d.get("ann_id", ""),
        )
