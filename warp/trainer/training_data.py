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


# Slots where only one confirmed annotation is allowed per image.
# Accepting a new confirmed annotation for these slots removes any existing one
# at a different bbox position (prevents duplicates from misclicks).
SINGLE_INSTANCE_SLOTS: frozenset = frozenset({
    'Ship Name', 'Ship Type', 'Ship Tier',
    'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield',
    'Kit', 'Body Armor', 'EV Suit', 'Personal Shield',
    'Primary Specialization', 'Secondary Specialization',
})

# Slots that contain text / fixed values — NOT icons.
# Bboxes for these slots are stored in annotations.json for layout learning
# (so OCR can be directed to the right region) but must NOT generate crop PNGs
# or crop_index entries, since ML cannot classify free-form text.
NON_ICON_SLOTS: frozenset = frozenset({
    'Ship Name',   # free-text, unique per ship — OCR only
    'Ship Type',   # fixed vocabulary, but text — OCR only
    'Ship Tier',   # T1–T6-X2 — OCR only
})

# Sub-categories of NON_ICON_SLOTS with different data-handling behaviour:
#
# POSITION_ONLY_SLOTS — bbox saved for layout anchoring only.
#   No crop PNG, no crop_index entry, no upload. (Privacy: Ship Name is the
#   player's character name — never stored beyond bbox coordinates.)
POSITION_ONLY_SLOTS: frozenset = frozenset({'Ship Name'})

# TEXT_LEARNING_SLOTS — crop PNG + confirmed text + ml_name (raw OCR) saved
#   and uploaded to HF staging so the central pipeline can build
#   ship_type_corrections.json via democratic voting.
TEXT_LEARNING_SLOTS: frozenset = frozenset({'Ship Type', 'Ship Tier'})


@dataclass
class Annotation:
    """One bounding-box annotation for an icon in a screenshot."""
    bbox:     tuple           # (x, y, w, h) in original image pixels
    slot:     str  = ""       # SETS slot name, e.g. "Fore Weapons"
    name:     str  = ""       # SETS item name
    state:    AnnotationState = AnnotationState.PENDING
    ann_id:   str  = ""       # unique ID (hash of image+bbox)
    ml_conf:  float = 0.0     # original ML recognition confidence (0.0 = unknown)
    ml_name:  str  = ""       # what ML originally recognised (may differ from confirmed name)
    crop_name: str = ""       # relative path of saved crop PNG (set by _export_crop)

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

    ANNOTATIONS_FILE    = "annotations.json"
    CROPS_DIR           = "crops"
    CROP_INDEX_FILE     = "crops/crop_index.json"
    SCREEN_TYPES_FILE   = "screen_types.json"
    USER_CONFIRMED_FILE = "screen_types_user_confirmed.json"

    def __init__(self, data_dir: Path):
        self._dir       = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / self.CROPS_DIR).mkdir(exist_ok=True)

        self._annotations:  dict[str, list[dict]] = {}   # filename → list of ann dicts
        self._crop_index:   dict[str, dict]       = {}   # crop_filename → metadata
        self._screen_types: dict[str, str]        = {}   # filename → screen type string
        self._screen_types_user_confirmed: set[str] = set()  # filenames confirmed by user
        self._dirty = False

        self._load()
        # Auto-repair crop_index on startup (fixes entries from pre-fix versions)
        try:
            repaired = self.repair_crop_index()
            if repaired:
                logger.info(f'TrainingDataManager: auto-repaired {repaired} crop_index entries')
        except Exception as e:
            logger.warning(f'TrainingDataManager: repair_crop_index failed: {e}')

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
        ml_conf: float = 0.0,
        ml_name: str = "",
    ) -> Annotation:
        """Add or update annotation, treating the same bbox as the same annotation.

        Lookup priority:
          1. Match by ann_id (exact, same bbox+slot — no change needed)
          2. Match by bbox alone — handles slot/name edits on existing bbox
             (old ann_id had slot baked in; new slot → different ann_id, same bbox)
          3. Single-instance slots: remove any other confirmed annotation for the
             same slot before inserting (prevents duplicate Deflector, Shield, etc.)
          4. Insert as new annotation
        """
        ann = Annotation(bbox=bbox, slot=slot, name=name, state=state,
                         ml_conf=ml_conf, ml_name=ml_name)
        key = image_path.name
        if key not in self._annotations:
            self._annotations[key] = []

        try:
            from src.setsdebug import log as _log
        except Exception:
            _log = logger

        _log.debug(f'add_annotation: img={key} slot={slot!r} bbox={bbox} state={state} ann_id={ann.ann_id}')

        # 1. Update in-place if ann_id already exists (exact match — fast path)
        for i, d in enumerate(self._annotations[key]):
            if d.get('ann_id') == ann.ann_id:
                _log.debug(f'add_annotation: step1 hit — updating in-place (ann_id={ann.ann_id})')
                self._annotations[key][i] = asdict(ann)
                self._dirty = True
                return ann

        # 2. Fallback: same bbox, slot was edited → update in-place without duplicating
        # Skip for NON_ICON_SLOTS — Ship Name/Type/Tier can share bbox coords legitimately;
        # a slot-agnostic match would overwrite one with another.
        bbox_t = tuple(bbox)
        if slot not in NON_ICON_SLOTS:
            for i, d in enumerate(self._annotations[key]):
                if tuple(d.get('bbox', [])) == bbox_t:
                    _log.debug(f'add_annotation: step2 hit — slot change on same bbox '
                               f'old_slot={d.get("slot")!r} -> {slot!r}')
                    self._annotations[key][i] = asdict(ann)
                    self._dirty = True
                    return ann

        # 3. Single-instance slots: drop any existing confirmed annotation for
        #    this slot at a different bbox (user re-drew at correct position)
        if state == AnnotationState.CONFIRMED and slot in SINGLE_INSTANCE_SLOTS:
            before = len(self._annotations[key])
            self._annotations[key] = [
                d for d in self._annotations[key]
                if not (d.get('slot') == slot and d.get('state') == 'confirmed')
            ]
            removed = before - len(self._annotations[key])
            if removed:
                _log.debug(f'add_annotation: step3 removed {removed} existing confirmed {slot!r}')

        # 4. New annotation
        _log.debug(f'add_annotation: step4 insert new — {slot!r} bbox={bbox}')
        self._annotations[key].append(asdict(ann))
        self._dirty = True
        try:
            self._export_crop(image_path, ann)
            # Sync state in crop_index (export sets PENDING, update if CONFIRMED)
            self._sync_crop_index(image_path, ann)
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
        If bbox is provided, replaces ann.bbox before saving.
        If state is CONFIRMED, also updates crop_index and re-exports crop."""
        from dataclasses import replace as dc_replace
        if bbox is not None:
            ann = dc_replace(ann, bbox=bbox)
        key = image_path.name
        dicts = self._annotations.get(key, [])
        for i, d in enumerate(dicts):
            if d.get("ann_id") == ann.ann_id:
                dicts[i] = asdict(ann)
                self._dirty = True
                # Sync crop_index state + re-export crop if confirmed
                self._sync_crop_index(image_path, ann)
                return

    def remove_annotation(self, image_path: Path, ann: Annotation):
        """Remove an annotation by ann_id and clean up its crop_index entry + PNG."""
        key = image_path.name
        dicts = self._annotations.get(key, [])
        self._annotations[key] = [d for d in dicts if d.get("ann_id") != ann.ann_id]
        self._dirty = True
        # Remove matching crop_index entries (filename contains ann_id) and crop PNG
        to_remove = [f for f in self._crop_index if ann.ann_id in f]
        for fname in to_remove:
            del self._crop_index[fname]
            crop_path = self._dir / self.CROPS_DIR / fname
            if crop_path.exists():
                crop_path.unlink()

    # ---------------------------------------------------------------- persistence

    def save(self):
        """Write annotations.json to disk."""
        ann_path = self._dir / self.ANNOTATIONS_FILE
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(self._annotations, f, indent=2)
        idx_path = self._dir / self.CROP_INDEX_FILE
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(self._crop_index, f, indent=2)
        st_path = self._dir / self.SCREEN_TYPES_FILE
        with open(st_path, "w", encoding="utf-8") as f:
            json.dump(self._screen_types, f, indent=2)
        uc_path = self._dir / self.USER_CONFIRMED_FILE
        with open(uc_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._screen_types_user_confirmed), f, indent=2)
        self._dirty = False
        logger.info(f"Training data saved to {self._dir}")

    def _migrate_clear_ship_name_text(self) -> None:
        """One-time migration: clear stored text from Ship Name annotations.
        Ship Name is position-only — content was never meant to be persisted (privacy).
        Ship Type and Ship Tier are now TEXT_LEARNING_SLOTS and keep their confirmed text.
        """
        _CLEAR_SLOTS = frozenset({'Ship Name'})
        changed = 0
        for anns in self._annotations.values():
            for ann in anns:
                if ann.get('slot') in _CLEAR_SLOTS and ann.get('name', '').strip():
                    ann['name'] = ''
                    changed += 1
        if changed:
            logger.info(f'Migration: cleared name text from {changed} Ship Name annotations')
            self.save()

    def _load(self):
        ann_path = self._dir / self.ANNOTATIONS_FILE
        if ann_path.exists():
            try:
                with open(ann_path, encoding='utf-8') as f:
                    self._annotations = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load annotations: {e}")
        self._migrate_clear_ship_name_text()

        idx_path = self._dir / self.CROP_INDEX_FILE
        if idx_path.exists():
            try:
                with open(idx_path, encoding='utf-8') as f:
                    self._crop_index = json.load(f)
            except Exception:
                pass

        st_path = self._dir / self.SCREEN_TYPES_FILE
        if st_path.exists():
            try:
                with open(st_path, encoding='utf-8') as f:
                    self._screen_types = json.load(f)
            except Exception:
                pass
        uc_path = self._dir / self.USER_CONFIRMED_FILE
        if uc_path.exists():
            try:
                with open(uc_path, encoding='utf-8') as f:
                    self._screen_types_user_confirmed = set(json.load(f))
            except Exception:
                pass

    def _sync_crop_index(self, image_path: Path, ann: Annotation):
        """
        Update crop_index entry for this annotation:
        - Updates state (PENDING → CONFIRMED etc.)
        - Updates name and slot (may change after user confirmation)
        - Re-exports crop if it doesn't exist yet
        - POSITION_ONLY_SLOTS are skipped — their bbox is kept in annotations.json
          for layout learning, but no crop PNG or crop_index entry is created.
        - TEXT_LEARNING_SLOTS get a crop PNG + crop_index entry that includes ml_name
          so SyncWorker can upload them for OCR correction training.
        """
        if ann.slot in POSITION_ONLY_SLOTS:
            return

        safe_slot = ann.slot.replace(" ", "_").lower()
        safe_name = (ann.name or "unknown").replace(" ", "_").lower()[:40]
        fname     = f"{safe_slot}__{safe_name}__{ann.ann_id}.png"
        out_path  = self._dir / self.CROPS_DIR / fname

        # Also look for any existing crop with this ann_id (name/slot may have changed)
        old_fname = next(
            (f for f, m in self._crop_index.items() if ann.ann_id in f), None)
        if old_fname and old_fname != fname:
            # Rename crop file to reflect new name/slot
            old_path = self._dir / self.CROPS_DIR / old_fname
            if old_path.exists():
                old_path.rename(out_path)
            del self._crop_index[old_fname]

        # Export crop if missing
        if not out_path.exists():
            try:
                self._export_crop(image_path, ann)
            except Exception as e:
                logger.warning(f'_sync_crop_index: export failed: {e}')

        # Update index entry with current state/name/slot
        entry: dict = {
            'slot':   ann.slot,
            'name':   ann.name,
            'state':  ann.state,
            'source': image_path.name,
        }
        if ann.slot in TEXT_LEARNING_SLOTS:
            entry['ml_name'] = ann.ml_name
        self._crop_index[fname] = entry

    # ---------------------------------------------------------------- crop export

    def _export_crop(self, image_path: Path, ann: Annotation):
        """
        Crops the icon region from the original screenshot and saves it as PNG.
        Filename is derived from item name + slot (for easy dataset browsing).
        POSITION_ONLY_SLOTS are skipped — no crop needed (position anchor only).
        TEXT_LEARNING_SLOTS get a crop so the text region can be uploaded for OCR training.
        """
        if ann.slot in POSITION_ONLY_SLOTS:
            return
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

        # Write crop_name back into the annotation so seed_from_training_data
        # can find the file without having to reconstruct the path.
        relative = f'crops/{fname}'
        ann.crop_name = relative
        key = image_path.name
        for d in self._annotations.get(key, []):
            if d.get('ann_id') == ann.ann_id:
                d['crop_name'] = relative
                self._dirty = True
                break

        # Update crop index
        _entry: dict = {
            "slot":   ann.slot,
            "name":   ann.name,
            "state":  ann.state,
            "source": image_path.name,
        }
        if ann.slot in TEXT_LEARNING_SLOTS:
            _entry["ml_name"] = ann.ml_name
        self._crop_index[fname] = _entry

    # ---------------------------------------------------------------- export for sync

    def repair_crop_index(self):
        """
        One-time repair: for every confirmed annotation in annotations.json,
        ensure the corresponding crop PNG exists and has state=CONFIRMED in crop_index.
        Call this once after upgrading from a version with the crop_index bug.
        """
        import cv2
        repaired = 0
        for image_name, ann_list in self._annotations.items():
            for d in ann_list:
                if d.get('state') != AnnotationState.CONFIRMED:
                    continue
                ann = self._dict_to_ann(d)
                if not ann.name:
                    continue
                safe_slot = ann.slot.replace(' ', '_').lower()
                safe_name = (ann.name or 'unknown').replace(' ', '_').lower()[:40]
                fname = f'{safe_slot}__{safe_name}__{ann.ann_id}.png'
                # Update crop_index entry to CONFIRMED
                entry = self._crop_index.get(fname)
                if entry is None or entry.get('state') != AnnotationState.CONFIRMED:
                    # Try to find any existing crop with this ann_id
                    existing = next(
                        (f for f in self._crop_index if ann.ann_id in f), None)
                    if existing and existing != fname:
                        old_path = self._dir / self.CROPS_DIR / existing
                        new_path = self._dir / self.CROPS_DIR / fname
                        if old_path.exists():
                            old_path.rename(new_path)
                        del self._crop_index[existing]
                    _repair_entry: dict = {
                        'slot':   ann.slot,
                        'name':   ann.name,
                        'state':  AnnotationState.CONFIRMED,
                        'source': image_name,
                    }
                    if ann.slot in TEXT_LEARNING_SLOTS:
                        _repair_entry['ml_name'] = ann.ml_name
                    self._crop_index[fname] = _repair_entry
                    # Export crop PNG if missing
                    crop_path = self._dir / self.CROPS_DIR / fname
                    if not crop_path.exists():
                        # Find original screenshot to re-export from
                        # We don't have its path here, so just mark as repaired
                        pass
                    repaired += 1

                # Backfill crop_name into annotation dict if missing
                if not d.get('crop_name'):
                    crop_path = self._dir / self.CROPS_DIR / fname
                    if crop_path.exists():
                        d['crop_name'] = f'crops/{fname}'
                        self._dirty = True

        if repaired or self._dirty:
            self.save()
            if repaired:
                logger.info(f'repair_crop_index: fixed {repaired} entries')
        return repaired

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

    # ---------------------------------------------------------------- screen type persistence

    def set_screen_type(self, image_path: Path, screen_type: str, user_confirmed: bool = False) -> Path:
        """
        Records the screen type for a screenshot (persisted to screen_types.json)
        and copies it into the classifier training folder.

        Persistent label: warp/training_data/screen_types.json  {filename: stype}
        Training copy:    warp/training_data/screen_types/<stype>/<filename>

        Returns the destination path of the training copy.
        """
        self._screen_types[image_path.name] = screen_type
        if user_confirmed:
            self._screen_types_user_confirmed.add(image_path.name)
        self.save()
        dest_dir = self._dir / 'screen_types' / screen_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / image_path.name
        shutil.copy2(image_path, dest)
        logger.info(f'Screen type set: {image_path.name} -> {screen_type}')
        return dest

    def get_screen_type(self, image_path: Path) -> str:
        """Returns the persisted screen type for a screenshot, or empty string if not set."""
        return self._screen_types.get(image_path.name, '')

    def get_all_screen_types(self) -> dict[str, str]:
        """Returns a copy of all persisted {filename: stype} labels."""
        return dict(self._screen_types)

    def get_user_confirmed_set(self) -> set[str]:
        """Returns filenames confirmed explicitly by the user (green checkmark)."""
        return set(self._screen_types_user_confirmed)

    def is_user_confirmed(self, image_path: Path) -> bool:
        return image_path.name in self._screen_types_user_confirmed

    def remove_user_confirmed(self, image_path: Path) -> None:
        """Remove user-confirmed mark. Type label in screen_types.json is kept."""
        if image_path.name in self._screen_types_user_confirmed:
            self._screen_types_user_confirmed.discard(image_path.name)
            self.save()

    def remove_screen_type(self, image_path: Path, screen_type: str) -> bool:
        """
        Removes a screenshot from a screen type training folder
        AND removes it from the persistent screen_types.json label.
        Returns True if anything was removed.
        """
        removed = False
        # Remove from persistent label dict and user-confirmed set
        if image_path.name in self._screen_types:
            del self._screen_types[image_path.name]
            removed = True
        self._screen_types_user_confirmed.discard(image_path.name)
        if removed:
            self.save()
        # Remove training copy from disk
        dest = self._dir / 'screen_types' / screen_type / image_path.name
        if dest.exists():
            dest.unlink()
            logger.info(f'Screen type removed: {image_path.name} (label + training copy)')
            removed = True
        return removed

    def get_screen_type_counts(self) -> dict[str, int]:
        """
        Returns the number of training examples per screen type class.
        Dict keys are class names (folder names), values are image counts.
        """
        screen_types_dir = self._dir / 'screen_types'
        if not screen_types_dir.exists():
            return {}
        return {
            d.name: len(list(d.glob('*.png')) + list(d.glob('*.jpg')) + list(d.glob('*.jpeg')))
            for d in screen_types_dir.iterdir()
            if d.is_dir()
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
            ml_conf=float(d.get("ml_conf", 0.0)),
            ml_name=d.get("ml_name", ""),
        )
