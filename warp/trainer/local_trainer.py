# warp/trainer/local_trainer.py
#
# Local incremental training pipeline for WARP icon classifier.
#
# Flow:
#   1. Load all confirmed annotation crops from TrainingDataManager
#   2. Fine-tune a small EfficientNet-B0 (torchvision) on those crops
#   3. Export to ONNX  ->  warp/models/icon_classifier.onnx
#   4. Write label_map.json
#   5. Clear SETSIconMatcher ML session cache so next call loads new model
#
# The worker emits progress(int, str) during training and
# finished(success: bool, message: str) when done.
#
# Requirements (already in SETS .venv via torchvision):
#   torch, torchvision, onnx, onnxruntime
#
# If torch is not available, the worker falls back gracefully and tells the
# user what to install.

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)

# ── Training hyper-parameters ─────────────────────────────────────────────────
IMG_SIZE        = 64          # match icon_matcher MATCH_SIZE
MODEL_IMG_SIZE  = 224         # EfficientNet input
BATCH_SIZE      = 16
MAX_EPOCHS      = 30
LR              = 3e-4
MIN_SAMPLES     = 1           # minimum confirmed items to start training
PATIENCE        = 5           # early-stop patience (epochs without improvement)
FOCAL_GAMMA     = 2.0         # focal loss focusing parameter
                              # easy samples (p≥0.9) contribute <1% of standard loss


class LocalTrainWorker(QThread):
    """
    Background thread that trains the local icon classifier.

    Signals:
        progress(percent: int, message: str)
        finished(success: bool, message: str)
    """
    progress = Signal(int, str)
    finished = Signal(bool, str)

    def __init__(self, data_mgr, sets_root: Path, aug_passes: int = 1, parent=None):
        super().__init__(parent)
        self._data_mgr   = data_mgr
        self._sets_root  = sets_root
        self._aug_passes = max(1, int(aug_passes))

    # ── entry point ───────────────────────────────────────────────────────────

    def run(self):
        try:
            # Phase 1: Icon Classifier
            icon_msg = self._train_icons()
            
            # Phase 2: Layout Regressor (P4)
            layout_msg = self._train_layout()
            
            summary = icon_msg or ""
            if layout_msg:
                summary += "\n\n" + layout_msg
            
            if not summary:
                summary = "Training skipped: not enough confirmed data."
                
            self.finished.emit(True, summary)
        except Exception as e:
            log.exception('LocalTrainWorker error')
            self.finished.emit(False, f'Training failed: {e}')

    # ── main pipeline ─────────────────────────────────────────────────────────

    # ── Phase 1: Icon Classifier ──────────────────────────────────────────────

    def _train_icons(self) -> Optional[str]:
        from src.setsdebug import log as _slog
        from pathlib import Path
        self.progress.emit(2, 'Loading confirmed annotations...')
        _slog.info('LocalTrainer: starting icon classifier training')

        # ── 1. Collect crops ─────────────────────────────────────────────────
        crops, labels = self._collect_crops()
        n_classes = len(set(labels))

        if len(crops) < MIN_SAMPLES:
            _slog.info('LocalTrainer: no confirmed crops found — skipping icon training')
            return 'No confirmed icon annotations yet — icon classifier skipped.'

        self.progress.emit(8, f'Loaded {len(crops)} crops, {n_classes} classes.')

        # ── 2. Check torch ───────────────────────────────────────────────────
        try:
            import torch
            import torchvision
        except ImportError:
            raise ImportError(
                'PyTorch not found in SETS environment.\n'
                'Run:  pip install torch torchvision onnx  inside the SETS .venv')

        self.progress.emit(12, 'Building dataset...')

        # ── 3. Build label map ───────────────────────────────────────────────
        unique_labels = sorted(set(labels))
        label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
        idx_to_label  = {i: l for l, i in label_to_idx.items()}
        y = [label_to_idx[l] for l in labels]

        # ── 4. Preprocess crops -> tensors ───────────────────────────────────
        import torch
        import torchvision.transforms as T
        import torchvision.models as models
        import cv2

        transform_train = T.Compose([
            T.ToPILImage(),
            # RandomResizedCrop simulates different UI scales (UI Scale in STO)
            T.RandomResizedCrop(MODEL_IMG_SIZE, scale=(0.8, 1.0)),
            # Slight rotation and color jitter for different Lighting 2.0 settings
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_val = T.Compose([
            T.ToPILImage(),
            T.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Minimal Dataset
        class CropDataset(torch.utils.data.Dataset):
            def __init__(self, crops, labels, transform):
                self.crops     = crops
                self.labels    = labels
                self.transform = transform
            def __len__(self):  return len(self.crops)
            def __getitem__(self, i):
                bgr = self.crops[i]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return self.transform(rgb), self.labels[i]

        # Stratified train/val split — classes with only 1 sample stay in train.
        # For classes with ≥2 samples one example goes to val, rest to train.
        import random
        from collections import defaultdict
        by_class: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(y):
            by_class[lbl].append(i)
        train_idx: list[int] = []
        val_idx:   list[int] = []
        for lbl, idxs in by_class.items():
            random.shuffle(idxs)
            if len(idxs) >= 2:
                val_idx.append(idxs[0])
                train_idx.extend(idxs[1:])
            else:
                train_idx.extend(idxs)
        random.shuffle(train_idx)

        train_crops  = [crops[i]  for i in train_idx]
        train_labels = [y[i]      for i in train_idx]
        val_crops    = [crops[i]  for i in val_idx]  if val_idx else train_crops[:1]
        val_labels   = [y[i]      for i in val_idx]  if val_idx else train_labels[:1]

        # Augmentation passes — repeat training data N times;
        # each copy gets different random transforms via transform_train
        if self._aug_passes > 1:
            train_crops  = train_crops  * self._aug_passes
            train_labels = train_labels * self._aug_passes
            self.progress.emit(
                15, f'Aug passes={self._aug_passes}: '
                    f'{len(train_crops)} effective training samples '
                    f'({len(val_crops)} val, unchanged)')

        ds_train = CropDataset(train_crops, train_labels, transform_train)
        ds_val   = CropDataset(val_crops,   val_labels,   transform_val)

        dl_train = torch.utils.data.DataLoader(
            ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        dl_val   = torch.utils.data.DataLoader(
            ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        self.progress.emit(18, 'Building EfficientNet-B0 model...')

        # ── 5. Model ─────────────────────────────────────────────────────────
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Replace classifier head (must happen before weight loading so strict=False
        # only skips the head mismatch, not the backbone)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, n_classes)
        model = model.to(device)

        # Load backbone weights from existing .pt model if available.
        # strict=False: backbone layers are restored; the classifier head is
        # skipped when its size differs (different n_classes) and stays as
        # the randomly-initialised head above.
        existing_pt = self._sets_root / 'warp' / 'models' / 'icon_classifier.pt'
        if existing_pt.exists():
            try:
                state = torch.load(str(existing_pt), map_location=device)
                # Strip classifier keys — strict=False skips missing/unexpected keys
                # but still raises on size mismatch (same key, different shape).
                # Removing classifier keys lets backbone load cleanly regardless of
                # how many classes the previous model had.
                backbone_state = {k: v for k, v in state.items()
                                  if not k.startswith('classifier')}
                missing, unexpected = model.load_state_dict(backbone_state, strict=False)
                non_head = [k for k in (missing + unexpected) if 'classifier' not in k]
                if non_head:
                    self.progress.emit(20, f'Previous model: {len(non_head)} unexpected backbone keys — using ImageNet')
                else:
                    self.progress.emit(20, f'Previous model found — fine-tuning backbone')
            except Exception as e:
                self.progress.emit(20, f'Previous model load failed ({e}) — using ImageNet weights')
        else:
            self.progress.emit(20, 'No previous model — training from ImageNet weights')

        # Freeze backbone for first half of training if we have few samples
        if len(crops) < 50:
            for p in model.features.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS)

        # Class weights — penalise rare items more (many items have only 1 crop)
        import torch.nn.functional as _F
        from collections import Counter as _Counter
        counts = _Counter(y)
        _cw = torch.tensor(
            [1.0 / max(counts[i], 1) for i in range(n_classes)],
            dtype=torch.float32, device=device)
        _cw = _cw / _cw.sum() * n_classes   # normalise so average weight ≈ 1

        class _FocalLoss(torch.nn.Module):
            """Focal loss — downweights easy samples, focuses on hard/uncertain ones.
            Samples already predicted with p≥0.9 contribute <1% of standard CE loss.
            Automatically increases attention if an icon drops below that threshold.
            """
            def forward(self, logits, targets):
                ce  = _F.cross_entropy(logits, targets, weight=_cw, reduction='none')
                pt  = torch.exp(-ce)           # probability of the correct class
                return ((1.0 - pt) ** FOCAL_GAMMA * ce).mean()

        criterion = _FocalLoss().to(device)

        self.progress.emit(25, f'Training on {device}...')

        # ── 6. Training loop ─────────────────────────────────────────────────
        best_val_acc   = 0.0
        best_state     = None
        patience_count = 0
        start_pct      = 25
        end_pct        = 88

        for epoch in range(MAX_EPOCHS):
            if self.isInterruptionRequested():
                self.finished.emit(False, 'Training cancelled.')
                return

            # --- train ---
            model.train()
            # Unfreeze backbone after halfway point
            if epoch == MAX_EPOCHS // 2 and len(crops) < 50:
                for p in model.features.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.1)

            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # --- validate ---
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb, yb = xb.to(device), yb.to(device)
                    preds  = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total   += yb.size(0)
            val_acc = correct / total if total > 0 else 0.0

            pct = int(start_pct + (epoch + 1) / MAX_EPOCHS * (end_pct - start_pct))
            self.progress.emit(
                pct,
                f'Epoch {epoch+1}/{MAX_EPOCHS}  val_acc={val_acc:.1%}  '
                f'best={best_val_acc:.1%}')

            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    self.progress.emit(pct, f'Early stop at epoch {epoch+1}.')
                    break

        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)

        self.progress.emit(89, 'Saving model...')

        # Save using PyTorch native format — no ONNX needed
        models_dir = self._sets_root / 'warp' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        pt_path    = models_dir / 'icon_classifier.pt'
        meta_path  = models_dir / 'icon_classifier_meta.json'
        label_path = models_dir / 'label_map.json'

        model.eval().to('cpu')
        try:
            torch.save(model.state_dict(), str(pt_path))
        except Exception as e:
            self.finished.emit(False, f'Model save failed: {e}')
            return

        # Write label map, meta, and version stamp
        with open(label_path, 'w') as f:
            json.dump(idx_to_label, f, ensure_ascii=False, indent=2)
        with open(meta_path, 'w') as f:
            json.dump({'n_classes': n_classes, 'input_size': MODEL_IMG_SIZE}, f)

        import hashlib as _hl
        from datetime import datetime as _dt, timezone as _tz
        _model_sha = _hl.sha256(pt_path.read_bytes()).hexdigest()[:16]
        version_path = models_dir / 'model_version.json'
        with open(version_path, 'w') as f:
            json.dump({
                'version':    _model_sha,
                'trained_at': _dt.now(_tz.utc).isoformat() + 'Z',
                'n_classes':  n_classes,
                'val_acc':    round(best_val_acc, 4),
                'n_samples':  len(crops),
                'source':     'local',
            }, f, indent=2)

        # Remove unavailability sentinel if it exists
        flag = models_dir / 'model_unavailable.flag'
        flag.unlink(missing_ok=True)

        self.progress.emit(95, 'Reloading icon matcher with new model...')

        # Reload matcher
        try:
            from warp.recognition.icon_matcher import SETSIconMatcher
            SETSIconMatcher.reset_ml_session()
        except Exception as e:
            log.warning(f'Matcher reload failed: {e}')

        self.progress.emit(90, 'Icon training complete.')
        
        return (f'Icon Classifier trained successfully.\n'
                f'{len(crops)} crops | {n_classes} classes | '
                f'accuracy: {best_val_acc:.1%}')

    # ── Phase 2: Layout Regressor (P4) ────────────────────────────────────────

    def _train_layout(self) -> Optional[str]:
        """Trains the CNN layout regressor (P4) on confirmed UI structures."""
        from src.setsdebug import log as _slog
        import torch
        import torch.nn as nn
        import cv2
        from warp.trainer.layout_dataset_builder import LayoutDatasetBuilder
        from warp.trainer.layout_trainer import LayoutRegressor, REGRESSOR_SLOTS, NUM_SLOTS, OUTPUT_SIZE

        self.progress.emit(91, 'Loading confirmed layouts for P4...')
        builder = LayoutDatasetBuilder(self._sets_root)
        samples = builder.build()
        
        if len(samples) < 5:
            _slog.info('LocalTrainer: not enough confirmed layouts for P4 training (min 5)')
            return

        self.progress.emit(92, f'Training Layout CNN on {len(samples)} samples...')
        
        # Hyper-parameters for layout
        LAYOUT_EPOCHS = 20
        LAYOUT_LR     = 5e-4
        LAYOUT_BATCH  = 4 # Layout images are large-ish

        # Build training tensors
        x_train, y_train = [], []
        for s in samples:
            img = cv2.imread(str(s['image_path']), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            # Resize to model input size (224x224)
            img_resized = cv2.resize(img, (224, 224))
            x_train.append(img_resized[np.newaxis, :, :].astype(np.float32) / 255.0)
            
            # Label vector (normalized coords for REGRESSOR_SLOTS)
            vec = np.zeros(NUM_SLOTS * 4, dtype=np.float32)
            for i, slot_name in enumerate(REGRESSOR_SLOTS):
                if slot_name in s['labels']:
                    bbox = s['labels'][slot_name] # [nx, ny, nw, nh]
                    vec[i*4 : i*4+4] = bbox
            y_train.append(vec)

        if not x_train: return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LayoutRegressor().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LAYOUT_LR)

        x_t = torch.from_numpy(np.array(x_train)).to(device)
        y_t = torch.from_numpy(np.array(y_train)).to(device)

        model.train()
        for epoch in range(LAYOUT_EPOCHS):
            # Simple batching (not ideal for large datasets but ok for P4 alpha)
            optimizer.zero_grad()
            pred = model(x_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                self.progress.emit(92 + int(epoch/LAYOUT_EPOCHS*5), 
                                   f'Layout Epoch {epoch+1}/{LAYOUT_EPOCHS} loss={loss.item():.4f}')

        # Save P4 model
        models_dir = self._sets_root / 'warp' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        pt_path = models_dir / 'layout_regressor.pt'
        
        model.eval().to('cpu')
        torch.save(model.state_dict(), str(pt_path))
        _slog.info(f'LocalTrainer: P4 Layout Regressor saved to {pt_path.name}')
        self.progress.emit(100, 'Layout training complete.')
        
        return (f'Layout Regressor (P4) trained successfully.\n'
                f'{len(samples)} layouts | Loss: {loss.item():.4f}\n'
                f'Saved to: {pt_path.name}')
    # ── helpers ───────────────────────────────────────────────────────────────

    def _collect_crops(self) -> tuple[list, list]:
        """
        Gather all confirmed annotation crops.
        Strategy 1: pre-exported PNGs from crop_index (fast).
        Strategy 2: re-export from original screenshots via annotations.json
                    (fallback when crop_index is empty or stale).
        Returns (crops_bgr, label_strings).
        """
        from src.setsdebug import log as _slog
        from pathlib import Path
        import cv2
        crops, labels = [], []

        # ── Strategy 1: pre-exported crops from crop_index ───────────────────
        confirmed = self._data_mgr.get_confirmed_crops()
        _slog.info(f'LocalTrainer: {len(confirmed)} crop(s) in crop_index')
        for item in confirmed:
            crop_path = item.get('path')
            name      = item.get('name', '').strip()
            if not crop_path or not name:
                continue
            crop = cv2.imread(str(crop_path))
            if crop is None:
                _slog.warning(f'LocalTrainer: missing crop file {crop_path}')
                continue
            crops.append(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)))
            labels.append(name)

        if crops:
            _slog.info(f'LocalTrainer: {len(crops)} crops from index')
            return crops, labels

        # ── Strategy 2: fallback — read from annotations.json ────────────────
        _slog.info('LocalTrainer: crop_index empty — fallback to annotations.json')
        from warp.trainer.training_data import AnnotationState
        data_dir = self._data_mgr._dir
        screen_types_dir = data_dir / 'screen_types'

        for image_name, ann_list in self._data_mgr._annotations.items():
            for d in ann_list:
                if d.get('state') != AnnotationState.CONFIRMED:
                    continue
                name = (d.get('name') or '').strip()
                bbox = d.get('bbox')
                if not name or not bbox:
                    continue
                # Find screenshot — copies exist in screen_types/<STYPE>/
                img = None
                if screen_types_dir.exists():
                    for stype_dir in screen_types_dir.iterdir():
                        if not stype_dir.is_dir():
                            continue
                        candidate = stype_dir / image_name
                        if candidate.exists():
                            img = cv2.imread(str(candidate))
                            if img is not None:
                                break
                if img is None:
                    _slog.warning(f'LocalTrainer: screenshot not found: {image_name}')
                    continue
                x, y, w, h = bbox
                ih, iw = img.shape[:2]
                crop = img[max(0,y):min(ih,y+h), max(0,x):min(iw,x+w)]
                if crop.size == 0:
                    continue
                crops.append(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)))
                labels.append(name)
                _slog.info(f'LocalTrainer:   [fallback] {name!r} from {image_name}')
                # Save to crop_index so next run uses strategy 1
                ann_obj = self._data_mgr._dict_to_ann(d)
                self._data_mgr._sync_crop_index(candidate, ann_obj)

        if crops:
            self._data_mgr.save()
            _slog.info(f'LocalTrainer: {len(crops)} crops from fallback')
        else:
            _slog.warning('LocalTrainer: no crops found in either strategy')
        return crops, labels
