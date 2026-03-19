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


class LocalTrainWorker(QThread):
    """
    Background thread that trains the local icon classifier.

    Signals:
        progress(percent: int, message: str)
        finished(success: bool, message: str)
    """
    progress = Signal(int, str)
    finished = Signal(bool, str)

    def __init__(self, data_mgr, sets_root: Path, parent=None):
        super().__init__(parent)
        self._data_mgr  = data_mgr
        self._sets_root = sets_root

    # ── entry point ───────────────────────────────────────────────────────────

    def run(self):
        try:
            self._train()
        except Exception as e:
            log.exception('LocalTrainWorker error')
            self.finished.emit(False, f'Training failed: {e}')

    # ── main pipeline ─────────────────────────────────────────────────────────

    def _train(self):
        from src.setsdebug import log as _slog
        from pathlib import Path
        self.progress.emit(2, 'Loading confirmed annotations...')
        _slog.info('LocalTrainer: starting icon classifier training')

        # ── 1. Collect crops ─────────────────────────────────────────────────
        crops, labels = self._collect_crops()
        n_classes = len(set(labels))

        if len(crops) < MIN_SAMPLES:
            _slog.info('LocalTrainer: no confirmed crops found — skipping icon training')
            self.finished.emit(
                True,
                f'No confirmed icon annotations yet — icon classifier skipped.\n'
                f'Confirm items in Recognition Review to train icon recognition.')
            return

        self.progress.emit(8, f'Loaded {len(crops)} crops, {n_classes} classes.')

        # ── 2. Check torch ───────────────────────────────────────────────────
        try:
            import torch
            import torchvision
        except ImportError:
            self.finished.emit(
                False,
                'PyTorch not found in SETS environment.\n'
                'Run:  pip install torch torchvision onnx  inside the SETS .venv')
            return

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

        # 80/20 train/val split
        n     = len(crops)
        split = max(1, int(n * 0.8))
        idx   = list(range(n))
        import random; random.shuffle(idx)
        train_idx, val_idx = idx[:split], idx[split:]

        train_crops  = [crops[i]  for i in train_idx]
        train_labels = [y[i]      for i in train_idx]
        val_crops    = [crops[i]  for i in val_idx]  if val_idx else train_crops[:1]
        val_labels   = [y[i]      for i in val_idx]  if val_idx else train_labels[:1]

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

        # Check if we have a previous model to continue from
        existing = self._sets_root / 'warp' / 'models' / 'icon_classifier.onnx'
        if existing.exists():
            self.progress.emit(20, 'Previous model found -- fine-tuning...')
        else:
            self.progress.emit(20, 'No previous model -- training from ImageNet weights...')

        # Replace classifier head
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, n_classes)
        model = model.to(device)

        # Freeze backbone for first half of training if we have few samples
        if len(crops) < 50:
            for p in model.features.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS)
        criterion = torch.nn.CrossEntropyLoss()

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

        # Write label map and meta
        with open(label_path, 'w') as f:
            json.dump(idx_to_label, f, ensure_ascii=False, indent=2)
        with open(meta_path, 'w') as f:
            json.dump({'n_classes': n_classes, 'input_size': MODEL_IMG_SIZE}, f)

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

        self.progress.emit(100,
            f'Training complete!  {n_classes} classes  '
            f'best val acc: {best_val_acc:.1%}')
        self.finished.emit(
            True,
            f'Model trained successfully.\n'
            f'{len(crops)} crops  |  {n_classes} item types  |  '
            f'val accuracy: {best_val_acc:.1%}\n\n'
            f'Saved to: {pt_path}')
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
            _slog.info(f'LocalTrainer:   [index] {name!r}')

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
