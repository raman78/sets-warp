# warp/trainer/screen_type_trainer.py
#
# Fine-tunes MobileNetV3-Small on user-corrected STO screenshots.
#
# Training data layout:
#   <data_root>/screen_types/
#       SPACE/           *.png  (224×224, saved by trainer_window on correction)
#       GROUND/          *.png
#       SPACE_TRAITS/    *.png
#       GROUND_TRAITS/   *.png
#       BOFFS/           *.png
#       SPEC/            *.png
#       SPACE_MIXED/     *.png
#       GROUND_MIXED/    *.png
#
# Output:
#   <models_dir>/screen_classifier.onnx
#   <models_dir>/screen_classifier_labels.json
#
# Requirements (already in SETS venv):
#   torch torchvision onnx
#
# Runs in a QThread so it never blocks the UI.
# Emits progress(int pct, str msg) and finished(bool ok, str msg).

from __future__ import annotations

import logging
import json
from pathlib import Path

log = logging.getLogger(__name__)

SCREEN_TYPES = [
    'SPACE_EQ', 'GROUND_EQ', 'TRAITS',
    'BOFFS', 'SPECIALIZATIONS', 'SPACE_MIXED', 'GROUND_MIXED',
]

# Legacy folder names that map to the new TRAITS class
TRAITS_LEGACY_FOLDERS = ['SPACE_TRAITS', 'GROUND_TRAITS', 'TRAITS']
# Legacy top-level class names (before rename)
LEGACY_CLASS_MAP = {
    'SPACE':  'SPACE_EQ',
    'GROUND': 'GROUND_EQ',
    'SPEC':   'SPECIALIZATIONS',
}

MIN_IMAGES_PER_CLASS = 1    # accept any class with at least 1 image
INPUT_SIZE           = 224
BATCH_SIZE           = 8
MAX_EPOCHS           = 60   # more epochs needed for small dataset
LR                   = 3e-4
PATIENCE             = 10   # early stopping


# ── QThread worker ─────────────────────────────────────────────────────────────

class ScreenTypeTrainerWorker:
    """
    QThread-compatible worker.  Import and instantiate inside a QThread.

    Usage (from trainer_window):
        from warp.trainer.screen_type_trainer import ScreenTypeTrainerWorker
        from PySide6.QtCore import QThread, Signal

        class _TrainThread(QThread):
            progress = Signal(int, str)
            finished = Signal(bool, str)
            def __init__(self, data_root, models_dir):
                super().__init__()
                self._w = ScreenTypeTrainerWorker(data_root, models_dir)
            def run(self):
                self._w.run(self.progress.emit, self.finished.emit)

    progress(pct: int, msg: str)  — 0-100
    finished(ok: bool, msg: str)
    """

    def __init__(self, data_root: Path, models_dir: Path):
        self._data_root  = Path(data_root)
        self._models_dir = Path(models_dir)

    def run(self,
            progress_cb=None,   # (pct: int, msg: str) -> None
            finished_cb=None,   # (ok: bool, msg: str) -> None
            interrupt_check=None):  # () -> bool  (return True to stop)
        def prog(pct, msg):
            log.info(f'ScreenTypeTrainer [{pct}%] {msg}')
            if progress_cb:
                progress_cb(pct, msg)

        def done(ok, msg):
            if ok:
                log.info(f'ScreenTypeTrainer: {msg}')
            else:
                log.error(f'ScreenTypeTrainer: {msg}')
            if finished_cb:
                finished_cb(ok, msg)

        try:
            self._train(prog, done, interrupt_check or (lambda: False))
        except Exception as e:
            log.exception('ScreenTypeTrainer unexpected error')
            done(False, str(e))

    # ── Internal ────────────────────────────────────────────────────────────────

    def _train(self, prog, done, interrupted):
        # ── Imports ────────────────────────────────────────────────────────────
        prog(0, 'Importing PyTorch...')
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset
            from torchvision import transforms
            from torchvision.models import mobilenet_v3_small
        except ImportError as e:
            done(False, f'PyTorch not available: {e}')
            return

        if interrupted():
            done(False, 'Cancelled')
            return

        # ── Load dataset ───────────────────────────────────────────────────────
        prog(5, 'Scanning training data...')
        screen_types_dir = self._data_root / 'screen_types'
        if not screen_types_dir.exists():
            done(False, 'No screen_types training data found. '
                        'Correct some screen types first.')
            return

        label_map: dict[int, str] = {}
        samples: list[tuple[Path, int]] = []   # (path, class_idx)

        def _canonical(folder_name: str) -> str | None:
            """Map any folder name (including legacy) to current class name."""
            if folder_name in TRAITS_LEGACY_FOLDERS:
                return 'TRAITS'
            if folder_name in LEGACY_CLASS_MAP:
                return LEGACY_CLASS_MAP[folder_name]
            if folder_name in SCREEN_TYPES:
                return folder_name
            return None

        # Build class index from folders present on disk (deduplicated)
        present_classes = sorted({
            _canonical(d.name)
            for d in screen_types_dir.iterdir()
            if d.is_dir() and _canonical(d.name) is not None
        })
        if not present_classes:
            done(False, 'No class folders found in screen_types/.')
            return

        for idx, cls in enumerate(present_classes):
            label_map[idx] = cls

        # Load samples — merge all legacy folders into canonical class index
        cls_to_idx = {c: i for i, c in label_map.items()}
        for d in screen_types_dir.iterdir():
            if not d.is_dir():
                continue
            canonical = _canonical(d.name)
            if canonical is None:
                continue
            idx = cls_to_idx[canonical]
            for p in d.glob('*.png'):
                samples.append((p, idx))

        if not samples:
            done(False,
                 f'Not enough images. Need ≥{MIN_IMAGES_PER_CLASS} per class.')
            return

        n_classes = len(label_map)
        prog(10, f'{len(samples)} images across {n_classes} classes — '
                 f'classes: {list(label_map.values())}')

        if interrupted():
            done(False, 'Cancelled')
            return

        # ── Dataset ────────────────────────────────────────────────────────────
        # Aggressive augmentation — essential for small datasets (10-30 images/class).
        # Each image is seen with different crops/colors/flips each epoch,
        # effectively multiplying the training set ~10x.
        aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        import cv2, random
        import numpy as np
        random.shuffle(samples)
        split    = max(1, int(len(samples) * 0.85))
        tr_s, va_s = samples[:split], samples[split:]

        class _DS(Dataset):
            def __init__(self, items, tf):
                self._items = items
                self._tf    = tf
            def __len__(self):  return len(self._items)
            def __getitem__(self, i):
                path, lbl = self._items[i]
                img = cv2.imread(str(path))
                if img is None:
                    img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return self._tf(img), lbl

        # ── Weighted sampler — equalises class imbalance ───────────────────────
        import math
        class_counts = [0] * n_classes
        for _, lbl in tr_s:
            class_counts[lbl] += 1
        class_weights = [1.0 / max(c, 1) for c in class_counts]
        sample_weights = torch.tensor([class_weights[lbl] for _, lbl in tr_s])
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(tr_s), replacement=True)

        # Loss weights — penalise errors on rare classes more
        loss_weights = torch.tensor(class_weights, dtype=torch.float32)
        loss_weights = loss_weights / loss_weights.sum() * n_classes  # normalise

        tr_loader = DataLoader(_DS(tr_s, aug),  batch_size=BATCH_SIZE,
                               sampler=sampler, num_workers=0)
        va_loader = DataLoader(_DS(va_s, val_tf), batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

        # ── Model ──────────────────────────────────────────────────────────────
        prog(15, 'Building model...')
        device = torch.device('cpu')
        model  = mobilenet_v3_small(weights='DEFAULT')
        # Replace classifier head for our n_classes
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_classes)
        model = model.to(device)

        # Phase 1: freeze backbone, train only classifier head (warmup)
        for param in model.features.parameters():
            param.requires_grad = False

        optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR)
        criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))

        WARMUP_EPOCHS = 10  # epochs with frozen backbone

        # ── Training loop ──────────────────────────────────────────────────────
        class_dist = ', '.join(f'{label_map[i]}:{class_counts[i]}' for i in range(n_classes))
        prog(16, f'Class distribution: {class_dist}')

        best_val_acc = 0.0
        best_state   = None
        no_improve   = 0
        backbone_unfrozen = False

        for epoch in range(MAX_EPOCHS):
            if interrupted():
                done(False, 'Cancelled')
                return

            # Unfreeze backbone after warmup — fine-tune with lower LR
            if epoch == WARMUP_EPOCHS and not backbone_unfrozen:
                for param in model.features.parameters():
                    param.requires_grad = True
                optimiser = torch.optim.AdamW(model.parameters(), lr=LR * 0.1)
                backbone_unfrozen = True
                prog(15 + int(75 * epoch / MAX_EPOCHS),
                     f'Epoch {epoch+1}: backbone unfrozen, fine-tuning...')

            model.train()
            train_loss = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimiser.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    preds = model(xb.to(device)).argmax(1).cpu()
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            val_acc = correct / total if total > 0 else 0.0

            pct = 15 + int(75 * (epoch + 1) / MAX_EPOCHS)
            prog(pct,
                 f'Epoch {epoch+1}/{MAX_EPOCHS}  '
                 f'loss={train_loss/max(len(tr_loader),1):.3f}  '
                 f'val_acc={val_acc:.1%}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    prog(pct, f'Early stop (no improvement for {PATIENCE} epochs)')
                    break

        # Save model using PyTorch native format — no ONNX export needed
        prog(92, 'Saving model...')
        if best_state:
            model.load_state_dict(best_state)
        model.eval().cpu()
        self._models_dir.mkdir(parents=True, exist_ok=True)

        pt_path     = self._models_dir / 'screen_classifier.pt'
        labels_path = self._models_dir / 'screen_classifier_labels.json'
        meta_path   = self._models_dir / 'screen_classifier_meta.json'
        try:
            torch.save(model.state_dict(), str(pt_path))
            _slog.info(f'ScreenTypeTrainer: model saved to {pt_path}')
        except Exception as e:
            done(False, f'Model save failed: {e}')
            return

        # Save label map and metadata
        with open(labels_path, 'w') as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
        with open(meta_path, 'w') as f:
            json.dump({'n_classes': n_classes, 'input_size': INPUT_SIZE}, f)

        prog(100, f'Done — val accuracy {best_val_acc:.1%}')
        done(True,
             f'Model saved to {pt_path.name}  '
             f'(val_acc={best_val_acc:.1%}, '
             f'{len(samples)} images, {n_classes} classes)')
