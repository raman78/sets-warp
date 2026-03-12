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
    'SPACE', 'GROUND', 'SPACE_TRAITS', 'GROUND_TRAITS',
    'BOFFS', 'SPEC', 'SPACE_MIXED', 'GROUND_MIXED',
]

MIN_IMAGES_PER_CLASS = 1    # accept any class with at least 1 image
INPUT_SIZE           = 224
BATCH_SIZE           = 8
MAX_EPOCHS           = 20
LR                   = 1e-4
PATIENCE             = 4    # early stopping


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
            import sys as _sys
            _triton = _sys.modules.get('triton')
            if _triton is not None:
                import logging
                logging.getLogger('SETS').warning(
                    f'triton already in sys.modules: file={getattr(_triton, "__file__", "?")}'
                    f' path={getattr(_triton, "__path__", "?")}')
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

        present_classes = sorted([
            d.name for d in screen_types_dir.iterdir()
            if d.is_dir() and d.name in SCREEN_TYPES
        ])
        if not present_classes:
            done(False, 'No class folders found in screen_types/.')
            return

        for idx, cls in enumerate(present_classes):
            label_map[idx] = cls
            cls_dir = screen_types_dir / cls
            pngs    = list(cls_dir.glob('*.png'))
            for p in pngs:
                samples.append((p, idx))

        if not samples:
            done(False,
                 f'Not enough images. Need ≥{MIN_IMAGES_PER_CLASS} per class.')
            return

        n_classes = len(label_map)
        prog(10, f'{len(samples)} images across {n_classes} classes')

        if interrupted():
            done(False, 'Cancelled')
            return

        # ── Dataset ────────────────────────────────────────────────────────────
        aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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

        import numpy as np
        tr_loader = DataLoader(_DS(tr_s, aug),  batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0)
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

        # Freeze backbone, train only classifier first
        for param in model.features.parameters():
            param.requires_grad = False

        optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR)
        criterion = nn.CrossEntropyLoss()

        # ── Training loop ──────────────────────────────────────────────────────
        best_val_acc = 0.0
        best_state   = None
        no_improve   = 0

        for epoch in range(MAX_EPOCHS):
            if interrupted():
                done(False, 'Cancelled')
                return

            # Unfreeze backbone after epoch 3
            if epoch == 3:
                for param in model.features.parameters():
                    param.requires_grad = True
                optimiser = torch.optim.AdamW(model.parameters(), lr=LR * 0.1)

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

        # ── Export ONNX ────────────────────────────────────────────────────────
        prog(92, 'Exporting ONNX...')
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        self._models_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = self._models_dir / 'screen_classifier.onnx'
        dummy     = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
        try:
            import importlib.util, torch.onnx
            has_onnxscript = importlib.util.find_spec('onnxscript') is not None
            if has_onnxscript:
                torch.onnx.export(
                    model, dummy, str(onnx_path),
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                    opset_version=17,
                )
            else:
                # onnxscript not yet installed — use legacy TorchScript path.
                # Restart SETS to auto-install onnxscript via bootstrap.
                scripted = torch.jit.trace(model, dummy)
                torch.onnx.export(
                    scripted, dummy, str(onnx_path),
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=12,
                )
        except Exception as e:
            if 'onnxscript' in str(e):
                done(False,
                     'Missing module: onnxscript.\n\n'
                     'Restart SETS — bootstrap will install it automatically.')
            else:
                done(False, f'ONNX export failed: {e}')
            return

        # ── Save label map ─────────────────────────────────────────────────────
        labels_path = self._models_dir / 'screen_classifier_labels.json'
        with open(labels_path, 'w') as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

        prog(100, f'Done — val accuracy {best_val_acc:.1%}')
        done(True,
             f'Model saved to {onnx_path.name}  '
             f'(val_acc={best_val_acc:.1%}, '
             f'{len(samples)} images, {n_classes} classes)')
