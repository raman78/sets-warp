# warp/trainer/local_trainer.py
#
# Local layout regressor training pipeline.
#
# Icon classifier training was removed — it is now handled exclusively by the
# central training pipeline (admin_train.py on GitHub Actions).  Users contribute
# confirmed crops via SyncWorker; the central model trains on all community data
# and is distributed back via ModelUpdater.
#
# This worker only trains the local layout regressor (layout_regressor.pt),
# which is not part of the central pipeline and is specific to each user's
# confirmed UI layouts.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


class LocalTrainWorker(QThread):
    """
    Background thread that trains the local layout regressor (P4).

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

    def run(self):
        try:
            msg = self._train_layout()
            self.finished.emit(True, msg or 'Layout training skipped: not enough confirmed layouts.')
        except Exception as e:
            log.exception('LocalTrainWorker error')
            self.finished.emit(False, f'Training failed: {e}')

    # ── Layout Regressor (P4) ─────────────────────────────────────────────────

    def _train_layout(self) -> Optional[str]:
        """Trains the CNN layout regressor (P4) on confirmed UI structures."""
        from src.setsdebug import log as _slog
        import torch
        import torch.nn as nn
        import cv2
        import random
        from warp.trainer.layout_dataset_builder import LayoutDatasetBuilder
        from warp.trainer.layout_trainer import LayoutRegressor, REGRESSOR_SLOTS, NUM_SLOTS, OUTPUT_SIZE

        self.progress.emit(5, 'Loading confirmed layouts...')
        _slog.info('LocalTrainer: starting P4 Layout CNN training')
        builder = LayoutDatasetBuilder(self._sets_root)
        samples = builder.build()

        if len(samples) < 5:
            _slog.info('LocalTrainer: not enough confirmed layouts for P4 training (min 5)')
            return None

        self.progress.emit(15, f'Preparing {len(samples)} layout samples...')

        LAYOUT_EPOCHS = 100
        LAYOUT_LR     = 3e-4
        LAYOUT_BATCH  = 8

        x_all, y_all = [], []
        for s in samples:
            img = cv2.imread(str(s['image_path']), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (224, 224))
            x_all.append(img_resized[np.newaxis, :, :].astype(np.float32) / 255.0)

            vec = np.zeros(NUM_SLOTS * 5, dtype=np.float32)
            for i, slot_name in enumerate(REGRESSOR_SLOTS):
                if slot_name in s['labels']:
                    bbox = s['labels'][slot_name]
                    vec[i*5 : i*5+4] = bbox
                    vec[i*5 + 4] = 1.0
            y_all.append(vec)

        if not x_all:
            return None

        idx = list(range(len(x_all)))
        random.shuffle(idx)
        split = int(len(idx) * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LayoutRegressor().to(device)
        criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=LAYOUT_LR)

        x_train = torch.from_numpy(np.array([x_all[i] for i in train_idx])).to(device)
        y_train = torch.from_numpy(np.array([y_all[i] for i in train_idx])).to(device)
        x_val   = torch.from_numpy(np.array([x_all[i] for i in val_idx])).to(device) if val_idx else x_train
        y_val   = torch.from_numpy(np.array([y_all[i] for i in val_idx])).to(device) if val_idx else y_train

        best_val_loss = float('inf')
        best_state    = None

        _slog.info(f'LocalTrainer: P4 starting (epochs={LAYOUT_EPOCHS}, device={device}, samples={len(x_all)})')
        for epoch in range(LAYOUT_EPOCHS):
            if self.isInterruptionRequested():
                return None

            model.train()
            optimizer.zero_grad()
            pred_train = model(x_train)

            p_coords = pred_train.view(-1, NUM_SLOTS, 5)[:, :, :4]
            t_coords = y_train.view(-1, NUM_SLOTS, 5)[:, :, :4]
            p_pres   = pred_train.view(-1, NUM_SLOTS, 5)[:, :, 4]
            t_pres   = y_train.view(-1, NUM_SLOTS, 5)[:, :, 4]

            mask    = t_pres.unsqueeze(-1)
            loss_c  = (criterion(p_coords, t_coords) * mask).sum() / (mask.sum() + 1e-6)
            loss_p  = nn.functional.mse_loss(p_pres, t_pres)
            loss_train = loss_c + loss_p
            loss_train.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val  = model(x_val)
                pv_coords = pred_val.view(-1, NUM_SLOTS, 5)[:, :, :4]
                tv_coords = y_val.view(-1, NUM_SLOTS, 5)[:, :, :4]
                pv_pres   = pred_val.view(-1, NUM_SLOTS, 5)[:, :, 4]
                tv_pres   = y_val.view(-1, NUM_SLOTS, 5)[:, :, 4]
                mv        = tv_pres.unsqueeze(-1)
                lv_c      = (criterion(pv_coords, tv_coords) * mv).sum() / (mv.sum() + 1e-6)
                lv_p      = nn.functional.mse_loss(pv_pres, tv_pres)
                loss_val  = lv_c + lv_p

            if loss_val < best_val_loss:
                best_val_loss = loss_val.item()
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            pct = 20 + int(epoch / LAYOUT_EPOCHS * 75)
            msg = f'Layout Epoch {epoch+1}/{LAYOUT_EPOCHS} loss={loss_train.item():.4f} val={loss_val.item():.4f}'
            self.progress.emit(pct, msg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                _slog.info(f'LocalTrainer: {msg}')

        if best_state:
            model.load_state_dict(best_state)

        models_dir = self._sets_root / 'warp' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        pt_path = models_dir / 'layout_regressor.pt'

        model.eval().to('cpu')
        torch.save(model.state_dict(), str(pt_path))
        _slog.info(f'LocalTrainer: P4 Layout Regressor saved (best val_loss={best_val_loss:.5f})')
        self.progress.emit(100, 'Layout training complete.')

        return (f'Layout Regressor trained.\n'
                f'{len(samples)} layouts | val_loss: {best_val_loss:.5f}')
