# warp/trainer/trainer_window.py
# WARP CORE — Interactive ML trainer. PySide6, integrated with SETS.
# Opens as a standalone QMainWindow, references sets_app for config/cache.

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QComboBox, QLineEdit, QGroupBox,
    QProgressBar, QToolBar, QStatusBar, QMessageBox,
    QInputDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtGui import QFont, QAction

from warp.trainer.annotation_widget import AnnotationWidget
from warp.trainer.training_data      import TrainingDataManager, AnnotationState
from warp.trainer.sync               import SyncWorker, HFTokenDialog

log = logging.getLogger(__name__)

_KEY_LAST_DIR  = 'warp_core/last_dir'
_KEY_HF_TOKEN  = 'warp_core/hf_token'


class WarpCoreWindow(QMainWindow):
    """
    WARP CORE trainer window.
    Left: screenshot file list + progress.
    Right: AnnotationWidget canvas (draw bboxes, click existing, delete).
    Bottom: slot selector, item name entry, accept/skip buttons.
    Toolbar: Open, Save, Auto-Detect, Sync to Hub, (Train future).
    """

    def __init__(self, sets_app=None, parent=None):
        super().__init__(parent)
        self._sets      = sets_app
        self._settings  = QSettings()
        self._sets_root = self._find_sets_root()
        self._data_mgr  = TrainingDataManager(
            self._sets_root / 'warp' / 'training_data')
        self._screenshots: list[Path] = []
        self._current_idx = -1

        self.setWindowTitle('WARP CORE — ML Trainer')
        self.setMinimumSize(1100, 700)
        self._build_ui()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            'Ready — open a folder of STO screenshots to start annotating.')

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        root = QHBoxLayout(c)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(sp)

        # Left panel
        left = QWidget(); left.setFixedWidth(220)
        ll   = QVBoxLayout(left); ll.setContentsMargins(8, 8, 8, 8)
        lbl  = QLabel('Screenshots')
        lbl.setFont(QFont('', 10, QFont.Weight.Bold))
        self._file_list = QListWidget()
        self._file_list.currentRowChanged.connect(self._load_screenshot)
        self._prog_lbl  = QLabel('0 / 0 annotated')
        self._prog_lbl.setStyleSheet('color:#aaa;font-size:10px;')
        self._prog_bar  = QProgressBar()
        self._prog_bar.setRange(0, 100); self._prog_bar.setValue(0)
        self._prog_bar.setFixedHeight(6); self._prog_bar.setTextVisible(False)
        for w in (lbl, self._file_list, self._prog_lbl, self._prog_bar):
            ll.addWidget(w, 1 if w is self._file_list else 0)

        # Right panel
        right   = QWidget()
        rl      = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)

        self._ann_widget = AnnotationWidget(self._data_mgr)
        self._ann_widget.annotation_added.connect(self._on_bbox_drawn)
        self._ann_widget.item_selected.connect(self._on_item_selected)
        rl.addWidget(self._ann_widget, 1)
        rl.addWidget(self._make_bottom_panel())

        sp.addWidget(left); sp.addWidget(right)
        sp.setStretchFactor(1, 1)

    def _make_bottom_panel(self) -> QGroupBox:
        g   = QGroupBox('Annotate Selected Icon')
        g.setFixedHeight(110)
        lay = QHBoxLayout(g); lay.setSpacing(10)

        # Slot combo
        sc = QVBoxLayout()
        sc.addWidget(QLabel('Slot:'))
        self._slot_combo = QComboBox(); self._slot_combo.setFixedWidth(180)
        from warp.warp_importer import SPACE_SLOTS
        for name, _ in SPACE_SLOTS:
            self._slot_combo.addItem(name)
        sc.addWidget(self._slot_combo)
        lay.addLayout(sc)

        # Name entry
        nc = QVBoxLayout()
        nc.addWidget(QLabel('Item name:'))
        nr = QHBoxLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Item name (or leave blank for 'Unknown')")
        self._name_edit.returnPressed.connect(self._on_accept)
        btn_search = QPushButton('🔍'); btn_search.setFixedWidth(30)
        btn_search.clicked.connect(self._on_search)
        nr.addWidget(self._name_edit, 1); nr.addWidget(btn_search)
        nc.addLayout(nr)
        lay.addLayout(nc, 1)

        # Buttons
        bc = QVBoxLayout(); bc.addStretch()
        br = QHBoxLayout()
        self._btn_accept = QPushButton('✓  Accept')
        self._btn_accept.setStyleSheet(
            'QPushButton{background:#1a5c3a;color:#7effc8;'
            'border:1px solid #3aac6a;border-radius:3px;padding:5px 12px;font-weight:bold;}'
            'QPushButton:hover{background:#2a8c5a;}')
        self._btn_accept.clicked.connect(self._on_accept)
        self._btn_skip = QPushButton('Skip →')
        self._btn_skip.clicked.connect(self._on_skip)
        br.addWidget(self._btn_accept); br.addWidget(self._btn_skip)
        bc.addLayout(br)
        lay.addLayout(bc)
        return g

    def _build_toolbar(self):
        tb = QToolBar('Main'); tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        def act(icon, tip, slot):
            a = QAction(icon, self)
            a.setToolTip(tip); a.triggered.connect(slot)
            tb.addAction(a); tb.addSeparator()
            return a

        act('📂  Open Folder', 'Open screenshots folder', self._on_open)
        act('💾  Save',        'Save annotations locally', self._on_save)
        act('⚡  Auto-Detect', 'Auto-detect icons in all screenshots', self._on_auto_detect)
        act('☁  Sync to Hub', 'Upload annotations to Hugging Face Hub', self._on_sync)

    # ── File handling ────────────────────────────────────────────────────────

    def _on_open(self):
        last = self._settings.value(_KEY_LAST_DIR, '')
        dlg  = QFileDialog(self)
        dlg.setWindowTitle('Open Screenshots Folder')
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if last and Path(last).is_dir():
            dlg.setDirectory(last)
        if not dlg.exec():
            return
        folder = Path(dlg.selectedFiles()[0])
        self._settings.setValue(_KEY_LAST_DIR, str(folder))
        exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        self._screenshots = sorted([f for f in folder.iterdir()
                                     if f.suffix.lower() in exts])
        self._file_list.clear()
        for p in self._screenshots:
            item = QListWidgetItem(p.name)
            if self._data_mgr.has_annotations(p):
                item.setForeground(Qt.GlobalColor.green)
            self._file_list.addItem(item)
        if self._screenshots:
            self._file_list.setCurrentRow(0)
        self._update_progress()
        self.statusBar().showMessage(
            f'Loaded {len(self._screenshots)} screenshot(s) from {folder}')

    def _load_screenshot(self, row: int):
        if row < 0 or row >= len(self._screenshots):
            return
        self._current_idx = row
        self._ann_widget.load_image(self._screenshots[row])
        p = self._screenshots[row]
        self.statusBar().showMessage(f'{p.name}  ({row+1}/{len(self._screenshots)})')

    # ── Auto-detect ──────────────────────────────────────────────────────────

    def _on_auto_detect(self):
        if not self._screenshots:
            return
        from warp.recognition.layout_detector import LayoutDetector
        import cv2
        det = LayoutDetector()
        new = 0
        for path in self._screenshots:
            img = cv2.imread(str(path))
            if img is None:
                continue
            for slot_name, bboxes in det.detect(img, 'SPACE').items():
                for i, bbox in enumerate(bboxes):
                    if self._data_mgr.add_candidate(path, slot_name, i, bbox):
                        new += 1
        self._update_progress()
        self._load_screenshot(self._current_idx)
        self.statusBar().showMessage(f'Auto-detect done — {new} candidates added.')

    # ── Annotation callbacks ─────────────────────────────────────────────────

    def _on_bbox_drawn(self, bbox: tuple):
        self._name_edit.setFocus(); self._name_edit.clear()

    def _on_item_selected(self, ann: dict):
        self._slot_combo.setCurrentText(ann.get('slot', ''))
        self._name_edit.setText(ann.get('name', ''))

    def _on_accept(self):
        slot = self._slot_combo.currentText()
        name = self._name_edit.text().strip()
        self._ann_widget.confirm_current(slot=slot, name=name)
        self._name_edit.clear()
        self._update_progress()
        if self._ann_widget.all_confirmed():
            next_row = self._current_idx + 1
            if next_row < len(self._screenshots):
                self._file_list.setCurrentRow(next_row)

    def _on_skip(self):
        self._ann_widget.skip_current()
        self._name_edit.clear()

    def _on_search(self):
        """Search SETS cache for item names matching typed text."""
        query = self._name_edit.text().strip().lower()
        candidates: list[str] = []
        if self._sets:
            for cat_items in self._sets.cache.equipment.values():
                for name in cat_items:
                    if query in name.lower():
                        candidates.append(name)
            for name in self._sets.cache.starship_traits:
                if query in name.lower():
                    candidates.append(name)
        if not candidates:
            self.statusBar().showMessage('No matching items found in SETS cache.')
            return
        candidates = sorted(set(candidates))[:40]
        chosen, ok = QInputDialog.getItem(
            self, 'Search Item', 'Select item:', candidates, 0, False)
        if ok and chosen:
            self._name_edit.setText(chosen)

    # ── Save / Sync ──────────────────────────────────────────────────────────

    def _on_save(self):
        self._data_mgr.save()
        self._update_progress()
        self.statusBar().showMessage('Annotations saved.')

    def _on_sync(self):
        token = self._settings.value(_KEY_HF_TOKEN, '')
        if not token:
            dlg = HFTokenDialog(self)
            if dlg.exec():
                token = dlg.get_token()
                self._settings.setValue(_KEY_HF_TOKEN, token)
            else:
                return
        self._sync_worker = SyncWorker(
            data_manager=self._data_mgr, hf_token=token, mode='upload')
        self._sync_worker.progress.connect(
            lambda p, m: self.statusBar().showMessage(f'Sync: {m} ({p}%)'))
        self._sync_worker.finished.connect(
            lambda ok: self.statusBar().showMessage(
                'Sync complete.' if ok else 'Sync failed — check log.'))
        self._sync_worker.start()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _update_progress(self):
        total     = len(self._screenshots)
        annotated = sum(1 for p in self._screenshots
                        if self._data_mgr.has_annotations(p))
        self._prog_lbl.setText(f'{annotated} / {total} annotated')
        self._prog_bar.setValue(int(100 * annotated / max(1, total)))
        # Green file list items for annotated files
        for row, p in enumerate(self._screenshots):
            item = self._file_list.item(row)
            if item:
                item.setForeground(
                    Qt.GlobalColor.green
                    if self._data_mgr.has_annotations(p)
                    else Qt.GlobalColor.white)

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(8):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')
