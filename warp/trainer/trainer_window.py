# warp/trainer/trainer_window.py
# WARP CORE — Interactive ML trainer + recognition review.
# PySide6, integrated with SETS.
#
# Layout (splitter):
#   LEFT   — screenshot file list + progress
#   CENTER — AnnotationWidget canvas (draw bboxes, click, delete)
#   RIGHT  — Recognition review panel:
#              * List of auto-detected items (name, slot, confidence)
#              * Confirm / Edit / Reject per item
#              * "Edit bbox" button → click+drag on canvas to correct

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QComboBox, QLineEdit, QGroupBox,
    QProgressBar, QToolBar, QStatusBar, QMessageBox,
    QInputDialog, QSizePolicy, QFrame,
    QAbstractItemView
)
from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtGui import QFont, QAction, QColor

from warp.trainer.annotation_widget import AnnotationWidget
from warp.trainer.training_data      import TrainingDataManager, AnnotationState
from warp.trainer.sync               import SyncWorker, HFTokenDialog

log = logging.getLogger(__name__)

_KEY_LAST_DIR = 'warp_core/last_dir'
_KEY_HF_TOKEN = 'warp_core/hf_token'

CONF_HIGH   = 0.85
CONF_MEDIUM = 0.70


class WarpCoreWindow(QMainWindow):
    """
    WARP CORE trainer window.

    Left:   screenshot file list + progress bar.
    Center: AnnotationWidget canvas — draw bboxes, click existing, delete.
    Right:  Recognition review panel — list of detected items with
            Confirm / Edit bbox / Reject, plus manual bbox drawing mode.
    Bottom (center): slot selector + item name entry + Accept/Skip.
    Toolbar: Open, Save, Auto-Detect, Sync to Hub.
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
        self._recognition_items: list[dict] = []
        self._manual_bbox_mode = False

        self.setWindowTitle('WARP CORE — ML Trainer')
        self.setMinimumSize(1280, 740)
        self._build_ui()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            'Ready — open a folder of STO screenshots to start annotating.')

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QHBoxLayout(c)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(sp)

        sp.addWidget(self._make_left_panel())
        sp.addWidget(self._make_center_panel())
        sp.addWidget(self._make_right_panel())
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 0)
        sp.setSizes([200, 800, 300])

    def _make_left_panel(self) -> QWidget:
        left = QWidget()
        left.setFixedWidth(200)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)

        lbl = QLabel('Screenshots')
        lbl.setFont(QFont('', 10, QFont.Weight.Bold))

        self._file_list = QListWidget()
        self._file_list.currentRowChanged.connect(self._load_screenshot)

        self._prog_lbl = QLabel('0 / 0 annotated')
        self._prog_lbl.setStyleSheet('color:#aaa;font-size:10px;')
        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setTextVisible(False)

        for w in (lbl, self._file_list, self._prog_lbl, self._prog_bar):
            ll.addWidget(w, 1 if w is self._file_list else 0)
        return left

    def _make_center_panel(self) -> QWidget:
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        self._ann_widget = AnnotationWidget(self._data_mgr)
        self._ann_widget.annotation_added.connect(self._on_bbox_drawn)
        self._ann_widget.item_selected.connect(self._on_item_selected)
        cl.addWidget(self._ann_widget, 1)
        cl.addWidget(self._make_bottom_panel())
        return center

    def _make_bottom_panel(self) -> QGroupBox:
        g = QGroupBox('Annotate Selected Icon')
        g.setFixedHeight(110)
        lay = QHBoxLayout(g)
        lay.setSpacing(10)

        sc = QVBoxLayout()
        sc.addWidget(QLabel('Slot:'))
        self._slot_combo = QComboBox()
        self._slot_combo.setFixedWidth(180)
        from warp.warp_importer import SPACE_SLOTS
        for name, _ in SPACE_SLOTS:
            self._slot_combo.addItem(name)
        sc.addWidget(self._slot_combo)
        lay.addLayout(sc)

        nc = QVBoxLayout()
        nc.addWidget(QLabel('Item name:'))
        nr = QHBoxLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Item name (or leave blank for 'Unknown')")
        self._name_edit.returnPressed.connect(self._on_accept)
        btn_search = QPushButton('🔍')
        btn_search.setFixedWidth(30)
        btn_search.clicked.connect(self._on_search)
        nr.addWidget(self._name_edit, 1)
        nr.addWidget(btn_search)
        nc.addLayout(nr)
        lay.addLayout(nc, 1)

        bc = QVBoxLayout()
        bc.addStretch()
        br = QHBoxLayout()
        self._btn_accept = QPushButton('✓  Accept')
        self._btn_accept.setStyleSheet(
            'QPushButton{background:#1a5c3a;color:#7effc8;'
            'border:1px solid #3aac6a;border-radius:3px;padding:5px 12px;font-weight:bold;}'
            'QPushButton:hover{background:#2a8c5a;}')
        self._btn_accept.clicked.connect(self._on_accept)
        self._btn_skip = QPushButton('Skip →')
        self._btn_skip.clicked.connect(self._on_skip)
        br.addWidget(self._btn_accept)
        br.addWidget(self._btn_skip)
        bc.addLayout(br)
        lay.addLayout(bc)
        return g

    def _make_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(360)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(6, 8, 6, 8)
        pl.setSpacing(6)

        hdr = QLabel('Recognition Review')
        hdr.setFont(QFont('', 10, QFont.Weight.Bold))
        hdr.setStyleSheet('color:#7ec8e3;')
        pl.addWidget(hdr)

        hint = QLabel(
            'Green = high confidence\n'
            'Yellow = uncertain  ·  Red = unmatched\n'
            'Click item → select on canvas. Edit bbox to redraw.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#888;font-size:10px;')
        pl.addWidget(hint)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color:#333;')
        pl.addWidget(sep)

        self._review_list = QListWidget()
        self._review_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._review_list.currentRowChanged.connect(self._on_review_row_changed)
        pl.addWidget(self._review_list, 1)

        self._review_summary = QLabel('')
        self._review_summary.setStyleSheet('color:#aaa;font-size:10px;')
        self._review_summary.setWordWrap(True)
        pl.addWidget(self._review_summary)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet('color:#333;')
        pl.addWidget(sep2)

        btn_row = QHBoxLayout()
        self._btn_confirm = QPushButton('✓ Confirm')
        self._btn_confirm.setStyleSheet(
            'QPushButton{background:#1a5c3a;color:#7effc8;border:1px solid #3aac6a;'
            'border-radius:3px;padding:4px 8px;}'
            'QPushButton:hover{background:#2a8c5a;}')
        self._btn_confirm.clicked.connect(self._on_review_confirm)

        self._btn_edit_bbox = QPushButton('✏ Edit bbox')
        self._btn_edit_bbox.setStyleSheet(
            'QPushButton{background:#1a3a5c;color:#7ec8e3;border:1px solid #3a6aac;'
            'border-radius:3px;padding:4px 8px;}'
            'QPushButton:hover{background:#2a5a8c;}'
            'QPushButton:checked{background:#0a2a4c;border:2px solid #7ec8e3;}')
        self._btn_edit_bbox.setCheckable(True)
        self._btn_edit_bbox.clicked.connect(self._on_edit_bbox_toggle)

        self._btn_reject = QPushButton('✗ Reject')
        self._btn_reject.setStyleSheet(
            'QPushButton{background:#3a1a1a;color:#ff9999;border:1px solid #ac3a3a;'
            'border-radius:3px;padding:4px 8px;}'
            'QPushButton:hover{background:#5a2a2a;}')
        self._btn_reject.clicked.connect(self._on_review_reject)

        btn_row.addWidget(self._btn_confirm)
        btn_row.addWidget(self._btn_edit_bbox)
        btn_row.addWidget(self._btn_reject)
        pl.addLayout(btn_row)

        self._manual_mode_lbl = QLabel('')
        self._manual_mode_lbl.setStyleSheet(
            'color:#e8c060;font-size:10px;background:#2a2a00;'
            'border:1px solid #6a6000;border-radius:3px;padding:3px;')
        self._manual_mode_lbl.setWordWrap(True)
        self._manual_mode_lbl.setVisible(False)
        pl.addWidget(self._manual_mode_lbl)

        self._set_review_buttons_enabled(False)
        return panel

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar('Main')
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        def act(icon, tip, slot):
            a = QAction(icon, self)
            a.setToolTip(tip)
            a.triggered.connect(slot)
            tb.addAction(a)
            tb.addSeparator()
            return a

        act('📂  Open Folder', 'Open screenshots folder',                     self._on_open)
        act('💾  Save',        'Save annotations locally',                     self._on_save)
        act('⚡  Auto-Detect', 'Auto-detect icons in all screenshots',         self._on_auto_detect)
        act('☁  Sync to Hub', 'Upload annotations to Hugging Face Hub',        self._on_sync)

    # ── File handling ─────────────────────────────────────────────────────────

    def _on_open(self):
        last = self._settings.value(_KEY_LAST_DIR, '')
        dlg  = QFileDialog(self)
        dlg.setWindowTitle('Open Screenshots Folder')
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilter('Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)')
        if last and Path(last).is_dir():
            dlg.setDirectory(last)
        from PySide6.QtWidgets import QListView, QTreeView
        for view in dlg.findChildren(QListView) + dlg.findChildren(QTreeView):
            view.setSelectionMode(view.SelectionMode.NoSelection)
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
        path = self._screenshots[row]
        self._ann_widget.load_image(path)
        self._exit_manual_bbox_mode()
        self._run_recognition(path)
        self.statusBar().showMessage(f'{path.name}  ({row+1}/{len(self._screenshots)})')

    # ── Auto-detect ───────────────────────────────────────────────────────────

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

    # ── Recognition review ────────────────────────────────────────────────────

    def _run_recognition(self, path: Path):
        """Run WARP recognition on current screenshot and populate review panel."""
        self._recognition_items = []
        self._review_list.clear()
        self._review_summary.setText('')
        self._set_review_buttons_enabled(False)

        try:
            import cv2
            from warp.warp_importer import WarpImporter
            importer = WarpImporter(sets_app=self._sets, build_type='SPACE')
            img = cv2.imread(str(path))
            if img is None:
                return
            result = importer._process_image(img, str(path))
        except Exception as e:
            self.statusBar().showMessage(f'Recognition error: {e}')
            log.exception('WARP CORE recognition error')
            return

        for ri in result.items:
            self._recognition_items.append({
                'name':  ri.name,
                'slot':  ri.slot,
                'conf':  ri.confidence,
                'bbox':  ri.bbox,
                'state': 'pending',
                'thumb': ri.thumbnail,
            })
            self._add_review_row(ri.name, ri.slot, ri.confidence)

        n       = len(result.items)
        matched = sum(1 for i in result.items if i.name)
        self._review_summary.setText(
            f'{matched}/{n} identified  ·  Ship: {result.ship_name or "unknown"}')
        self._set_review_buttons_enabled(n > 0)
        if n > 0:
            self._review_list.setCurrentRow(0)

    def _add_review_row(self, name: str, slot: str, conf: float):
        label = f'{slot}  →  {name or "— unmatched —"}  [{conf:.0%}]'
        item  = QListWidgetItem(label)
        if not name:
            item.setForeground(QColor('#ff7e7e'))
        elif conf >= CONF_HIGH:
            item.setForeground(QColor('#7effc8'))
        elif conf >= CONF_MEDIUM:
            item.setForeground(QColor('#e8c060'))
        else:
            item.setForeground(QColor('#ff9966'))
        self._review_list.addItem(item)

    def _on_review_row_changed(self, row: int):
        if row < 0 or row >= len(self._recognition_items):
            self._set_review_buttons_enabled(False)
            return
        self._set_review_buttons_enabled(True)
        ri = self._recognition_items[row]
        self._slot_combo.setCurrentText(ri['slot'])
        self._name_edit.setText(ri['name'])
        if ri.get('bbox'):
            self._ann_widget.highlight_bbox(ri['bbox'])

    def _on_review_confirm(self):
        row = self._review_list.currentRow()
        if row < 0 or row >= len(self._recognition_items):
            return
        ri   = self._recognition_items[row]
        slot = self._slot_combo.currentText()
        name = self._name_edit.text().strip()
        ri['name']  = name
        ri['slot']  = slot
        ri['state'] = 'confirmed'
        if ri.get('bbox') and self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            self._data_mgr.add_annotation(
                image_path=path,
                bbox=ri['bbox'],
                slot=slot,
                name=name,
                state=AnnotationState.CONFIRMED,
            )
        litem = self._review_list.item(row)
        if litem:
            litem.setText(f'{slot}  →  {name or "— unmatched —"}  [✓ confirmed]')
            litem.setForeground(QColor('#7effc8'))
        self._update_progress()
        self._advance_to_next_unconfirmed(row)

    def _on_review_reject(self):
        row = self._review_list.currentRow()
        if row < 0 or row >= len(self._recognition_items):
            return
        ri = self._recognition_items[row]
        ri['state'] = 'rejected'
        litem = self._review_list.item(row)
        if litem:
            litem.setText(f'{ri["slot"]}  →  [✗ rejected]')
            litem.setForeground(QColor('#888888'))
        self._advance_to_next_unconfirmed(row)

    def _on_edit_bbox_toggle(self, checked: bool):
        if checked:
            self._enter_manual_bbox_mode()
        else:
            self._exit_manual_bbox_mode()

    def _enter_manual_bbox_mode(self):
        self._manual_bbox_mode = True
        self._btn_edit_bbox.setChecked(True)
        row  = self._review_list.currentRow()
        slot = self._recognition_items[row]['slot'] if row >= 0 else '?'
        self._manual_mode_lbl.setText(
            f'📌 Draw a rectangle on the screenshot to redefine region for:\n{slot}')
        self._manual_mode_lbl.setVisible(True)
        self._ann_widget.set_draw_mode(True)
        self.statusBar().showMessage(
            'Manual bbox mode — drag a rectangle on the image.')

    def _exit_manual_bbox_mode(self):
        self._manual_bbox_mode = False
        self._btn_edit_bbox.setChecked(False)
        self._manual_mode_lbl.setVisible(False)
        self._ann_widget.set_draw_mode(False)

    def _advance_to_next_unconfirmed(self, current_row: int):
        for i in range(current_row + 1, len(self._recognition_items)):
            if self._recognition_items[i]['state'] == 'pending':
                self._review_list.setCurrentRow(i)
                return

    def _set_review_buttons_enabled(self, enabled: bool):
        for btn in (self._btn_confirm, self._btn_edit_bbox, self._btn_reject):
            btn.setEnabled(enabled)

    # ── Canvas callbacks ──────────────────────────────────────────────────────

    def _on_bbox_drawn(self, bbox: tuple):
        if self._manual_bbox_mode:
            row = self._review_list.currentRow()
            if 0 <= row < len(self._recognition_items):
                self._recognition_items[row]['bbox'] = bbox
            self._exit_manual_bbox_mode()
            self._rematch_current_item(row, bbox)
        else:
            self._name_edit.setFocus()
            self._name_edit.clear()

    def _rematch_current_item(self, row: int, bbox: tuple):
        if row < 0 or self._current_idx < 0:
            return
        try:
            import cv2
            from warp.recognition.icon_matcher import SETSIconMatcher
            path = self._screenshots[self._current_idx]
            img  = cv2.imread(str(path))
            if img is None:
                return
            x, y, w, h = bbox
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                return
            matcher     = SETSIconMatcher(self._sets)
            name, conf, thumb = matcher.match(crop)
            ri = self._recognition_items[row]
            ri.update({'name': name, 'conf': conf, 'thumb': thumb})
            self._name_edit.setText(name)
            litem = self._review_list.item(row)
            if litem:
                colour = ('#7effc8' if conf >= CONF_HIGH
                          else '#e8c060' if conf >= CONF_MEDIUM
                          else '#ff7e7e')
                litem.setText(f'{ri["slot"]}  →  {name or "— unmatched —"}  [{conf:.0%}]')
                litem.setForeground(QColor(colour))
            self.statusBar().showMessage(
                f'Rematched: {name or "no match"} ({conf:.0%})')
        except Exception as e:
            log.exception('Rematch error')
            self.statusBar().showMessage(f'Rematch error: {e}')

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

    # ── Save / Sync ───────────────────────────────────────────────────────────

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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_progress(self):
        total     = len(self._screenshots)
        annotated = sum(1 for p in self._screenshots
                        if self._data_mgr.has_annotations(p))
        self._prog_lbl.setText(f'{annotated} / {total} annotated')
        self._prog_bar.setValue(int(100 * annotated / max(1, total)))
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
