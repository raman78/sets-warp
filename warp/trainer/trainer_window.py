# warp/trainer/trainer_window.py
# WARP CORE — Interactive ML trainer + recognition review.
# PySide6, integrated with SETS.
#
# KEY FEATURES:
#   - Multi-screen build support: each screenshot can be a different type
#     (SPACE equipment, TRAITS, BOFFS, SPEC, GROUND)
#   - Auto-detects build type per screenshot from OCR/layout heuristics
#   - Slot combo dynamically updates based on detected screen type
#   - All slot types supported: SPACE, GROUND, TRAITS, BOFFS, SPEC
#   - Specializations (Primary/Secondary) fully handled
#
# Layout (splitter):
#   LEFT   — screenshot file list + per-file type badge + type override
#   CENTER — AnnotationWidget canvas (draw bboxes, click, delete)
#   RIGHT  — Recognition review panel:
#              * List of auto-detected items (name, slot, confidence)
#              * Confirm / Edit / Reject per item
#              * "Edit bbox" button -> click+drag on canvas to correct

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QComboBox, QLineEdit, QGroupBox,
    QProgressBar, QToolBar, QStatusBar, QMessageBox,
    QInputDialog, QSizePolicy, QFrame, QScrollArea,
    QAbstractItemView, QCompleter
)
from PySide6.QtCore import Qt, QSettings, QThread, Signal, QSortFilterProxyModel
from PySide6.QtGui import QFont, QAction, QColor, QStandardItemModel, QStandardItem

from warp.trainer.annotation_widget import AnnotationWidget
from warp.trainer.training_data      import TrainingDataManager, AnnotationState
from warp.trainer.sync               import SyncWorker, HFTokenDialog

log = logging.getLogger(__name__)

_KEY_LAST_DIR = 'warp_core/last_dir'
_KEY_HF_TOKEN = 'warp_core/hf_token'

CONF_HIGH   = 0.85
CONF_MEDIUM = 0.70

# ── All available slots grouped by screen type --------------------------------
# Used to populate the slot combo based on detected screen type.

SLOT_GROUPS: dict[str, list[str]] = {
    'SPACE': [
        'Fore Weapons',
        'Deflector',
        'Sec-Def',
        'Engines',
        'Warp Core',
        'Shield',
        'Aft Weapons',
        'Experimental',
        'Devices',
        'Universal Consoles',
        'Engineering Consoles',
        'Science Consoles',
        'Tactical Consoles',
        'Hangars',
    ],
    'GROUND': [
        'Body Armor',
        'EV Suit',
        'Personal Shield',
        'Weapons',
        'Kit',
        'Kit Modules',
        'Ground Devices',
    ],
    'SPACE_TRAITS': [
        'Personal Space Traits',
        'Starship Traits',
        'Space Reputation',
        'Active Space Rep',
    ],
    'GROUND_TRAITS': [
        'Personal Ground Traits',
        'Ground Reputation',
        'Active Ground Rep',
    ],
    'BOFFS': [
        'Boff Tactical',
        'Boff Engineering',
        'Boff Science',
        'Boff Operations',
        'Boff Intelligence',
        'Boff Command',
        'Boff Pilot',
        'Boff Miracle Worker',
        'Boff Temporal',
    ],
    'SPEC': [
        'Primary Specialization',
        'Secondary Specialization',
    ],
}

# Human-readable labels per screen type
SCREEN_TYPE_LABELS: dict[str, str] = {
    'SPACE':         'Space Equipment',
    'GROUND':        'Ground Equipment',
    'SPACE_TRAITS':  'Space Traits',
    'GROUND_TRAITS': 'Ground Traits',
    'BOFFS':         'Bridge Officers',
    'SPEC':          'Specializations',
    'SPACE_MIXED':   'Space Mixed (merged)',
    'GROUND_MIXED':  'Ground Mixed (merged)',
    'UNKNOWN':       'Unknown',
}

SCREEN_TYPE_ICONS: dict[str, str] = {
    'SPACE':         '🚀',
    'GROUND':        '🦶',
    'SPACE_TRAITS':  '✨',
    'GROUND_TRAITS': '🌿',
    'BOFFS':         '👥',
    'SPEC':          '🎯',
    'SPACE_MIXED':   '🌌',
    'GROUND_MIXED':  '🗺️',
    'UNKNOWN':       '❓',
}

# Map detected build_type -> which SLOT_GROUPS key to use in combo
SCREEN_TO_SLOT_GROUP: dict[str, str] = {
    'SPACE':         'SPACE',
    'GROUND':        'GROUND',
    'SPACE_TRAITS':  'SPACE_TRAITS',
    'GROUND_TRAITS': 'GROUND_TRAITS',
    'BOFFS':         'BOFFS',
    'SPEC':          'SPEC',
    'SPACE_MIXED':   'ALL',
    'GROUND_MIXED':  'ALL',
    'UNKNOWN':       'SPACE',
}

# Slots that represent free-text strings, not icons.
# Matching / crop logic is skipped for these.
# Slots whose value is a free-text string (no icon matching / no autocomplete)
TEXT_SLOTS: frozenset[str] = frozenset(['Ship Name'])
# Slots with a fixed/searchable dropdown of values (no icon matching)
FIXED_VALUE_SLOTS: frozenset[str] = frozenset(['Ship Tier', 'Ship Type'])
# All non-icon slots combined
NON_ICON_SLOTS: frozenset[str] = TEXT_SLOTS | FIXED_VALUE_SLOTS

SHIP_TIER_VALUES: list[str] = [
    'T1', 'T2', 'T3', 'T4',
    'T5', 'T5-U', 'T5-X', 'T5-X2',
    'T6', 'T6-X', 'T6-X2',
]

# Ship info slots are available on every equipment screen
_SHIP_INFO_SLOTS = ['Ship Name', 'Ship Type', 'Ship Tier']
for _grp in ('SPACE', 'GROUND', 'ALL'):
    pass   # added below after ALL_SLOTS is built

# All slots combined (fallback + MIXED screens)
ALL_SLOTS: list[str] = []
for _slots in SLOT_GROUPS.values():
    for _s in _slots:
        if _s not in ALL_SLOTS:
            ALL_SLOTS.append(_s)
# Append ship info slots to groups where they are relevant
for _grp_key in ('SPACE', 'GROUND'):
    for _s in _SHIP_INFO_SLOTS:
        if _s not in SLOT_GROUPS[_grp_key]:
            SLOT_GROUPS[_grp_key].append(_s)
# Build ALL after extending SPACE/GROUND
ALL_SLOTS = []
for _slots in SLOT_GROUPS.values():
    for _s in _slots:
        if _s not in ALL_SLOTS:
            ALL_SLOTS.append(_s)
for _s in _SHIP_INFO_SLOTS:
    if _s not in ALL_SLOTS:
        ALL_SLOTS.append(_s)
SLOT_GROUPS['ALL'] = ALL_SLOTS

# Specialization names known to WARP (from STO wiki)
SPECIALIZATION_NAMES: list[str] = [
    'Command Officer',
    'Intelligence Officer',
    'Miracle Worker',
    'Pilot',
    'Temporal Operative',
    'Constable',
    'Commando',
    'Strategist',
]


# ── Background workers ────────────────────────────────────────────────────────

class ScreenTypeDetectorWorker(QThread):
    """
    Detects screen type for each screenshot using OCR.
    Runs in background — never blocks the UI.

    Signals:
        progress(index, total, filename, detected_type)
        finished(results: dict[str, str])   filename -> screen type
    """
    progress = Signal(int, int, str, str)    # idx, total, filename, stype
    finished = Signal(dict)                  # {filename: stype}

    def __init__(self, paths: list, models_dir=None, parent=None):
        super().__init__(parent)
        self._paths      = paths
        self._models_dir = models_dir

    def run(self):
        results: dict[str, str] = {}
        total = len(self._paths)

        # ML classifier only — OCR is no longer used for screen type detection.
        # OCR is reserved for reading text from labelled bboxes (e.g. Ship Name).
        classifier = None
        if self._models_dir is not None:
            try:
                from warp.recognition.screen_classifier import ScreenTypeClassifier
                classifier = ScreenTypeClassifier(self._models_dir)
            except Exception as e:
                log.debug(f'ScreenTypeDetector: classifier init failed: {e}')

        import cv2
        for idx, path in enumerate(self._paths):
            if self.isInterruptionRequested():
                break
            stype = 'UNKNOWN'
            try:
                img = cv2.imread(str(path))
                if img is not None and classifier is not None:
                    ml_stype, ml_conf = classifier.classify(img)
                    if ml_stype and ml_conf >= 0.70:
                        stype = ml_stype
            except Exception as e:
                log.debug(f'Screen type detection error for {path.name}: {e}')
            results[path.name] = stype
            self.progress.emit(idx + 1, total, path.name, stype)

        self.finished.emit(results)


class RecognitionWorker(QThread):
    """
    Runs WARP recognition (icon matching) on a single screenshot.
    Used when user manually changes the screen type override.

    Signals:
        finished(items: list[dict])   list of recognition result dicts
        error(message: str)
    """
    finished = Signal(list)
    error    = Signal(str)

    def __init__(self, path, stype: str, sets_app, parent=None):
        super().__init__(parent)
        self._path     = path
        self._stype    = stype
        self._sets_app = sets_app

    def run(self):
        importer_type = {
            'SPACE':         'SPACE',
            'GROUND':        'GROUND',
            'SPACE_TRAITS':  'SPACE_TRAITS',
            'GROUND_TRAITS': 'GROUND_TRAITS',
            'BOFFS':         'BOFFS',
            'SPEC':          'SPEC',
            'SPACE_MIXED':   'SPACE',
            'GROUND_MIXED':  'GROUND',
        }.get(self._stype, 'SPACE')

        try:
            import cv2
            from warp.warp_importer import WarpImporter
            importer = WarpImporter(
                sets_app=self._sets_app, build_type=importer_type)
            img = cv2.imread(str(self._path))
            if img is None:
                self.finished.emit([])
                return
            result = importer._process_image(img, str(self._path))
        except Exception as e:
            log.exception('RecognitionWorker error')
            self.error.emit(str(e))
            return

        items = []
        try:
            import cv2
            img2 = cv2.imread(str(self._path))
        except Exception:
            img2 = None

        for ri in result.items:
            crop_bgr = None
            if ri.bbox and img2 is not None:
                try:
                    x, y, w, h = ri.bbox
                    crop_bgr = img2[y:y+h, x:x+w].copy()
                except Exception:
                    pass
            items.append({
                'name':      ri.name,
                'slot':      ri.slot,
                'conf':      ri.confidence,
                'bbox':      ri.bbox,
                'state':     'pending',
                'thumb':     ri.thumbnail,
                'crop_bgr':  crop_bgr,
                'orig_name': ri.name,
                'ship_name': result.ship_name,
            })
        self.finished.emit(items)


class _DetectProgressDialog(QWidget):
    """
    Floating progress dialog shown during screen type detection.
    Has a Cancel button that requests interruption of the worker thread.
    """
    cancelled = Signal()

    def __init__(self, total: int, parent=None):
        super().__init__(parent,
                         Qt.WindowType.Window |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('WARP CORE — Detecting Screen Types')
        self.setFixedSize(460, 140)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        self._title_lbl = QLabel('Analysing screenshots with OCR…')
        self._title_lbl.setFont(QFont('', 10, QFont.Weight.Bold))
        self._title_lbl.setStyleSheet('color:#7ec8e3;')

        self._file_lbl = QLabel('')
        self._file_lbl.setStyleSheet('color:#aaa;font-size:10px;')
        self._file_lbl.setWordWrap(True)

        self._bar = QProgressBar()
        self._bar.setRange(0, total)
        self._bar.setValue(0)

        btn_cancel = QPushButton('Cancel')
        btn_cancel.setFixedWidth(80)
        btn_cancel.clicked.connect(self.cancelled.emit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)

        lay.addWidget(self._title_lbl)
        lay.addWidget(self._file_lbl)
        lay.addWidget(self._bar)
        lay.addLayout(btn_row)

    def update_progress(self, idx: int, total: int, filename: str, stype: str):
        self._bar.setValue(idx)
        icon  = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        self._file_lbl.setText(f'{filename}  →  {icon} {label}')


class _RecognitionProgressDialog(QWidget):
    cancelled = Signal()

    def __init__(self, filename: str, stype: str, parent=None):
        super().__init__(parent,
                         Qt.WindowType.Window |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('WARP CORE -- Recognising Icons')
        self.setFixedSize(420, 130)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        icon  = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, stype)
        title = QLabel('Matching icons against SETS library...')
        title.setFont(QFont('', 10, QFont.Weight.Bold))
        title.setStyleSheet('color:#7ec8e3;')
        file_lbl = QLabel(f'{icon} {label}   {filename}')
        file_lbl.setStyleSheet('color:#aaa;font-size:10px;')
        file_lbl.setWordWrap(True)
        bar = QProgressBar()
        bar.setRange(0, 0)   # indeterminate busy indicator
        bar.setFixedHeight(12)
        btn_cancel = QPushButton('Cancel')
        btn_cancel.setFixedWidth(80)
        btn_cancel.clicked.connect(self.cancelled.emit)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        lay.addWidget(title)
        lay.addWidget(file_lbl)
        lay.addWidget(bar)
        lay.addLayout(btn_row)


class _TrainProgressDialog(QWidget):
    """
    Progress dialog for local model training.
    Shows epoch progress bar + live log lines + Cancel button.
    """
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent,
                         Qt.WindowType.Window |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('WARP CORE -- Training Icon Classifier')
        self.setFixedSize(520, 200)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)

        title = QLabel('Training local icon classifier...')
        title.setFont(QFont('', 11, QFont.Weight.Bold))
        title.setStyleSheet('color:#e8d870;')

        self._status_lbl = QLabel('Starting...')
        self._status_lbl.setStyleSheet('color:#ccc;font-size:10px;')
        self._status_lbl.setWordWrap(True)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)

        self._log = QLabel('')
        self._log.setStyleSheet(
            'color:#888;font-size:9px;font-family:monospace;'
            'background:#111;border:1px solid #333;padding:3px;border-radius:2px;')
        self._log.setWordWrap(True)
        self._log.setFixedHeight(36)

        btn_cancel = QPushButton('Cancel')
        btn_cancel.setFixedWidth(80)
        btn_cancel.clicked.connect(self.cancelled.emit)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)

        lay.addWidget(title)
        lay.addWidget(self._status_lbl)
        lay.addWidget(self._bar)
        lay.addWidget(self._log)
        lay.addLayout(btn_row)

        self._log_lines: list[str] = []

    def update_progress(self, pct: int, message: str):
        self._bar.setValue(pct)
        self._status_lbl.setText(message)
        self._log_lines.append(message)
        if len(self._log_lines) > 3:
            self._log_lines = self._log_lines[-3:]
        self._log.setText('\n'.join(self._log_lines))


class WarpCoreWindow(QMainWindow):
    """
    WARP CORE trainer window.

    Left:   screenshot file list with per-file screen type badge + type override combo.
    Center: AnnotationWidget canvas.
    Right:  Recognition review panel.
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
        self._current_idx  = -1
        self._screen_types: dict[str, str] = {}       # filename -> screen type
        self._screen_types_manual: set[str] = set()   # filenames overridden by user
        self._recognition_cache: dict[str, list] = {} # filename -> recognition items
        self._recognition_items: list[dict] = []
        self._manual_bbox_mode = False
        self._add_bbox_mode    = False
        self._sync_client = None
        self._sync_timer  = None
        self._init_sync_client()
        self._train_worker = None
        self._detect_worker: ScreenTypeDetectorWorker | None = None
        self._recog_worker:  RecognitionWorker | None = None
        self._detect_dlg:    _DetectProgressDialog | None = None
        self._recog_dlg:     _RecognitionProgressDialog | None = None

        self.setWindowTitle('WARP CORE — ML Trainer')
        self.setMinimumSize(1280, 740)
        self._build_ui()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            'Ready — open a folder of STO screenshots to start annotating.')

    # ── UI ────────────────────────────────────────────────────────────────────

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
        sp.setSizes([220, 780, 300])

    def _make_left_panel(self) -> QWidget:
        left = QWidget()
        left.setFixedWidth(220)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)

        lbl = QLabel('Screenshots')
        lbl.setFont(QFont('', 10, QFont.Weight.Bold))
        ll.addWidget(lbl)

        # Screen type override for current screenshot
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel('Type:'))
        self._type_override_combo = QComboBox()
        self._type_override_combo.setToolTip(
            'Override auto-detected screen type for this screenshot')
        for key in SCREEN_TYPE_LABELS:
            icon  = SCREEN_TYPE_ICONS.get(key, '')
            label = SCREEN_TYPE_LABELS[key]
            self._type_override_combo.addItem(f'{icon} {label}', key)
        self._type_override_combo.currentIndexChanged.connect(
            self._on_type_override_changed)
        type_row.addWidget(self._type_override_combo, 1)
        ll.addLayout(type_row)

        self._file_list = QListWidget()
        self._file_list.currentRowChanged.connect(self._load_screenshot)
        ll.addWidget(self._file_list, 1)

        self._prog_lbl = QLabel('0 / 0 annotated')
        self._prog_lbl.setStyleSheet('color:#aaa;font-size:10px;')
        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setTextVisible(False)
        ll.addWidget(self._prog_lbl)
        ll.addWidget(self._prog_bar)
        return left

    def _make_center_panel(self) -> QWidget:
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # AnnotationWidget displays at 1:1 pixel scale.
        # QScrollArea provides panning when the image is larger than the viewport.
        self._ann_widget = AnnotationWidget(self._data_mgr)
        self._ann_widget.annotation_added.connect(self._on_bbox_drawn)
        self._ann_widget.item_selected.connect(self._on_item_selected)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(self._ann_widget)
        self._scroll_area.setWidgetResizable(False)  # keep 1:1 — no auto-resize
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setStyleSheet('QScrollArea { background: #111; border: none; }')

        cl.addWidget(self._scroll_area, 1)
        cl.addWidget(self._make_bottom_panel())
        return center

    def _make_bottom_panel(self) -> QGroupBox:
        g = QGroupBox('Annotate Selected Icon')
        g.setFixedHeight(120)
        lay = QHBoxLayout(g)
        lay.setSpacing(10)

        sc = QVBoxLayout()
        sc.addWidget(QLabel('Slot:'))
        self._slot_combo = QComboBox()
        self._slot_combo.setFixedWidth(200)
        self._slot_combo.setEditable(False)
        for s in ALL_SLOTS:
            self._slot_combo.addItem(s)
        self._slot_combo.currentTextChanged.connect(self._on_slot_changed)
        sc.addWidget(self._slot_combo)
        lay.addLayout(sc)

        nc = QVBoxLayout()
        self._name_label = QLabel('Item name:')
        nc.addWidget(self._name_label)

        # Free-text name entry (icon slots + Ship Name / Ship Type)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Item name (or leave blank for 'Unknown')")
        self._name_edit.returnPressed.connect(self._on_accept)
        self._name_edit.textEdited.connect(self._on_name_edited)
        self._completer_model = QStandardItemModel()
        self._completer = QCompleter(self._completer_model, self._name_edit)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.setMaxVisibleItems(12)
        self._completer.activated.connect(self._name_edit.setText)
        self._name_edit.setCompleter(self._completer)
        nc.addWidget(self._name_edit)

        # Fixed-value combo for Ship Tier
        self._tier_combo = QComboBox()
        for t in SHIP_TIER_VALUES:
            self._tier_combo.addItem(t)
        self._tier_combo.hide()
        nc.addWidget(self._tier_combo)

        # Searchable combo for Ship Type (populated from SETS cache)
        self._ship_type_combo = QComboBox()
        self._ship_type_combo.setEditable(True)
        self._ship_type_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._ship_type_combo.lineEdit().setPlaceholderText('Type to search ship...')
        ship_type_completer = QCompleter(self._ship_type_combo.model(),
                                         self._ship_type_combo)
        ship_type_completer.setCompletionMode(
            QCompleter.CompletionMode.PopupCompletion)
        ship_type_completer.setCaseSensitivity(
            Qt.CaseSensitivity.CaseInsensitive)
        ship_type_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        ship_type_completer.setMaxVisibleItems(14)
        self._ship_type_combo.setCompleter(ship_type_completer)
        self._ship_type_combo.hide()
        nc.addWidget(self._ship_type_combo)

        lay.addLayout(nc, 1)

        bc = QVBoxLayout()
        bc.addStretch()
        br = QHBoxLayout()
        self._btn_accept = QPushButton('Accept')
        self._btn_accept.setStyleSheet(
            'QPushButton{background:#1a5c3a;color:#7effc8;'
            'border:1px solid #3aac6a;border-radius:3px;padding:5px 12px;font-weight:bold;}'
            'QPushButton:hover{background:#2a8c5a;}')
        self._btn_accept.clicked.connect(self._on_accept)
        br.addWidget(self._btn_accept)
        bc.addLayout(br)
        lay.addLayout(bc)
        return g

    def _make_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(380)
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
            'Click item to select on canvas.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#888;font-size:10px;')
        pl.addWidget(hint)

        # Screen type info badge
        self._screen_type_badge = QLabel('Screen type: —')
        self._screen_type_badge.setStyleSheet(
            'color:#e8c060;background:#1a1a00;border:1px solid #4a4000;'
            'border-radius:3px;padding:2px 6px;font-size:11px;')
        pl.addWidget(self._screen_type_badge)

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
        self._btn_edit_bbox = QPushButton('Edit BBox')
        self._btn_edit_bbox.setStyleSheet(
            'QPushButton{background:#1a3a5c;color:#7ec8e3;border:1px solid #3a6aac;'
            'border-radius:3px;padding:4px 8px;}'
            'QPushButton:hover{background:#2a5a8c;}'
            'QPushButton:checked{background:#0a2a4c;border:2px solid #7ec8e3;}')
        self._btn_edit_bbox.setCheckable(True)
        self._btn_edit_bbox.clicked.connect(self._on_edit_bbox_toggle)

        btn_row.addWidget(self._btn_edit_bbox)
        pl.addLayout(btn_row)

        # Add bbox / Remove row
        bbox_mgmt_row = QHBoxLayout()
        self._btn_add_bbox = QPushButton('+ Add BBox')
        self._btn_add_bbox.setToolTip(
            'Draw a new bounding box on the canvas to add a missing item')
        self._btn_add_bbox.setStyleSheet(
            'QPushButton{background:#2a2a1a;color:#e8d870;border:1px solid #8a8030;'
            'border-radius:3px;padding:4px 8px;}'
            'QPushButton:hover{background:#4a4a2a;}'
            'QPushButton:checked{background:#1a1a08;border:2px solid #e8d870;}')
        self._btn_add_bbox.setCheckable(True)
        self._btn_add_bbox.clicked.connect(self._on_add_bbox_toggle)

        self._btn_remove_item = QPushButton('- Remove BBox')
        self._btn_remove_item.setToolTip(
            'Remove selected item from Recognition Review list')
        self._btn_remove_item.setStyleSheet(
            'QPushButton{background:#2a1a2a;color:#e870e8;border:1px solid #803080;'
            'border-radius:3px;padding:4px 8px;}'
            'QPushButton:hover{background:#4a2a4a;}')
        self._btn_remove_item.clicked.connect(self._on_remove_item)

        bbox_mgmt_row.addWidget(self._btn_add_bbox)
        bbox_mgmt_row.addWidget(self._btn_remove_item)
        pl.addLayout(bbox_mgmt_row)

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

        def act(label, tip, slot):
            a = QAction(label, self)
            a.setToolTip(tip)
            a.triggered.connect(slot)
            tb.addAction(a)
            tb.addSeparator()
            return a

        act('Open Folder',  'Open screenshots folder',              self._on_open)
        act('Save',         'Save annotations locally',             self._on_save)
        act('Auto-Detect Slots', 'Auto-detect icons in all screenshots', self._on_auto_detect)
        act('Detect Screen Types', 'Re-classify screen types using trained model',
            self._on_detect_screen_types)
        act('Train Model',  'Train icon + screen-type classifiers on confirmed data',
            self._on_train)
        act('Sync to Hub',  'Upload annotations to Hugging Face Hub', self._on_sync)

    # ── File handling ─────────────────────────────────────────────────────────

    def _on_open(self):
        last = self._settings.value(_KEY_LAST_DIR, '')
        dlg  = QFileDialog(self)
        dlg.setWindowTitle('Open Screenshots Folder')
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
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
        if not self._screenshots:
            self.statusBar().showMessage('No images found in selected folder.')
            return
        self._screen_types.clear()
        self._screen_types_manual.clear()
        self._recognition_cache.clear()
        self._recognition_items = []
        self._current_idx = -1
        self._file_list.clear()
        for p in self._screenshots:
            self._screen_types[p.name] = 'UNKNOWN'
            self._file_list.addItem(self._make_file_list_item(p, 'UNKNOWN'))
        self._start_screen_type_detection()

    def _start_screen_type_detection(self):
        """Launch background OCR worker to detect screen types."""
        total = len(self._screenshots)
        self._detect_dlg = _DetectProgressDialog(total, parent=self)
        self._detect_dlg.cancelled.connect(self._on_detect_cancelled)
        self._detect_dlg.show()
        models_dir = self._sets_root / 'warp' / 'models'
        self._detect_worker = ScreenTypeDetectorWorker(
            self._screenshots, models_dir=models_dir, parent=self)
        self._detect_worker.progress.connect(self._on_detect_progress)
        self._detect_worker.finished.connect(self._on_detect_finished)
        self._detect_worker.start()
        self.statusBar().showMessage(
            f'Detecting screen types for {total} screenshot(s)...')

    def _on_detect_progress(self, idx: int, total: int, filename: str, stype: str):
        """Update badge live as each screenshot is processed.
        Skip filenames that the user has manually overridden.
        """
        if filename in self._screen_types_manual:
            if self._detect_dlg:
                self._detect_dlg.update_progress(
                    idx, total, filename,
                    self._screen_types.get(filename, 'UNKNOWN'))
            return
        self._screen_types[filename] = stype
        for row, p in enumerate(self._screenshots):
            if p.name == filename:
                item = self._file_list.item(row)
                if item:
                    icon  = SCREEN_TYPE_ICONS.get(stype, '?')
                    label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                    item.setText(f'{icon} {label}\n  {filename}')
                    item.setForeground(
                        QColor('#7effc8') if self._data_mgr.has_annotations(p)
                        else Qt.GlobalColor.white)
                # Load first screenshot as soon as its type is known
                if row == 0 and self._current_idx < 0:
                    self._file_list.setCurrentRow(0)
                break
        self.statusBar().showMessage(
            f'Detecting screen types... {idx}/{total}  ({filename})')
        if self._detect_dlg:
            self._detect_dlg.update_progress(idx, total, filename, stype)

    def _on_detect_finished(self, results: dict):
        """Background detection complete — skip manually overridden types."""
        for fname, stype in results.items():
            if fname not in self._screen_types_manual:
                self._screen_types[fname] = stype
        for row, p in enumerate(self._screenshots):
            stype = self._screen_types.get(p.name, 'UNKNOWN')
            item  = self._file_list.item(row)
            if item:
                icon  = SCREEN_TYPE_ICONS.get(stype, '?')
                label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                item.setText(f'{icon} {label}\n  {p.name}')
                item.setForeground(
                    QColor('#7effc8') if self._data_mgr.has_annotations(p)
                    else Qt.GlobalColor.white)
        if self._detect_dlg:
            self._detect_dlg.close()
            self._detect_dlg = None
        self._detect_worker = None
        if self._screenshots:
            if self._current_idx < 0:
                self._file_list.setCurrentRow(0)
            else:
                # Refresh current view — type badge and recognition may now be correct
                self._load_screenshot(self._current_idx)
        self._update_progress()
        self.statusBar().showMessage(
            f'Ready -- {len(self._screenshots)} screenshot(s) loaded.')

    def _on_detect_cancelled(self):
        """User cancelled screen type detection."""
        if self._detect_worker and self._detect_worker.isRunning():
            self._detect_worker.requestInterruption()
            self._detect_worker.wait(3000)
        if self._detect_dlg:
            self._detect_dlg.close()
            self._detect_dlg = None
        if self._screenshots:
            self._file_list.setCurrentRow(0)
        self._update_progress()
        self.statusBar().showMessage(
            'Detection cancelled -- some types may show as Unknown.')

    def _make_file_list_item(self, p: Path, stype: str) -> QListWidgetItem:
        icon  = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        item  = QListWidgetItem(f'{icon} {label}\n  {p.name}')
        item.setForeground(
            QColor('#7effc8') if self._data_mgr.has_annotations(p)
            else Qt.GlobalColor.white)
        return item

    def _load_screenshot(self, row: int):
        """
        Load screenshot into canvas and restore cached recognition.
        Does NOT run OCR -- that only happens once per unique screen type.
        """
        if row < 0 or row >= len(self._screenshots):
            return
        self._current_idx = row
        path  = self._screenshots[row]
        stype = self._screen_types.get(path.name, 'UNKNOWN')
        self._ann_widget.load_image(path)
        self._exit_manual_bbox_mode()
        self._update_screen_type_ui(stype)
        cached = self._recognition_cache.get(path.name)
        if cached is not None:
            self._populate_review_panel(cached, stype)
        else:
            self._recognition_items = []
            self._review_list.clear()
            if stype == 'UNKNOWN':
                self._review_summary.setText('Detecting screen type...')
            else:
                self._review_summary.setText(
                    'Click Auto-Detect to recognise items on this screenshot.')
            self._set_review_buttons_enabled(False)
            self._ann_widget.set_review_items([])
        self._update_add_bbox_btn()
        icon  = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        self.statusBar().showMessage(
            f'{path.name}  ({row+1}/{len(self._screenshots)})  -- {icon} {label}')

    # -- Screen type UI -------------------------------------------------------

    def _update_screen_type_ui(self, stype: str):
        """Update badge, slot combo and type override for current screen."""
        icon  = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        self._screen_type_badge.setText(f'Screen: {icon} {label}')
        self._refresh_slot_combo(stype)
        self._type_override_combo.blockSignals(True)
        for i in range(self._type_override_combo.count()):
            if self._type_override_combo.itemData(i) == stype:
                self._type_override_combo.setCurrentIndex(i)
                break
        self._type_override_combo.blockSignals(False)

    def _refresh_slot_combo(self, stype: str):
        """Repopulate slot combo for the given screen type."""
        group_key    = SCREEN_TO_SLOT_GROUP.get(stype, 'SPACE')
        slots        = SLOT_GROUPS.get(group_key, ALL_SLOTS)
        current_slot = self._slot_combo.currentText()
        self._slot_combo.blockSignals(True)
        self._slot_combo.clear()
        for s in slots:
            self._slot_combo.addItem(s)
        idx = self._slot_combo.findText(current_slot)
        if idx >= 0:
            self._slot_combo.setCurrentIndex(idx)
        self._slot_combo.blockSignals(False)

    def _on_type_override_changed(self, index: int):
        """
        User manually overrides screen type.
        - Updates in-memory screen type
        - Saves 224×224 PNG as training example for the ML classifier
        - Adds to session k-NN immediately
        """
        if self._current_idx < 0:
            return
        stype = self._type_override_combo.itemData(index)
        path  = self._screenshots[self._current_idx]
        self._screen_types[path.name] = stype
        self._screen_types_manual.add(path.name)   # protect from auto re-detection
        self._recognition_cache.pop(path.name, None)
        item = self._file_list.item(self._current_idx)
        if item:
            icon  = SCREEN_TYPE_ICONS.get(stype, '?')
            label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
            item.setText(f'{icon} {label}\n  {path.name}')
        self._update_screen_type_ui(stype)
        self._save_screen_type_example(path, stype)

    # -- Auto-detect ----------------------------------------------------------

    def _on_auto_detect(self):
        """
        Run full recognition on the currently selected screenshot.
        Session examples (user-confirmed items) are included as extra evidence.
        """
        if self._current_idx < 0:
            self.statusBar().showMessage('No screenshot selected.')
            return
        path  = self._screenshots[self._current_idx]
        stype = self._screen_types.get(path.name, 'UNKNOWN')
        self._recognition_cache.pop(path.name, None)
        self._start_recognition(path, stype)
        self.statusBar().showMessage(f'Auto-Detect running on {path.name}...')

    def _on_detect_screen_types(self):
        """Re-run screen type classification on all loaded screenshots.
        Uses the trained ONNX model if available, falls back to OCR.
        Call this after Train Model to apply the new classifier."""
        if not self._screenshots:
            self.statusBar().showMessage('No screenshots loaded.')
            return
        self._start_screen_type_detection()

    # -- Recognition background worker ----------------------------------------

    def _start_recognition(self, path: Path, stype: str):
        """Launch RecognitionWorker with a progress dialog."""
        if self._recog_worker and self._recog_worker.isRunning():
            self._recog_worker.requestInterruption()
            self._recog_worker.wait(2000)
        if self._recog_dlg:
            self._recog_dlg.close()
            self._recog_dlg = None
        self._recognition_items = []
        self._review_list.clear()
        self._review_summary.setText('Running recognition...')
        self._set_review_buttons_enabled(False)
        self._recog_dlg = _RecognitionProgressDialog(
            path.name, stype, parent=self)
        self._recog_dlg.cancelled.connect(self._on_recognition_cancelled)
        self._recog_dlg.show()
        self.statusBar().showMessage(f'Recognising icons in {path.name}...')
        self._recog_worker = RecognitionWorker(
            path, stype, self._sets, parent=self)
        self._recog_worker.finished.connect(
            lambda items: self._on_recognition_done(path.name, stype, items))
        self._recog_worker.error.connect(self._on_recognition_error)
        self._recog_worker.start()

    def _on_recognition_done(self, filename: str, stype: str, items: list):
        """RecognitionWorker finished -- close dialog, cache results."""
        if self._recog_dlg:
            self._recog_dlg.close()
            self._recog_dlg = None
        self._recognition_cache[filename] = items
        if (self._current_idx >= 0 and
                self._screenshots[self._current_idx].name == filename):
            self._populate_review_panel(items, stype)
        self.statusBar().showMessage(
            f'Recognition done -- {len(items)} item(s) found.')

    def _on_recognition_error(self, msg: str):
        if self._recog_dlg:
            self._recog_dlg.close()
            self._recog_dlg = None
        self._review_summary.setText(f'Recognition error: {msg}')
        self.statusBar().showMessage(f'Recognition error: {msg}')

    def _on_recognition_cancelled(self):
        if self._recog_worker and self._recog_worker.isRunning():
            self._recog_worker.requestInterruption()
            self._recog_worker.wait(2000)
        if self._recog_dlg:
            self._recog_dlg.close()
            self._recog_dlg = None
        self._review_summary.setText('Recognition cancelled.')
        self.statusBar().showMessage('Recognition cancelled.')

    def _populate_review_panel(self, items: list, stype: str):
        """Fill right-panel review list.
        Merges recognition results with already-confirmed items from data_mgr
        so previous work is visible and Accept never creates duplicates.
        """
        confirmed_by_id: dict[str, dict] = {}
        if self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            for ann in self._data_mgr.get_annotations(path):
                if ann.state == AnnotationState.CONFIRMED:
                    confirmed_by_id[ann.ann_id] = {
                        'name': ann.name, 'slot': ann.slot,
                        'bbox': ann.bbox, 'state': 'confirmed',
                        'conf': 1.0, 'thumb': None,
                        'crop_bgr': None, 'orig_name': ann.name,
                        'ship_name': '', 'ann_id': ann.ann_id,
                    }
        from warp.trainer.training_data import Annotation as _Ann
        merged: list[dict] = []
        seen_ids: set[str] = set()
        for ri in items:
            bbox = ri.get('bbox')
            if bbox:
                aid = _Ann(bbox=bbox, slot=ri.get('slot',''),
                           name=ri.get('name','')).ann_id
                if aid in confirmed_by_id:
                    ri = dict(confirmed_by_id[aid])
                seen_ids.add(aid)
            merged.append(ri)
        for aid, ci in confirmed_by_id.items():
            if aid not in seen_ids:
                merged.append(ci)
        self._recognition_items = merged
        self._review_list.clear()
        self._review_summary.setText('')
        self._set_review_buttons_enabled(False)
        for ri in self._recognition_items:
            self._add_review_row(ri['name'], ri['slot'], ri.get('conf', 0.0),
                                 confirmed=(ri.get('state') == 'confirmed'))
        # Push to canvas so it draws from review items (single source of truth)
        self._ann_widget.set_review_items(self._recognition_items)
        self._ann_widget.set_selected_row(-1)
        n       = len(items)
        matched = sum(1 for i in items if i.get('name'))
        icon    = SCREEN_TYPE_ICONS.get(stype, '?')
        label   = SCREEN_TYPE_LABELS.get(stype, stype)
        ship    = (items[0].get('ship_name') or '--') if items else '--'
        self._review_summary.setText(
            f'{matched}/{n} identified  Ship: {ship}  {icon} {label}')
        self._set_review_buttons_enabled(n > 0)
        if n > 0:
            self._review_list.setCurrentRow(0)


    def _add_review_row(self, name: str, slot: str, conf: float,
                        confirmed: bool = False):
        if confirmed:
            label = f'{slot}  ->  {name or "—"}  [confirmed]'
        else:
            label = f'{slot}  ->  {name or "— unmatched —"}  [{conf:.0%}]'
        item = QListWidgetItem(label)
        if confirmed:
            item.setForeground(QColor('#7effc8'))
        elif not name:
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
            self._ann_widget.clear_highlight()
            return
        ri = self._recognition_items[row]
        is_confirmed = ri.get('state') == 'confirmed'
        # Always enable Remove; Edit BBox only when confirmed (to unlock editing)
        # or when pending/new (already editable via Edit BBox toggle)
        self._btn_remove_item.setEnabled(True)
        self._btn_edit_bbox.setEnabled(True)
        # Confirmed items: force canvas into view-only mode until Edit BBox clicked
        if is_confirmed:
            self._btn_edit_bbox.setChecked(False)
            self._ann_widget.set_draw_mode(False)
        slot = ri['slot']
        # Try to find slot in current combo; if missing expand to ALL
        idx = self._slot_combo.findText(slot)
        if idx < 0:
            self._slot_combo.blockSignals(True)
            self._slot_combo.clear()
            for s in ALL_SLOTS:
                self._slot_combo.addItem(s)
            self._slot_combo.blockSignals(False)
            idx = self._slot_combo.findText(slot)
        if idx >= 0:
            self._slot_combo.setCurrentIndex(idx)
        self._name_edit.setText(ri['name'])
        if ri.get('bbox'):
            self._ann_widget.set_selected_row(row)

    def _save_screen_type_example(self, path: Path, stype: str) -> None:
        """
        Save a 224x224 PNG as an ML training example and register it
        in the session k-NN so subsequent Auto-Detect uses it immediately.
        """
        try:
            import cv2, hashlib
            img = cv2.imread(str(path))
            if img is None:
                return
            small   = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            out_dir = (self._sets_root / 'warp' / 'training_data'
                       / 'screen_types' / stype)
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix   = hashlib.md5(path.name.encode()).hexdigest()[:8]
            out_path = out_dir / f'{path.stem}_{suffix}.png'
            cv2.imwrite(str(out_path), small)
            log.debug(f'Screen type example saved: {out_path}')
            # Session k-NN — effective immediately
            from warp.recognition.screen_classifier import ScreenTypeClassifier
            ScreenTypeClassifier.add_session_example(img, stype)
            n = len(list(out_dir.glob('*.png')))
            self.statusBar().showMessage(
                f'Screen type set to {stype}  ({n} training example(s) saved)')
        except Exception as e:
            log.warning(f'_save_screen_type_example failed: {e}')

    def _init_sync_client(self):
        """Start sync client immediately and schedule periodic refresh."""
        try:
            from warp.knowledge.sync_client import WARPSyncClient
            from PySide6.QtCore import QTimer
            self._sync_client = WARPSyncClient()
            self._sync_timer  = QTimer(self)
            self._sync_timer.setInterval(15 * 60 * 1000)
            self._sync_timer.timeout.connect(self._on_sync_timer)
            self._sync_timer.start()
            log.debug('WARP CORE: sync client started, refresh every 15 min')
        except Exception as e:
            log.warning(f'WARP CORE: sync client init failed: {e}')

    def _on_sync_timer(self):
        """Periodic background download of community knowledge."""
        if self._sync_client:
            try:
                self._sync_client.refresh_knowledge()
                log.debug('WARP CORE: periodic knowledge refresh triggered')
            except Exception as e:
                log.debug(f'WARP CORE: knowledge refresh error: {e}')

    def _contribute(self, ri: dict, confirmed_name: str):
        try:
            if self._sync_client is None:
                return
            wrong = ri.get('orig_name', '')
            if wrong == confirmed_name:
                wrong = ''
            self._sync_client.contribute(
                crop_bgr   = ri['crop_bgr'],
                item_name  = confirmed_name,
                wrong_name = wrong,
                confirmed  = True,
            )
        except Exception as e:
            log.warning(f'WARP CORE: contribute failed: {e}')

    def _on_edit_bbox_toggle(self, checked: bool):
        if checked:
            self._btn_add_bbox.setChecked(False)
            self._enter_manual_bbox_mode()
        else:
            self._exit_manual_bbox_mode()

    def _on_add_bbox_toggle(self, checked: bool):
        """Start drawing a brand-new bbox (not tied to an existing review item)."""
        if checked:
            self._btn_edit_bbox.setChecked(False)
            self._manual_bbox_mode = False
            self._add_bbox_mode = True
            self._manual_mode_lbl.setText(
                'Draw a rectangle to add a new item to the review list.')
            self._manual_mode_lbl.setVisible(True)
            self._ann_widget.set_draw_mode(True)
            self.statusBar().showMessage(
                'Add BBox mode -- drag a rectangle on the image.')
        else:
            self._add_bbox_mode = False
            self._manual_mode_lbl.setVisible(False)
            self._ann_widget.set_draw_mode(False)

    def _on_remove_item(self):
        """Remove the selected item from the recognition review list."""
        row = self._review_list.currentRow()
        if row < 0 or row >= len(self._recognition_items):
            return
        self._review_list.takeItem(row)
        self._recognition_items.pop(row)
        # Invalidate cache for this screenshot so the removal sticks
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        n = len(self._recognition_items)
        if n == 0:
            self._set_review_buttons_enabled(False)
            self._ann_widget.clear_highlight()
        else:
            self._review_list.setCurrentRow(min(row, n - 1))
        self._ann_widget.set_review_items(self._recognition_items)
        self._update_progress()

    def _enter_manual_bbox_mode(self):
        self._manual_bbox_mode = True
        self._btn_edit_bbox.setChecked(True)
        row  = self._review_list.currentRow()
        if 0 <= row < len(self._recognition_items):
            ri   = self._recognition_items[row]
            slot = ri['slot']
            # Unlock confirmed item back to pending so canvas allows move/resize
            if ri.get('state') == 'confirmed':
                ri['state'] = 'pending'
                self._ann_widget.set_review_items(self._recognition_items)
                self._ann_widget.set_selected_row(row)
        else:
            slot = '?'
        self._manual_mode_lbl.setText(
            f'Draw a rectangle to redefine region for:\n{slot}')
        self._manual_mode_lbl.setVisible(True)
        self._ann_widget.set_draw_mode(True)
        self.statusBar().showMessage(
            'Edit BBox mode — drag a rectangle on the image.')

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
        """Enable/disable per-item buttons. Add BBox is always on when a
        screenshot is loaded -- managed separately by _update_add_bbox_btn."""
        for btn in (self._btn_edit_bbox, self._btn_remove_item):
            btn.setEnabled(enabled)

    def _update_add_bbox_btn(self):
        """+ Add BBox is active whenever a screenshot is displayed."""
        self._btn_add_bbox.setEnabled(self._current_idx >= 0)

    # ── Canvas callbacks ──────────────────────────────────────────────────────

    def _on_bbox_drawn(self, bbox: tuple):
        if self._manual_bbox_mode:
            # Edit existing item's bbox
            row = self._review_list.currentRow()
            if 0 <= row < len(self._recognition_items):
                self._recognition_items[row]['bbox'] = bbox
            self._exit_manual_bbox_mode()
            self._rematch_current_item(row, bbox)
        elif getattr(self, '_add_bbox_mode', False):
            # Add a brand-new item to the review list
            self._add_bbox_mode = False
            self._btn_add_bbox.setChecked(False)
            self._manual_mode_lbl.setVisible(False)
            self._ann_widget.set_draw_mode(False)
            self._ann_widget.set_review_items(self._recognition_items)
            # Rematch the crop to get a name suggestion
            name, conf, thumb = '', 0.0, None
            crop_bgr = None
            if self._current_idx >= 0:
                try:
                    import cv2
                    path = self._screenshots[self._current_idx]
                    img  = cv2.imread(str(path))
                    if img is not None:
                        x, y, w, h = bbox
                        crop_bgr = img[y:y+h, x:x+w].copy()
                        from warp.recognition.icon_matcher import SETSIconMatcher
                        name, conf, thumb = SETSIconMatcher(self._sets).match(crop_bgr)
                except Exception:
                    pass
            slot = self._slot_combo.currentText()
            # Don't run icon matching for free-text slots
            if slot in NON_ICON_SLOTS:
                name, conf, thumb, crop_bgr = '', 0.0, None, None
            new_item = {
                'name':      name,
                'slot':      slot,
                'conf':      conf,
                'bbox':      bbox,
                'state':     'pending',
                'thumb':     thumb,
                'crop_bgr':  crop_bgr,
                'orig_name': name,
                'ship_name': '',
            }
            self._recognition_items.append(new_item)
            self._add_review_row(name, slot, conf)
            new_row = len(self._recognition_items) - 1
            self._review_list.setCurrentRow(new_row)
            self._set_review_buttons_enabled(True)
            # Update cache
            if self._current_idx >= 0:
                fname = self._screenshots[self._current_idx].name
                self._recognition_cache[fname] = list(self._recognition_items)
            self._name_edit.setText(name)
            self._name_edit.setFocus()
            self.statusBar().showMessage(
                f'New item added -- fill in name and click Confirm.')
        else:
            self._name_edit.setFocus()
            self._name_edit.clear()

    def _rematch_current_item(self, row: int, bbox: tuple):
        if row < 0 or self._current_idx < 0:
            return
        # Skip icon matching for free-text slots
        if 0 <= row < len(self._recognition_items):
            if self._recognition_items[row].get('slot', '') in NON_ICON_SLOTS:
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
            matcher          = SETSIconMatcher(self._sets)
            name, conf, thumb = matcher.match(crop)
            ri = self._recognition_items[row]
            ri.update({'name': name, 'conf': conf, 'thumb': thumb})
            self._name_edit.setText(name)
            litem = self._review_list.item(row)
            if litem:
                colour = ('#7effc8' if conf >= CONF_HIGH
                          else '#e8c060' if conf >= CONF_MEDIUM
                          else '#ff7e7e')
                litem.setText(
                    f'{ri["slot"]}  ->  {name or "— unmatched —"}  [{conf:.0%}]')
                litem.setForeground(QColor(colour))
            self.statusBar().showMessage(
                f'Rematched: {name or "no match"} ({conf:.0%})')
        except Exception as e:
            log.exception('Rematch error')
            self.statusBar().showMessage(f'Rematch error: {e}')

    def _on_item_selected(self, ann: dict):
        slot = ann.get('slot', '')
        self._slot_combo.setCurrentText(slot)
        self._on_slot_changed(slot)   # update visible widget
        name = ann.get('name', '')
        if slot == 'Ship Tier':
            idx = self._tier_combo.findText(name)
            if idx >= 0:
                self._tier_combo.setCurrentIndex(idx)
        elif slot == 'Ship Type':
            self._populate_ship_type_combo()
            idx = self._ship_type_combo.findText(name)
            if idx >= 0:
                self._ship_type_combo.setCurrentIndex(idx)
            else:
                self._ship_type_combo.lineEdit().setText(name)
        else:
            self._name_edit.setText(name)

    def _on_accept(self):
        """
        Unified accept: saves annotation from the currently selected
        Recognition Review item, updates data_mgr, contributes to community
        knowledge, and advances to the next unconfirmed item.
        """
        slot = self._slot_combo.currentText()
        if slot == 'Ship Tier':
            name = self._tier_combo.currentText()
        elif slot == 'Ship Type':
            name = self._ship_type_combo.currentText().strip()
        else:
            name = self._name_edit.text().strip()
        row  = self._review_list.currentRow()

        if 0 <= row < len(self._recognition_items):
            # ── Update recognition item ───────────────────────────────────
            ri = self._recognition_items[row]
            ri['name']  = name
            ri['slot']  = slot
            ri['state'] = 'confirmed'
            # ── Persist to data manager ───────────────────────────────────
            if ri.get('bbox') and self._current_idx >= 0:
                path = self._screenshots[self._current_idx]
                self._data_mgr.add_annotation(
                    image_path=path,
                    bbox=ri['bbox'],
                    slot=slot,
                    name=name,
                    state=AnnotationState.CONFIRMED,
                )
            # ── Update list item colour ───────────────────────────────────
            litem = self._review_list.item(row)
            if litem:
                litem.setText(f'{slot}  ->  {name or "—"}  [confirmed]')
                litem.setForeground(QColor('#7effc8'))
            # ── Session example: improves recognition immediately ─────────
            if name and ri.get('crop_bgr') is not None and slot not in NON_ICON_SLOTS:
                from warp.recognition.icon_matcher import SETSIconMatcher
                SETSIconMatcher.add_session_example(ri['crop_bgr'], name)
            # ── Contribute to community (non-blocking) ────────────────────
            if name and ri.get('crop_bgr') is not None and slot not in NON_ICON_SLOTS:
                self._contribute(ri, name)
        else:
            # Fallback: no review item selected, use legacy canvas confirm
            self._ann_widget.confirm_current(slot=slot, name=name)

        self._name_edit.clear()
        self._update_progress()
        self._advance_to_next_unconfirmed(row)
        # Update cache and canvas
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        self._ann_widget.clear_pending()          # remove yellow NEW bbox
        self._ann_widget.set_review_items(self._recognition_items)

    def _build_search_candidates(self, slot: str = '') -> list[str]:
        """
        Collect item names from SETS cache relevant to the given slot.
        If slot starts with 'Boff' → only BOFF abilities.
        Otherwise → equipment + traits + specializations.
        """
        candidates: list[str] = []
        if not self._sets:
            return candidates
        is_boff = slot.startswith('Boff')
        if is_boff:
            try:
                candidates.extend(
                    self._sets.cache.boff_abilities.get('all', {}).keys())
            except Exception:
                pass
        else:
            try:
                for cat_items in self._sets.cache.equipment.values():
                    candidates.extend(cat_items)
            except Exception:
                pass
            try:
                candidates.extend(self._sets.cache.starship_traits)
            except Exception:
                pass
            try:
                candidates.extend(self._sets.cache.traits)
            except Exception:
                pass
            try:
                # Also include boff abilities for mixed/unknown slots
                candidates.extend(
                    self._sets.cache.boff_abilities.get('all', {}).keys())
            except Exception:
                pass
            candidates.extend(SPECIALIZATION_NAMES)
        return sorted(set(candidates))

    def _on_slot_changed(self, slot: str):
        """Show the right input widget depending on slot type."""
        is_tier      = (slot == 'Ship Tier')
        is_ship_type = (slot == 'Ship Type')
        is_text      = (slot in TEXT_SLOTS)
        self._tier_combo.setVisible(is_tier)
        self._ship_type_combo.setVisible(is_ship_type)
        self._name_edit.setVisible(not is_tier and not is_ship_type)
        if is_tier:
            self._name_label.setText('Tier:')
        elif is_ship_type:
            self._name_label.setText('Ship Type:')
            self._populate_ship_type_combo()
        elif is_text:
            self._name_label.setText('Value:')
        else:
            self._name_label.setText('Item name:')

    def _populate_ship_type_combo(self):
        """Lazily fill _ship_type_combo from SETS cache (once per session)."""
        if self._ship_type_combo.count() > 0:
            return   # already populated
        names: list[str] = []
        if self._sets:
            try:
                names = sorted(self._sets.cache.ships.keys())
            except Exception:
                pass
        if not names:
            self._ship_type_combo.lineEdit().setPlaceholderText(
                'SETS cache not loaded — type manually')
            return
        for n in names:
            self._ship_type_combo.addItem(n)
        # Clear the text so placeholder shows
        self._ship_type_combo.setCurrentIndex(-1)
        self._ship_type_combo.lineEdit().clear()

    def _on_name_edited(self, text: str):
        """
        Called on every keystroke in Item name field.
        For TEXT_SLOTS (Ship Name/Type/Tier) -- no autocomplete.
        For icon slots -- filters SETS cache and updates the QCompleter popup.
        """
        slot = self._slot_combo.currentText()
        if slot in NON_ICON_SLOTS:
            self._completer_model.clear()
            return
        query = text.strip().lower()
        if len(query) < 2:
            self._completer_model.clear()
            return
        all_names = self._build_search_candidates(slot)
        matches   = [n for n in all_names if query in n.lower()][:60]
        self._completer_model.clear()
        for name in matches:
            self._completer_model.appendRow(QStandardItem(name))
        if matches:
            self._completer.complete()

    # ── Save / Sync ───────────────────────────────────────────────────────────

    def _on_save(self):
        self._data_mgr.save()
        self._update_progress()
        self.statusBar().showMessage('Annotations saved.')

    def _on_train(self):
        """
        Train both models in sequence:
          1. Screen-type classifier  (MobileNetV3) — 0-45%
          2. Icon classifier         (EfficientNet) — 45-100%
        Uses all confirmed data: screen-type corrections + icon annotations.
        """
        if self._train_worker and self._train_worker.isRunning():
            self.statusBar().showMessage('Training already running...')
            return

        self._train_dlg = _TrainProgressDialog(parent=self)
        self._train_dlg.cancelled.connect(self._on_train_cancelled)
        self._train_dlg.show()

        from PySide6.QtCore import QThread, Signal as _Signal

        data_root  = self._sets_root / 'warp' / 'training_data'
        models_dir = self._sets_root / 'warp' / 'models'
        data_mgr   = self._data_mgr
        sets_root  = self._sets_root

        class _CombinedTrainWorker(QThread):
            progress = _Signal(int, str)
            finished = _Signal(bool, str)

            def __init__(self_, parent=None):
                super().__init__(parent)

            def run(self_):
                interrupted = self_.isInterruptionRequested

                # ── Phase 1: Screen-type classifier (0-45%) ────────────────
                sc_ok  = True
                sc_msg = ''
                sc_dir = data_root / 'screen_types'
                has_sc_data = sc_dir.exists() and any(
                    len(list(d.glob('*.png'))) > 0
                    for d in sc_dir.iterdir() if d.is_dir()
                ) if sc_dir.exists() else False

                if has_sc_data:
                    def sc_prog(pct, msg):
                        self_.progress.emit(int(pct * 0.45), f'[Screen types] {msg}')
                    def sc_done(ok, msg):
                        nonlocal sc_ok, sc_msg
                        sc_ok, sc_msg = ok, msg

                    from warp.trainer.screen_type_trainer import ScreenTypeTrainerWorker
                    w = ScreenTypeTrainerWorker(data_root, models_dir)
                    w.run(sc_prog, sc_done, interrupted)
                    if interrupted():
                        self_.finished.emit(False, 'Cancelled')
                        return
                else:
                    self_.progress.emit(5, '[Screen types] No data yet — skipping')

                # ── Phase 2: Icon classifier (45-100%) ────────────────────
                self_.progress.emit(45, '[Icons] Starting icon classifier training...')

                ic_ok  = True
                ic_msg = ''

                try:
                    from warp.trainer.local_trainer import LocalTrainWorker as _LTW
                    from PySide6.QtCore import QThread as _QT, Signal as _S

                    class _IconThread(_QT):
                        progress = _S(int, str)
                        finished = _S(bool, str)
                        def __init__(self2, parent=None):
                            super().__init__(parent)
                            self2._worker = _LTW.__new__(_LTW)
                            self2._worker._data_mgr  = data_mgr
                            self2._worker._sets_root = sets_root
                            # Wire worker signals to thread signals
                            self2._worker.progress = self2.progress
                            self2._worker.finished = self2.finished
                        def run(self2):
                            self2._worker._train()

                    # Run synchronously in current thread via direct call
                    icon_worker = _LTW.__new__(_LTW)
                    icon_worker._data_mgr  = data_mgr
                    icon_worker._sets_root = sets_root

                    # Capture progress/finished via monkey-patch
                    class _FakeSignal:
                        def __init__(self2, cb): self2._cb = cb
                        def emit(self2, *a):     self2._cb(*a)

                    icon_worker.progress = _FakeSignal(
                        lambda pct, msg: self_.progress.emit(
                            45 + int(pct * 0.55), f'[Icons] {msg}'))
                    icon_worker.finished = _FakeSignal(
                        lambda ok, msg: (
                            ic_ok.__class__,  # noop, captured below
                        ))

                    results = {}
                    icon_worker.progress = _FakeSignal(
                        lambda pct, msg: self_.progress.emit(
                            45 + int(pct * 0.55), f'[Icons] {msg}'))
                    icon_worker.finished = _FakeSignal(
                        lambda ok, msg: results.update({'ok': ok, 'msg': msg}))

                    icon_worker._train()
                    ic_ok  = results.get('ok',  False)
                    ic_msg = results.get('msg', 'No result')
                except Exception as e:
                    ic_ok  = False
                    ic_msg = str(e)

                if interrupted():
                    self_.finished.emit(False, 'Cancelled')
                    return

                # ── Summary ───────────────────────────────────────────────
                parts = []
                if has_sc_data:
                    parts.append(f'Screen types: {"OK" if sc_ok else "FAILED"} — {sc_msg}')
                parts.append(f'Icons: {"OK" if ic_ok else "FAILED"} — {ic_msg}')
                overall_ok = ic_ok and (sc_ok if has_sc_data else True)
                self_.finished.emit(overall_ok, '\n\n'.join(parts))

        self._train_worker = _CombinedTrainWorker(parent=self)
        self._train_worker.progress.connect(self._train_dlg.update_progress)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.start()
        self.statusBar().showMessage('Training started (screen types + icons)...')

    def _on_train_cancelled(self):
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.requestInterruption()
            self._train_worker.wait(3000)
        if hasattr(self, '_train_dlg') and self._train_dlg:
            self._train_dlg.close()
            self._train_dlg = None
        self.statusBar().showMessage('Training cancelled.')

    def _on_train_finished(self, success: bool, message: str):
        if hasattr(self, '_train_dlg') and self._train_dlg:
            self._train_dlg.close()
            self._train_dlg = None
        from PySide6.QtWidgets import QMessageBox
        if success:
            QMessageBox.information(self, 'Training Complete', message)
            # Clear caches — next Auto-Detect and folder-open use new models
            self._recognition_cache.clear()
            try:
                from warp.recognition.icon_matcher import SETSIconMatcher
                SETSIconMatcher.reset_ml_session()
            except Exception:
                pass
            try:
                from warp.recognition.screen_classifier import ScreenTypeClassifier
                ScreenTypeClassifier.clear_session()
            except Exception:
                pass
            # Re-run screen type detection with the newly trained model
            if self._screenshots:
                self._start_screen_type_detection()
        else:
            QMessageBox.warning(self, 'Training Failed', message)
        self._train_worker = None

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
                stype = self._screen_types.get(p.name, 'UNKNOWN')
                icon  = SCREEN_TYPE_ICONS.get(stype, '?')
                label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                item.setText(f'{icon} {label}\n  {p.name}')
                item.setForeground(
                    QColor('#7effc8')
                    if self._data_mgr.has_annotations(p)
                    else Qt.GlobalColor.white)

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(8):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')
