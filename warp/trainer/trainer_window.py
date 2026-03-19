# warp/trainer/trainer_window.py
# WARP CORE — Interactive ML trainer + recognition review.
# PySide6, integrated with SETS.

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QComboBox, QLineEdit, QGroupBox,
    QProgressBar, QToolBar, QStatusBar, QMessageBox,
    QInputDialog, QSizePolicy, QFrame, QScrollArea,
    QAbstractItemView, QCompleter, QMenu, QPlainTextEdit
)
from PySide6.QtCore import Qt, QSettings, QThread, Signal, QSortFilterProxyModel, QSize
from PySide6.QtGui import QFont, QAction, QColor, QStandardItemModel, QStandardItem

from warp.trainer.annotation_widget import AnnotationWidget
from warp.trainer.training_data      import TrainingDataManager, AnnotationState
from warp.trainer.sync               import SyncWorker, HFTokenDialog

log = logging.getLogger(__name__)

_KEY_LAST_DIR      = 'warp_core/last_dir'
_KEY_HF_TOKEN      = 'warp_core/hf_token'
_KEY_TRAIN_REPEATS = 'warp_core/train_repeats'
_KEY_TRAIN_DLG_SIZE = 'warp_core/train_dlg_size'

CONF_HIGH   = 0.85
CONF_MEDIUM = 0.70

SLOT_GROUPS: dict[str, list[str]] = {
    # SPACE_EQ: space equipment + ship metadata (name/type/tier live on space screenshots)
    'SPACE_EQ': [
        'Fore Weapons', 'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield',
        'Aft Weapons', 'Experimental', 'Devices', 'Universal Consoles',
        'Engineering Consoles', 'Science Consoles', 'Tactical Consoles', 'Hangars',
        'Ship Name', 'Ship Type', 'Ship Tier',
    ],
    # GROUND_EQ: ground equipment only — no ship metadata
    'GROUND_EQ': [
        'Body Armor', 'EV Suit', 'Personal Shield', 'Weapons', 'Kit', 'Kit Modules', 'Ground Devices',
    ],
    'TRAITS': [
        'Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep',
        'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep',
    ],
    'BOFFS': [
        'Boff Tactical', 'Boff Engineering', 'Boff Science', 'Boff Operations',
        'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
    ],
    'SPECIALIZATIONS': [
        'Primary Specialization', 'Secondary Specialization',
    ],
    # SPACE_MIXED: merged space screenshot — equipment + traits + boffs + specs, no ground gear
    'SPACE_MIXED': [
        'Fore Weapons', 'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield',
        'Aft Weapons', 'Experimental', 'Devices', 'Universal Consoles',
        'Engineering Consoles', 'Science Consoles', 'Tactical Consoles', 'Hangars',
        'Ship Name', 'Ship Type', 'Ship Tier',
        'Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep',
        'Boff Tactical', 'Boff Engineering', 'Boff Science', 'Boff Operations',
        'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
        'Primary Specialization', 'Secondary Specialization',
    ],
    # GROUND_MIXED: merged ground screenshot — ground gear + traits + boffs + specs, no space gear
    'GROUND_MIXED': [
        'Body Armor', 'EV Suit', 'Personal Shield', 'Weapons', 'Kit', 'Kit Modules', 'Ground Devices',
        'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep',
        'Boff Tactical', 'Boff Engineering', 'Boff Science', 'Boff Operations',
        'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
        'Primary Specialization', 'Secondary Specialization',
    ],
}

SCREEN_TYPE_LABELS: dict[str, str] = {
    'SPACE_EQ': 'Space Equipment', 'GROUND_EQ': 'Ground Equipment', 'TRAITS': 'Traits',
    'BOFFS': 'Bridge Officers', 'SPECIALIZATIONS': 'Specializations',
    'SPACE_MIXED': 'Space Mixed (merged)', 'GROUND_MIXED': 'Ground Mixed (merged)', 'UNKNOWN': 'Unknown',
}

SCREEN_TYPE_ICONS: dict[str, str] = {
    'SPACE_EQ': '🚀', 'GROUND_EQ': '🦶', 'TRAITS': '✨', 'BOFFS': '👥',
    'SPECIALIZATIONS': '🎯', 'SPACE_MIXED': '🌌', 'GROUND_MIXED': '🗺️', 'UNKNOWN': '❓',
}

SCREEN_TO_SLOT_GROUP: dict[str, str] = {
    'SPACE_EQ':       'SPACE_EQ',
    'GROUND_EQ':      'GROUND_EQ',
    'TRAITS':         'TRAITS',
    'BOFFS':          'BOFFS',
    'SPECIALIZATIONS':'SPECIALIZATIONS',
    'SPACE_MIXED':    'SPACE_MIXED',
    'GROUND_MIXED':   'GROUND_MIXED',
    'UNKNOWN':        'ALL',   # unknown type → show everything, let user decide
}

TEXT_SLOTS: frozenset[str] = frozenset(['Ship Name'])
FIXED_VALUE_SLOTS: frozenset[str] = frozenset(['Ship Tier', 'Ship Type'])
NON_ICON_SLOTS: frozenset[str] = TEXT_SLOTS | FIXED_VALUE_SLOTS
SHIP_TIER_VALUES: list[str] = ['T1', 'T2', 'T3', 'T4', 'T5', 'T5-U', 'T5-X', 'T5-X2', 'T6', 'T6-X', 'T6-X2']
_SHIP_INFO_SLOTS = ['Ship Name', 'Ship Type', 'Ship Tier']

# Build ALL_SLOTS as a flat deduplicated list of every slot across all groups
ALL_SLOTS: list[str] = []
for _slots in SLOT_GROUPS.values():
    for _s in _slots:
        if _s not in ALL_SLOTS:
            ALL_SLOTS.append(_s)
for _s in _SHIP_INFO_SLOTS:
    if _s not in ALL_SLOTS:
        ALL_SLOTS.append(_s)
SLOT_GROUPS['ALL'] = ALL_SLOTS

SPECIALIZATION_NAMES: list[str] = ['Command Officer', 'Intelligence Officer', 'Miracle Worker', 'Pilot', 'Temporal Operative', 'Constable', 'Commando', 'Strategist']

class ScreenTypeDetectorWorker(QThread):
    progress = Signal(int, int, str, str)
    finished = Signal(dict)
    def __init__(self, paths: list, models_dir=None, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._models_dir = models_dir
    def run(self):
        results: dict[str, str] = {}
        total = len(self._paths)
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
    finished = Signal(list)
    error    = Signal(str)
    def __init__(self, path, stype: str, sets_app, parent=None):
        super().__init__(parent)
        self._path = path
        self._stype = stype
        self._sets_app = sets_app
    def run(self):
        importer_type = {'SPACE_EQ': 'SPACE', 'GROUND_EQ': 'GROUND', 'TRAITS': 'SPACE_TRAITS', 'BOFFS': 'BOFFS', 'SPECIALIZATIONS': 'SPEC', 'SPACE_MIXED': 'SPACE', 'GROUND_MIXED': 'GROUND'}.get(self._stype, 'SPACE_EQ')
        try:
            import cv2
            from warp.warp_importer import WarpImporter
            importer = WarpImporter(sets_app=self._sets_app, build_type=importer_type)
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
        except:
            img2 = None
        for ri in result.items:
            crop_bgr = None
            if ri.bbox and img2 is not None:
                try:
                    x, y, w, h = ri.bbox
                    crop_bgr = img2[y:y+h, x:x+w].copy()
                except:
                    pass
            items.append({'name': ri.name, 'slot': ri.slot, 'conf': ri.confidence, 'bbox': ri.bbox, 'state': 'pending', 'thumb': ri.thumbnail, 'crop_bgr': crop_bgr, 'orig_name': ri.name, 'ship_name': result.ship_name})
        self.finished.emit(items)

class _DetectProgressDialog(QWidget):
    cancelled = Signal()
    def __init__(self, total: int, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('WARP CORE — Detecting Screen Types')
        self.setFixedSize(460, 140)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        self._title_lbl = QLabel('Classifying screenshots with ML model…')
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
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        self._file_lbl.setText(f'{filename}  →  {icon} {label}')

class _RecognitionProgressDialog(QWidget):
    cancelled = Signal()
    def __init__(self, filename: str, stype: str, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('WARP CORE -- Recognising Icons')
        self.setFixedSize(420, 130)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, stype)
        title = QLabel('Matching icons against SETS library...')
        title.setFont(QFont('', 10, QFont.Weight.Bold))
        title.setStyleSheet('color:#7ec8e3;')
        file_lbl = QLabel(f'{icon} {label}   {filename}')
        file_lbl.setStyleSheet('color:#aaa;font-size:10px;')
        file_lbl.setWordWrap(True)
        bar = QProgressBar()
        bar.setRange(0, 0)
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
    cancelled = Signal()
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('WARP CORE -- Training Icon Classifier')
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._settings = QSettings()
        size = self._settings.value(_KEY_TRAIN_DLG_SIZE, QSize(520, 450))
        self.resize(size)
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
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet('color:#888;font-size:10px;font-family:monospace;background:#111;border:1px solid #333;padding:3px;border-radius:2px;')
        btn_cancel = QPushButton('Cancel')
        btn_cancel.setFixedWidth(80)
        btn_cancel.clicked.connect(self.cancelled.emit)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        lay.addWidget(title)
        lay.addWidget(self._status_lbl)
        lay.addWidget(self._bar)
        lay.addWidget(self._log, 1)
        lay.addLayout(btn_row)
    def update_progress(self, pct: int, message: str):
        self._bar.setValue(pct)
        self._status_lbl.setText(message)
        self._log.appendPlainText(message)
        self._log.ensureCursorVisible()
    def hideEvent(self, event):
        self._settings.setValue(_KEY_TRAIN_DLG_SIZE, self.size())
        super().hideEvent(event)

class WarpCoreWindow(QMainWindow):
    BOFF_ABILITY_PROPERTIES: dict[str, tuple[str, str]] = {
        "Beams: Fire at Will": ("Tactical", "Space"), "Beams: Overload": ("Tactical", "Space"),
        "Tactical Team": ("Tactical", "Space"), "Torpedoes: High Yield": ("Tactical", "Space"),
        "Torpedoes: Spread": ("Tactical", "Space"), "Target Weapons Subsystems": ("Tactical", "Space"),
        "Target Engines Subsystems": ("Tactical", "Space"), "Target Shields Subsystems": ("Tactical", "Space"),
        "Target Auxiliary Subsystems": ("Tactical", "Space"), "Attack Pattern Beta": ("Tactical", "Space"),
        "Attack Pattern Delta": ("Tactical", "Space"), "Cannons: Rapid Fire": ("Tactical", "Space"),
        "Cannons: Scatter Volley": ("Tactical", "Space"), "Dispersal Pattern Alpha": ("Tactical", "Space"),
        "Dispersal Pattern Beta": ("Tactical", "Space"), "Focused Assault": ("Tactical", "Space"),
        "Attack Pattern Omega": ("Tactical", "Space"), "Photon Grenade": ("Tactical", "Ground"),
        "Sweeping Strikes": ("Tactical", "Ground"), "Battle Strategies": ("Tactical", "Ground"),
        "Draw Fire": ("Tactical", "Ground"), "Graviton Spike": ("Tactical", "Ground"),
        "Corrosive Grenade": ("Tactical", "Ground"), "Cease Fire": ("Tactical", "Ground"),
        "Lunge": ("Tactical", "Ground"), "Suppressing Fire": ("Tactical", "Ground"),
        "Target Optics": ("Tactical", "Ground"), "Smoke Grenade": ("Tactical", "Ground"),
        "Micro Cryonic Warhead": ("Tactical", "Ground"), "Gre'thor's Chains": ("Tactical", "Ground"),
        "Emergency Power to Auxiliary": ("Engineering", "Space"), "Emergency Power to Weapons": ("Engineering", "Space"),
        "Emergency Power to Engines": ("Engineering", "Space"), "Emergency Power to Shields": ("Engineering", "Space"),
        "Engineering Team": ("Engineering", "Space"), "Reverse Shield Polarity": ("Engineering", "Space"),
        "Boarding Party": ("Engineering", "Space"), "Auxiliary Power to the Emergency Battery": ("Engineering", "Space"),
        "Auxiliary Power to the Inertial Dampers": ("Engineering", "Space"), "Auxiliary Power to the Structural Integrity Field": ("Engineering", "Space"),
        "Eject Warp Plasma": ("Engineering", "Space"), "Aceton Beam": ("Engineering", "Space"),
        "Chroniton Mine Barrier": ("Engineering", "Ground"), "Quick Fix": ("Engineering", "Ground"),
        "Shield Recharge": ("Engineering", "Ground"), "Weapons Malfunction": ("Engineering", "Ground"),
        "Hurricane Turret": ("Engineering", "Ground"), "Molten Terrain": ("Engineering", "Ground"),
        "Photon Grenade Launcher Fabrication": ("Engineering", "Ground"), "Explosive Drone Fabrication": ("Engineering", "Ground"),
        "Cover Shield": ("Engineering", "Ground"), "Equipment Diagnostics": ("Engineering", "Ground"),
        "Medical Generator Fabrication": ("Engineering", "Ground"), "Quantum Mortar Fabrication": ("Engineering", "Ground"),
        "Hazard Emitters": ("Science", "Space"), "Science Team": ("Science", "Space"),
        "Tachyon Beam": ("Science", "Space"), "Gravity Well": ("Science", "Space"),
        "Photonic Officer": ("Science", "Space"), "Tyken's Rift": ("Science", "Space"),
        "Feedback Pulse": ("Science", "Space"), "Scramble Sensors": ("Science", "Space"),
        "Photonic Shockwave": ("Science", "Space"), "Viral Matrix": ("Science", "Space"),
        "Medical Tricorder": ("Science", "Ground"), "Stasis Field": ("Science", "Ground"),
        "Tricorder Scan": ("Science", "Ground"), "Vascular Regenerator": ("Science", "Ground"),
        "Exothermic Redistribution": ("Science", "Ground"), "Seismic Agitation Field": ("Science", "Ground"),
        "Sonic Disruption": ("Science", "Ground"), "Nanite Health Monitor": ("Science", "Ground"),
    }

    def __init__(self, sets_app=None, parent=None):
        super().__init__(parent)
        self._sets = sets_app
        self._settings = QSettings()
        self._sets_root = self._find_sets_root()
        self._data_mgr = TrainingDataManager(self._sets_root / 'warp' / 'training_data')
        self._screenshots: list[Path] = []
        self._current_idx = -1
        self._screen_types: dict[str, str] = {}
        self._screen_types_manual: set[str] = set()
        self._recognition_cache: dict[str, list] = {}
        self._recognition_items: list[dict] = []
        self._manual_bbox_mode = False
        self._add_bbox_mode = False
        self._sync_client = None
        self._sync_timer = None
        self._init_sync_client()
        self._train_worker = None
        self._detect_worker = None
        self._suppress_next_focus_popup = False  # set True after programmatic setFocus
        self._recog_worker = None
        self._detect_dlg = None
        self._recog_dlg = None
        self._sync_worker = None
        self._train_dlg = None
        self._selection_just_changed = False
        self.setWindowTitle('WARP CORE — ML Trainer')
        self.setMinimumSize(1280, 740)
        self._build_ui()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Ready — open a folder of STO screenshots to start annotating.')

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
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)
        lbl = QLabel('Screenshots')
        lbl.setFont(QFont('', 10, QFont.Weight.Bold))
        ll.addWidget(lbl)
        self._file_list = QListWidget()
        self._file_list.currentRowChanged.connect(self._load_screenshot)
        self._file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._file_list.customContextMenuRequested.connect(self._show_file_list_context_menu)
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
        self._ann_widget = AnnotationWidget(self._data_mgr)
        self._ann_widget.annotation_added.connect(self._on_bbox_drawn)
        self._ann_widget.item_selected.connect(self._on_item_selected)
        self._ann_widget.item_deselected.connect(self._on_canvas_deselected)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(self._ann_widget)
        self._scroll_area.setWidgetResizable(False)
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
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Item name (or leave blank for 'Unknown')")
        self._name_edit.returnPressed.connect(self._on_accept)
        self._name_edit.textEdited.connect(self._on_name_edited)
        self._name_edit.focusInEvent  = self._on_name_focus_in
        self._name_edit.mousePressEvent = self._on_name_mouse_press
        self._completer_model = QStandardItemModel()
        self._completer = QCompleter(self._completer_model, self._name_edit)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.setMaxVisibleItems(12)
        self._completer.activated.connect(self._on_completer_activated)
        self._name_edit.setCompleter(self._completer)
        nc.addWidget(self._name_edit)
        self._tier_combo = QComboBox()
        for t in SHIP_TIER_VALUES:
            self._tier_combo.addItem(t)
        self._tier_combo.hide()
        nc.addWidget(self._tier_combo)
        self._ship_type_combo = QComboBox()
        self._ship_type_combo.setEditable(True)
        self._ship_type_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._ship_type_combo.lineEdit().setPlaceholderText('Type to search ship...')
        stc = QCompleter(self._ship_type_combo.model(), self._ship_type_combo)
        stc.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        stc.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        stc.setFilterMode(Qt.MatchFlag.MatchContains)
        stc.setMaxVisibleItems(14)
        self._ship_type_combo.setCompleter(stc)
        self._ship_type_combo.hide()
        nc.addWidget(self._ship_type_combo)
        lay.addLayout(nc, 1)
        bc = QVBoxLayout()
        bc.addStretch()
        br = QHBoxLayout()
        self._btn_accept = QPushButton('Accept')
        self._btn_accept.setStyleSheet('QPushButton{background:#1a5c3a;color:#7effc8;border:1px solid #3aac6a;border-radius:3px;padding:5px 12px;font-weight:bold;}QPushButton:hover{background:#2a8c5a;}')
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
        hint = QLabel('Green = high confidence\nYellow = uncertain  ·  Red = unmatched\nClick item to select on canvas.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#888;font-size:10px;')
        pl.addWidget(hint)
        self._screen_type_badge = QLabel('Screen type: —')
        self._screen_type_badge.setStyleSheet('color:#e8c060;background:#1a1a00;border:1px solid #4a4000;border-radius:3px;padding:2px 6px;font-size:11px;')
        pl.addWidget(self._screen_type_badge)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color:#333;')
        pl.addWidget(sep)
        self._review_list = QListWidget()
        self._review_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._review_list.currentRowChanged.connect(self._on_review_row_changed)
        self._review_list.itemClicked.connect(self._on_review_item_clicked)
        pl.addWidget(self._review_list, 1)
        self._review_summary = QLabel('')
        self._review_summary.setStyleSheet('color:#aaa;font-size:10px;')
        self._review_summary.setWordWrap(True)
        pl.addWidget(self._review_summary)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet('color:#333;')
        pl.addWidget(sep2)
        self._btn_edit_bbox = QPushButton('Edit BBox')
        self._btn_edit_bbox.setStyleSheet('QPushButton{background:#1a3a5c;color:#7ec8e3;border:1px solid #3a6aac;border-radius:3px;padding:4px 8px;}QPushButton:hover{background:#2a5a8c;}QPushButton:checked{background:#0a2a4c;border:2px solid #7ec8e3;}')
        self._btn_edit_bbox.setCheckable(True)
        self._btn_edit_bbox.clicked.connect(self._on_edit_bbox_toggle)
        pl.addWidget(self._btn_edit_bbox)
        mgmt = QHBoxLayout()
        self._btn_add_bbox = QPushButton('+ Add BBox')
        self._btn_add_bbox.setStyleSheet('QPushButton{background:#2a2a1a;color:#e8d870;border:1px solid #8a8030;border-radius:3px;padding:4px 8px;}QPushButton:hover{background:#4a4a2a;}QPushButton:checked{background:#1a1a08;border:2px solid #e8d870;}')
        self._btn_add_bbox.setCheckable(True)
        self._btn_add_bbox.clicked.connect(self._on_add_bbox_toggle)
        self._btn_remove_item = QPushButton('- Remove BBox')
        self._btn_remove_item.setStyleSheet('QPushButton{background:#2a1a2a;color:#e870e8;border:1px solid #803080;border-radius:3px;padding:4px 8px;}QPushButton:hover{background:#4a2a4a;}')
        self._btn_remove_item.clicked.connect(self._on_remove_item)
        mgmt.addWidget(self._btn_add_bbox)
        mgmt.addWidget(self._btn_remove_item)
        pl.addLayout(mgmt)
        self._manual_mode_lbl = QLabel('')
        self._manual_mode_lbl.setStyleSheet('color:#e8c060;font-size:10px;background:#2a2a00;border:1px solid #6a6000;border-radius:3px;padding:3px;')
        self._manual_mode_lbl.setWordWrap(True)
        self._manual_mode_lbl.setVisible(False)
        pl.addWidget(self._manual_mode_lbl)
        self._set_review_buttons_enabled(False)
        return panel

    def _build_toolbar(self):
        tb = QToolBar('Main')
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        def act(l, t, s):
            a = QAction(l, self)
            a.setToolTip(t)
            a.triggered.connect(s)
            tb.addAction(a)
            tb.addSeparator()
            return a
        act('Open Folder', 'Open screenshots folder', self._on_open)
        act('Detect Screen Types', 'Re-classify screen types', self._on_detect_screen_types)
        act('Auto-Detect Slots', 'Auto-detect icons', self._on_auto_detect)
        act('Train Model', 'Train icon + screen-type classifiers', self._on_train)

    def _on_open(self):
        last = self._settings.value(_KEY_LAST_DIR, '')
        dlg = QFileDialog(self)
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
        self._screenshots = sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])
        if not self._screenshots:
            self.statusBar().showMessage('No images found.')
            return
        self._screen_types.clear()
        self._screen_types_manual.clear()
        self._recognition_cache.clear()
        self._recognition_items = []
        self._current_idx = -1
        self._file_list.clear()
        # Restore persisted manual screen type labels from TrainingDataManager
        persisted = self._data_mgr.get_all_screen_types()
        for p in self._screenshots:
            saved = persisted.get(p.name, '')
            self._screen_types[p.name] = saved if saved else 'UNKNOWN'
            if saved:
                self._screen_types_manual.add(p.name)
            self._file_list.addItem(self._make_file_list_item(p, self._screen_types[p.name]))
        self._start_screen_type_detection("open_folder")

    def _start_screen_type_detection(self, trigger: str = 'unknown'):
        total = len(self._screenshots)
        self._detect_dlg = _DetectProgressDialog(total, parent=self)
        self._detect_dlg.cancelled.connect(self._on_detect_cancelled)
        self._detect_dlg.show()
        models_dir = self._sets_root / 'warp' / 'models'
        self._detect_worker = ScreenTypeDetectorWorker(self._screenshots, models_dir=models_dir, parent=self)
        self._detect_worker.progress.connect(self._on_detect_progress)
        self._detect_worker.finished.connect(self._on_detect_finished)
        self._detect_worker.start()
        self.statusBar().showMessage(f'Detecting screen types for {total} screenshot(s)...')

    def _on_detect_progress(self, idx: int, total: int, filename: str, stype: str):
        if filename in self._screen_types_manual:
            if self._detect_dlg:
                self._detect_dlg.update_progress(idx, total, filename, self._screen_types.get(filename, 'UNKNOWN'))
            return
        self._screen_types[filename] = stype
        for row, p in enumerate(self._screenshots):
            if p.name == filename:
                item = self._file_list.item(row)
                if item:
                    icon  = SCREEN_TYPE_ICONS.get(stype, '?')
                    label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                    item.setText(f'{icon} {label}\n  {filename}')
                    if self._data_mgr.has_annotations(p):
                        item.setForeground(QColor('#7effc8'))
                    else:
                        item.setForeground(Qt.GlobalColor.white)
                if row == 0 and self._current_idx < 0:
                    self._file_list.setCurrentRow(0)
                break
        if self._detect_dlg:
            self._detect_dlg.update_progress(idx, total, filename, stype)

    def _on_detect_finished(self, results: dict):
        for fname, stype in results.items():
            if fname not in self._screen_types_manual:
                self._screen_types[fname] = stype
        for row, p in enumerate(self._screenshots):
            stype = self._screen_types.get(p.name, 'UNKNOWN')
            item = self._file_list.item(row)
            if item:
                icon  = SCREEN_TYPE_ICONS.get(stype, '?')
                label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                item.setText(f'{icon} {label}\n  {p.name}')
                if self._data_mgr.has_annotations(p):
                    item.setForeground(QColor('#7effc8'))
                else:
                    item.setForeground(Qt.GlobalColor.white)
        if self._detect_dlg:
            self._detect_dlg.close()
            self._detect_dlg = None
        self._detect_worker = None
        if self._screenshots:
            if self._current_idx < 0:
                self._file_list.setCurrentRow(0)
            else:
                self._load_screenshot(self._current_idx)
        self._update_progress()

    def _on_detect_cancelled(self):
        if self._detect_worker and self._detect_worker.isRunning():
            self._detect_worker.requestInterruption()
            self._detect_worker.wait(3000)
        if self._detect_dlg:
            self._detect_dlg.close()
            self._detect_dlg = None
        if self._screenshots:
            self._file_list.setCurrentRow(0)
        self._update_progress()

    def _make_file_list_item(self, p: Path, stype: str) -> QListWidgetItem:
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        item = QListWidgetItem(f'{icon} {label}\n  {p.name}')
        if self._data_mgr.has_annotations(p):
            item.setForeground(QColor('#7effc8'))
        else:
            item.setForeground(Qt.GlobalColor.white)
        return item

    def _load_screenshot(self, row: int):
        if row < 0 or row >= len(self._screenshots): return
        self._current_idx = row; path = self._screenshots[row]; stype = self._screen_types.get(path.name, 'UNKNOWN')
        self._ann_widget.load_image(path); self._exit_manual_bbox_mode(); self._update_screen_type_ui(stype)
        cached = self._recognition_cache.get(path.name)
        if cached is not None: self._populate_review_panel(cached, stype)
        else:
            self._populate_review_panel([], stype)
            if not self._recognition_items:
                if stype == 'UNKNOWN': self._review_summary.setText('Detecting screen type...')
                else: self._review_summary.setText('Click Auto-Detect to recognise items on this screenshot.')
        self._update_add_bbox_btn()

    def _show_file_list_context_menu(self, pos):
        item = self._file_list.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { font-size: 11px; }")
        row = self._file_list.row(item)
        if row < 0 or row >= len(self._screenshots): return
        path = self._screenshots[row]
        current_stype = self._screen_types.get(path.name, 'UNKNOWN')
        for key in SCREEN_TYPE_LABELS:
            icon = SCREEN_TYPE_ICONS.get(key, '')
            label = SCREEN_TYPE_LABELS[key]
            action = menu.addAction(f'{icon} {label}')
            action.setData(key)
            action.setCheckable(True)
            if key == current_stype:
                action.setChecked(True)
        action = menu.exec(self._file_list.mapToGlobal(pos))
        if action:
            self._on_type_override_changed(action.data())

    def _update_screen_type_ui(self, stype: str):
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        self._screen_type_badge.setText(f'Screen: {icon} {label}')
        self._refresh_slot_combo(stype)

    def _refresh_slot_combo(self, stype: str):
        group_key = SCREEN_TO_SLOT_GROUP.get(stype, 'SPACE_EQ')
        slots = SLOT_GROUPS.get(group_key, ALL_SLOTS)
        current_slot = self._slot_combo.currentText()
        self._slot_combo.blockSignals(True)
        self._slot_combo.clear()
        for s in slots:
            self._slot_combo.addItem(s)
        idx = self._slot_combo.findText(current_slot)
        if idx >= 0:
            self._slot_combo.setCurrentIndex(idx)
        self._slot_combo.blockSignals(False)

    def _on_type_override_changed(self, stype: str):
        if self._current_idx < 0: return
        path = self._screenshots[self._current_idx]
        self._screen_types[path.name] = stype
        self._screen_types_manual.add(path.name)
        self._recognition_cache.pop(path.name, None)
        item = self._file_list.item(self._current_idx)
        if item:
            icon = SCREEN_TYPE_ICONS.get(stype, '?')
            label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
            item.setText(f'{icon} {label}\n  {path.name}')
        self._update_screen_type_ui(stype)
        self._save_screen_type_example(path, stype)

    def _save_screen_type_example(self, path: Path, stype: str):
        """Zapisuje ręczne nadpisanie typu ekranu dla danego screenshotu."""
        try:
            # Informujemy menedżera danych o nowym typie ekranu
            self._data_mgr.set_screen_type(path, stype)

            # Aktualizujemy lokalny cache i oznaczamy jako manualnie zmienione
            self._screen_types[path.name] = stype
            self._screen_types_manual.add(path.name)

            # Odświeżamy UI (listę plików i plakietkę typu ekranu)
            self._update_progress()
            self._update_screen_type_ui(stype)

            log.info(f"Manual screen type override: {path.name} -> {stype}")
        except Exception as e:
            log.error(f"Failed to save screen type example: {e}")
            QMessageBox.critical(self, "Error", f"Could not save screen type: {e}")

    def _on_auto_detect(self):
        if self._current_idx < 0: return
        path = self._screenshots[self._current_idx]
        if self._data_mgr.has_annotations(path):
            reply = QMessageBox.question(self, 'Overwrite confirmed annotations?', f'"{path.name}" already has confirmed annotations.\n\nAuto-Detect Slots will overwrite them.\n\nContinue?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes: return
        stype = self._screen_types.get(path.name, 'UNKNOWN')
        self._recognition_cache.pop(path.name, None)
        self._start_recognition(path, stype)

    def _on_detect_screen_types(self):
        if not self._screenshots: return
        # Only detect files not yet manually labelled — never overwrite user choices
        self._start_screen_type_detection('detect_screen_types_button')

    def _start_recognition(self, path: Path, stype: str):
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
        self._recog_dlg = _RecognitionProgressDialog(path.name, stype, parent=self)
        self._recog_dlg.show()
        self._recog_worker = RecognitionWorker(path, stype, self._sets, parent=self)
        self._recog_worker.finished.connect(lambda items: self._on_recognition_done(path.name, stype, items))
        self._recog_worker.error.connect(self._on_recognition_error)
        self._recog_worker.start()

    def _on_recognition_done(self, filename: str, stype: str, items: list):
        if self._recog_dlg:
            self._recog_dlg.close()
            self._recog_dlg = None
        self._recognition_cache[filename] = items
        if self._current_idx >= 0 and self._screenshots[self._current_idx].name == filename:
            self._populate_review_panel(items, stype)

    def _on_recognition_error(self, msg: str):
        if self._recog_dlg:
            self._recog_dlg.close()
            self._recog_dlg = None
        self._review_summary.setText(f'Recognition error: {msg}')

    def _on_recognition_cancelled(self):
        if self._recog_worker and self._recog_worker.isRunning():
            self._recog_worker.requestInterruption()
            self._recog_worker.wait(2000)
        if self._recog_dlg:
            self._recog_dlg.close()
            self._recog_dlg = None
        self._review_summary.setText('Recognition cancelled.')

    def _populate_review_panel(self, items: list, stype: str):
        confirmed_by_id: dict[str, dict] = {}
        if self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            for ann in self._data_mgr.get_annotations(path):
                if ann.state == AnnotationState.CONFIRMED:
                    confirmed_by_id[ann.ann_id] = {'name': ann.name, 'slot': ann.slot, 'bbox': ann.bbox, 'state': 'confirmed', 'conf': 1.0, 'thumb': None, 'crop_bgr': None, 'orig_name': ann.name, 'ship_name': '', 'ann_id': ann.ann_id}
        from warp.trainer.training_data import Annotation as _Ann
        merged: list[dict] = []
        seen_ids: set[str] = set()
        for ri in items:
            bbox = ri.get('bbox')
            if bbox:
                aid = _Ann(bbox=bbox, slot=ri.get('slot',''), name=ri.get('name','')).ann_id
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
            self._add_review_row(ri['name'], ri['slot'], ri.get('conf', 0.0), confirmed=(ri.get('state') == 'confirmed'))
        self._ann_widget.set_review_items(self._recognition_items)
        self._ann_widget.set_selected_row(-1)
        n = len(self._recognition_items)
        matched = sum(1 for i in self._recognition_items if i.get('name'))
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, stype)
        ship = (self._recognition_items[0].get('ship_name') or '--') if self._recognition_items else '--'
        self._review_summary.setText(f'{matched}/{n} identified  Ship: {ship}  {icon} {label}')
        self._set_review_buttons_enabled(n > 0)
        if n > 0:
            self._review_list.setCurrentRow(0)

    def _add_review_row(self, name: str, slot: str, conf: float, confirmed: bool = False):
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

    def _on_review_item_clicked(self, item: QListWidgetItem):
        if not self._selection_just_changed:
            self._review_list.setCurrentRow(-1)
            self._ann_widget.clear_highlight()
        self._selection_just_changed = False

    def _on_review_row_changed(self, row: int):
        if row == -1:
            self._set_review_buttons_enabled(False)
            self._ann_widget.clear_highlight()
            return
        self._selection_just_changed = True
        if 0 <= row < len(self._recognition_items):
            ri = self._recognition_items[row]
            is_confirmed = ri.get('state') == 'confirmed'
            self._btn_remove_item.setEnabled(True)
            self._btn_edit_bbox.setEnabled(True)
            if is_confirmed:
                self._btn_edit_bbox.setChecked(False)
                self._ann_widget.set_draw_mode(False)
            slot = ri['slot']
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
            # Populate completer for this slot without triggering clear on name_edit
            self._populate_name_completer(slot)
            # Set name field directly (slot already set above, skip _on_slot_changed clear)
            self._name_edit.blockSignals(True)
            self._name_edit.setText(ri['name'])
            self._name_edit.blockSignals(False)
            if ri.get('bbox'):
                self._ann_widget.set_highlighted_row(row)

    def _init_sync_client(self):
        try:
            from warp.knowledge.sync_client import WARPSyncClient
            from PySide6.QtCore import QTimer
            self._sync_client = WARPSyncClient()
            self._sync_timer = QTimer(self)
            self._sync_timer.setInterval(15 * 60 * 1000)
            self._sync_timer.timeout.connect(self._on_sync_timer)
            self._sync_timer.start()
        except Exception as e:
            log.warning(f'WARP CORE: sync client init failed: {e}')

    def _on_sync_timer(self):
        if self._sync_client:
            try:
                self._sync_client.refresh_knowledge()
            except Exception as e:
                log.debug(f'WARP CORE: knowledge refresh error: {e}')

    def _contribute(self, ri: dict, confirmed_name: str):
        try:
            if self._sync_client is None:
                return
            wrong = ri.get('orig_name', '')
            if wrong == confirmed_name:
                wrong = ''
            self._sync_client.contribute(crop_bgr=ri['crop_bgr'], item_name=confirmed_name, wrong_name=wrong, confirmed=True)
        except Exception as e:
            log.warning(f'WARP CORE: contribute failed: {e}')

    def _on_edit_bbox_toggle(self, checked: bool):
        if checked:
            self._btn_add_bbox.setChecked(False)
            self._enter_manual_bbox_mode()
        else:
            self._exit_manual_bbox_mode()

    def _on_add_bbox_toggle(self, checked: bool):
        if checked:
            self._btn_edit_bbox.setChecked(False)
            self._manual_bbox_mode = False
            self._add_bbox_mode = True
            self._manual_mode_lbl.setText('Draw a rectangle to add a new item.')
            self._manual_mode_lbl.setVisible(True)
            self._ann_widget.set_draw_mode(True)
        else:
            self._add_bbox_mode = False
            self._manual_mode_lbl.setVisible(False)
            self._ann_widget.set_draw_mode(False)

    def _on_remove_item(self):
        row = self._review_list.currentRow()
        if row < 0 or row >= len(self._recognition_items):
            return
        self._review_list.takeItem(row)
        self._recognition_items.pop(row)
        self._exit_manual_bbox_mode()
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        n = len(self._recognition_items)
        if n == 0:
            self._set_review_buttons_enabled(False)
            self._ann_widget.clear_highlight()
        else:
            new_row = min(row, n - 1)
            self._review_list.setCurrentRow(new_row)
            self._on_review_row_changed(new_row)
        self._ann_widget.set_review_items(self._recognition_items)
        self._update_progress()

    def _enter_manual_bbox_mode(self):
        self._manual_bbox_mode = True
        self._btn_edit_bbox.setChecked(True)
        row = self._review_list.currentRow()
        if 0 <= row < len(self._recognition_items):
            ri = self._recognition_items[row]
            slot = ri['slot']
            if ri.get('state') == 'confirmed':
                ri['state'] = 'pending'
            self._ann_widget.set_review_items(self._recognition_items)
            self._ann_widget.set_selected_row(row)
        else:
            slot = '?'
        self._manual_mode_lbl.setText(f'Draw a rectangle to redefine region for:\n{slot}')
        self._manual_mode_lbl.setVisible(True)
        self._ann_widget.set_draw_mode(True)

    def _exit_manual_bbox_mode(self):
        self._manual_bbox_mode = False
        self._btn_edit_bbox.setChecked(False)
        self._manual_mode_lbl.setVisible(False)
        self._ann_widget.set_draw_mode(False)
        self._ann_widget.set_selected_row(-1)

    def _advance_to_next_unconfirmed(self, current_row: int):
        for i in range(current_row + 1, len(self._recognition_items)):
            if self._recognition_items[i]['state'] == 'pending':
                self._review_list.setCurrentRow(i)
                return

    def _set_review_buttons_enabled(self, enabled: bool):
        for btn in (self._btn_edit_bbox, self._btn_remove_item):
            btn.setEnabled(enabled)

    def _update_add_bbox_btn(self):
        self._btn_add_bbox.setEnabled(self._current_idx >= 0)

    def _on_bbox_drawn(self, bbox: tuple):
        if self._manual_bbox_mode:
            row = self._review_list.currentRow()
            if 0 <= row < len(self._recognition_items):
                self._recognition_items[row]['bbox'] = bbox
            self._rematch_current_item(row, bbox)
        elif getattr(self, '_add_bbox_mode', False):
            self._add_bbox_mode = False
            self._btn_add_bbox.setChecked(False)
            self._manual_mode_lbl.setVisible(False)
            self._ann_widget.set_draw_mode(False)
            self._ann_widget.set_review_items(self._recognition_items)
            name, conf, thumb, crop_bgr = '', 0.0, None, None
            if self._current_idx >= 0:
                try:
                    import cv2
                    path = self._screenshots[self._current_idx]
                    img = cv2.imread(str(path))
                    if img is not None:
                        x, y, w, h = bbox
                        crop_bgr = img[y:y+h, x:x+w].copy()
                        from src.setsdebug import log as _slog
                        _slog.info(f'add_bbox: crop {x},{y},{w},{h} px from {path.name}')
                        from warp.recognition.icon_matcher import SETSIconMatcher
                        # Build candidate set from current slot's allowed items
                        _current_slot = self._slot_combo.currentText()
                        _candidates = set(self._build_search_candidates(_current_slot)) or None
                        name, conf, thumb = SETSIconMatcher(self._sets).match(
                            crop_bgr, candidate_names=_candidates)
                        _slog.info(f'add_bbox: matcher → name={name!r} conf={conf:.2f} '
                                   f'(slot={_current_slot!r}, pool={len(_candidates) if _candidates else "all"})')
                except Exception as _e:
                    from src.setsdebug import log as _slog
                    _slog.warning(f'add_bbox: matcher error: {_e}')
            slot = self._slot_combo.currentText()
            if slot in NON_ICON_SLOTS:
                name, conf, thumb, crop_bgr = '', 0.0, None, None
            # If matcher found a name, infer the correct slot from cache item type
            # Restrict to slots allowed by the current screen type
            if name:
                stype = 'UNKNOWN'
                if self._current_idx >= 0:
                    stype = self._screen_types.get(
                        self._screenshots[self._current_idx].name, 'UNKNOWN')
                group_key = SCREEN_TO_SLOT_GROUP.get(stype, 'ALL')
                allowed = SLOT_GROUPS.get(group_key)  # None means no restriction
                inferred = self._infer_slot_from_name(name, allowed_slots=allowed)
                if inferred:
                    slot = inferred
            new_item = {'name': name, 'slot': slot, 'conf': conf, 'bbox': bbox, 'state': 'pending', 'thumb': thumb, 'crop_bgr': crop_bgr, 'orig_name': name, 'ship_name': ''}
            self._recognition_items.append(new_item)
            self._add_review_row(name, slot, conf)
            new_row = len(self._recognition_items) - 1
            self._review_list.setCurrentRow(new_row)
            self._set_review_buttons_enabled(True)
            if self._current_idx >= 0:
                fname = self._screenshots[self._current_idx].name
                self._recognition_cache[fname] = list(self._recognition_items)
            self._ann_widget.clear_pending()
            # Update slot combo to match inferred slot (suppressing textEdited on name field)
            if slot != self._slot_combo.currentText():
                self._slot_combo.blockSignals(True)
                self._slot_combo.setCurrentText(slot)
                self._slot_combo.blockSignals(False)
                self._populate_name_completer(slot)
            # Fill recognised name but do NOT open the dropdown automatically.
            # User can click the field to browse all slot-compatible items.
            self._name_edit.blockSignals(True)
            self._name_edit.setText(name)
            self._name_edit.blockSignals(False)
            self._suppress_next_focus_popup = True
            self._name_edit.setFocus()
        else:
            self._name_edit.setFocus()
            self._name_edit.clear()

    def _rematch_current_item(self, row: int, bbox: tuple):
        if row < 0 or self._current_idx < 0:
            return
        if 0 <= row < len(self._recognition_items):
            if self._recognition_items[row].get('slot', '') in NON_ICON_SLOTS:
                return
        try:
            import cv2
            from warp.recognition.icon_matcher import SETSIconMatcher
            path = self._screenshots[self._current_idx]
            img = cv2.imread(str(path))
            if img is None:
                return
            x, y, w, h = bbox
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                return
            name, conf, thumb = SETSIconMatcher(self._sets).match(crop)
            ri = self._recognition_items[row]
            ri.update({'name': name, 'conf': conf, 'thumb': thumb})
            self._name_edit.setText(name)
            litem = self._review_list.item(row)
            if litem:
                colour = ('#7effc8' if conf >= CONF_HIGH else '#e8c060' if conf >= CONF_MEDIUM else '#ff7e7e')
                litem.setText(f'{ri["slot"]}  ->  {name or "— unmatched —"}  [{conf:.0%}]')
                litem.setForeground(QColor(colour))
        except:
            pass

    def _on_canvas_deselected(self):
        """Canvas click on already-selected bbox or empty area → deselect everything."""
        self._review_list.blockSignals(True)
        self._review_list.setCurrentRow(-1)
        self._review_list.blockSignals(False)
        self._set_review_buttons_enabled(False)
        self._ann_widget.clear_highlight()

    def _on_item_selected(self, ann: dict):
        """Canvas bbox clicked → sync review list selection + fill slot/name fields."""
        slot = ann.get('slot', '')
        name = ann.get('name', '')

        # Sync review list selection to match canvas click
        bbox = ann.get('bbox')
        if bbox is not None:
            for row, ri in enumerate(self._recognition_items):
                if ri.get('bbox') == bbox:
                    self._review_list.blockSignals(True)
                    self._review_list.setCurrentRow(row)
                    self._review_list.blockSignals(False)
                    self._ann_widget.set_highlighted_row(row)
                    self._set_review_buttons_enabled(True)
                    break

        # Set slot without triggering _on_slot_changed's clear() on name_edit
        self._slot_combo.blockSignals(True)
        self._slot_combo.setCurrentText(slot)
        self._slot_combo.blockSignals(False)
        self._populate_name_completer(slot)

        # Set name fields
        if slot == 'Ship Tier':
            self._tier_combo.setVisible(True)
            self._name_edit.setVisible(False)
            self._name_label.setText('Tier:')
            idx = self._tier_combo.findText(name)
            if idx >= 0:
                self._tier_combo.setCurrentIndex(idx)
        elif slot == 'Ship Type':
            self._ship_type_combo.setVisible(True)
            self._name_edit.setVisible(False)
            self._name_label.setText('Ship Type:')
            self._populate_ship_type_combo()
            idx = self._ship_type_combo.findText(name)
            if idx >= 0:
                self._ship_type_combo.setCurrentIndex(idx)
            else:
                self._ship_type_combo.lineEdit().setText(name)
        else:
            self._tier_combo.setVisible(False)
            self._ship_type_combo.setVisible(False)
            self._name_edit.setVisible(True)
            self._name_label.setText('Item name:')
            self._name_edit.blockSignals(True)
            self._name_edit.setText(name)
            self._name_edit.blockSignals(False)

    def _on_accept(self):
        slot = self._slot_combo.currentText()
        if slot == 'Ship Tier':
            name = self._tier_combo.currentText()
        elif slot == 'Ship Type':
            name = self._ship_type_combo.currentText().strip()
        else:
            name = self._name_edit.text().strip()
        row = self._review_list.currentRow()
        if 0 <= row < len(self._recognition_items):
            ri = self._recognition_items[row]
            ri['name'] = name
            ri['slot'] = slot
            ri['state'] = 'confirmed'
            if ri.get('bbox') and self._current_idx >= 0:
                path = self._screenshots[self._current_idx]
                self._data_mgr.add_annotation(image_path=path, bbox=ri['bbox'], slot=slot, name=name, state=AnnotationState.CONFIRMED)
            litem = self._review_list.item(row)
            if litem:
                litem.setText(f'{slot}  ->  {name or "—"}  [confirmed]')
                litem.setForeground(QColor('#7effc8'))
            if name and ri.get('crop_bgr') is not None and slot not in NON_ICON_SLOTS:
                from warp.recognition.icon_matcher import SETSIconMatcher
                SETSIconMatcher.add_session_example(ri['crop_bgr'], name)
                self._contribute(ri, name)
        else:
            self._ann_widget.confirm_current(slot=slot, name=name)
        self._name_edit.clear()
        self._update_progress()
        self._advance_to_next_unconfirmed(row)
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        self._ann_widget.clear_pending()
        self._ann_widget.set_review_items(self._recognition_items)
        self._data_mgr.save()
        self._auto_sync()

    # cache.equipment item['type'] → trainer slot name
    # Mirrors EQUIPMENT_TYPES in src/constants.py + SLOT_TO_CACHE_KEY above.
    _ITEM_TYPE_TO_SLOT: dict[str, str] = {
        'Ship Fore Weapon':         'Fore Weapons',
        'Ship Aft Weapon':          'Aft Weapons',
        'Ship Weapon':              'Fore Weapons',   # generic weapon → fore by default
        'Experimental Weapon':      'Experimental',
        'Ship Deflector Dish':      'Deflector',
        'Ship Secondary Deflector': 'Sec-Def',
        'Impulse Engine':           'Engines',
        'Warp Engine':              'Warp Core',
        'Singularity Engine':       'Warp Core',
        'Ship Shields':             'Shield',
        'Ship Device':              'Devices',
        'Universal Console':        'Universal Consoles',
        'Ship Engineering Console': 'Engineering Consoles',
        'Ship Science Console':     'Science Consoles',
        'Ship Tactical Console':    'Tactical Consoles',
        'Hangar Bay':               'Hangars',
        'Body Armor':               'Body Armor',
        'EV Suit':                  'EV Suit',
        'Personal Shield':          'Personal Shield',
        'Ground Weapon':            'Weapons',
        'Kit':                      'Kit',
        'Kit Module':               'Kit Modules',
        'Ground Device':            'Ground Devices',
    }

    def _infer_slot_from_name(self, item_name: str, allowed_slots: list[str] | None = None) -> str:
        """
        Given a recognised item name, returns the most appropriate slot name
        by looking up the item's type in cache.

        allowed_slots: if provided, only slots in this list are considered.
        This enforces screen type restrictions — e.g. on a TRAITS screenshot
        only trait slots are valid, never Hangars or equipment slots.

        Returns '' if the item is not found or inferred slot is not allowed.
        """
        if not self._sets or not item_name:
            return ''

        def _allowed(slot: str) -> str:
            """Return slot if allowed, else empty string."""
            if allowed_slots is None or slot in allowed_slots:
                return slot
            return ''

        # Build reverse map: cache_key → slot name (canonical, non-cross-populated)
        canonical_cache_keys = {v: k for k, v in self._SLOT_TO_CACHE_KEY.items()}

        try:
            # First pass: look in canonical (non-cross-populated) buckets only
            for cache_key, slot_name in canonical_cache_keys.items():
                bucket = self._sets.cache.equipment.get(cache_key, {})
                entry = bucket.get(item_name)
                if entry:
                    item_type = entry.get('type', '')
                    # Use item type for most precise slot (handles Universal Console)
                    if item_type in self._ITEM_TYPE_TO_SLOT:
                        return _allowed(self._ITEM_TYPE_TO_SLOT[item_type])
                    return _allowed(slot_name)

            # Second pass: traits — check all buckets with correct cache structure
            # cache.traits[environment][trait_type][name]
            # environment: 'space' | 'ground'
            # trait_type:  'traits' | 'rep_traits' | 'active_rep_traits'
            if hasattr(self._sets.cache, 'starship_traits') and item_name in self._sets.cache.starship_traits:
                return _allowed('Starship Traits')
            if hasattr(self._sets.cache, 'traits'):
                t = self._sets.cache.traits
                trait_slot_map = [
                    ('space',  'rep_traits',        'Space Reputation'),
                    ('space',  'active_rep_traits', 'Active Space Rep'),
                    ('ground', 'rep_traits',        'Ground Reputation'),
                    ('ground', 'active_rep_traits', 'Active Ground Rep'),
                    ('space',  'traits',            'Personal Space Traits'),
                    ('ground', 'traits',            'Personal Ground Traits'),
                ]
                for env, ttype, slot_name in trait_slot_map:
                    try:
                        if item_name in t[env][ttype]:
                            return _allowed(slot_name)
                    except (KeyError, TypeError):
                        pass

            # Third pass: boff abilities — build reverse map from cache structure
            # cache.boff_abilities[env][career][rank_idx] = {ability_name: desc}
            if hasattr(self._sets.cache, 'boff_abilities'):
                boff_cache = self._sets.cache.boff_abilities
                for env in ('space', 'ground'):
                    env_data = boff_cache.get(env, {})
                    if not isinstance(env_data, dict):
                        continue
                    for career, rank_list in env_data.items():
                        if not isinstance(rank_list, list):
                            continue
                        for rank_dict in rank_list:
                            if isinstance(rank_dict, dict) and item_name in rank_dict:
                                return _allowed(f'Boff {career}')
                # Fallback: static BOFF_ABILITY_PROPERTIES
                props = self.BOFF_ABILITY_PROPERTIES.get(item_name)
                if props:
                    career, _ = props
                    return _allowed(f'Boff {career}')
        except Exception:
            pass

        return ''

    # Mapping: trainer slot name → cache.equipment key
    # Must stay in sync with EQUIPMENT_TYPES in src/constants.py and SLOT_GROUPS above.
    _SLOT_TO_CACHE_KEY: dict[str, str] = {
        'Fore Weapons':          'fore_weapons',
        'Aft Weapons':           'aft_weapons',
        'Experimental':          'experimental',
        'Deflector':             'deflector',
        'Sec-Def':               'sec_def',
        'Engines':               'engines',
        'Warp Core':             'core',
        'Shield':                'shield',
        'Devices':               'devices',
        'Universal Consoles':    'uni_consoles',
        'Engineering Consoles':  'eng_consoles',
        'Science Consoles':      'sci_consoles',
        'Tactical Consoles':     'tac_consoles',
        'Hangars':               'hangars',
        'Body Armor':            'armor',
        'EV Suit':               'ev_suit',
        'Personal Shield':       'personal_shield',
        'Weapons':               'weapons',
        'Kit':                   'kit',
        'Kit Modules':           'kit_modules',
        'Ground Devices':        'ground_devices',
    }

    def _build_search_candidates(self, slot: str = '') -> list[str]:
        candidates: list[str] = []
        if not self._sets:
            return candidates

        stype = 'UNKNOWN'
        if self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            stype = self._screen_types.get(path.name, 'UNKNOWN')
        target_domain = 'Ground' if 'GROUND' in stype else 'Space'

        if slot.startswith('Boff'):
            target_career = slot.replace('Boff ', '').strip()
            # Primary source: cache.boff_abilities[environment][career] — keyed by rank dicts
            # Structure: {environment: {career: [{ability: desc}, ...rank levels]}}
            try:
                domain_key = 'ground' if target_domain == 'Ground' else 'space'
                career_ranks = self._sets.cache.boff_abilities.get(domain_key, {}).get(target_career, [])
                for rank_dict in career_ranks:
                    if isinstance(rank_dict, dict):
                        candidates.extend(rank_dict.keys())
            except Exception:
                pass
            # Fallback: static BOFF_ABILITY_PROPERTIES (covers Tactical/Engineering/Science)
            if not candidates:
                for ability, (career, domain) in self.BOFF_ABILITY_PROPERTIES.items():
                    if career == target_career and domain == target_domain:
                        candidates.append(ability)
            # Last resort: all abilities
            if not candidates:
                try:
                    candidates.extend(self._sets.cache.boff_abilities.get('all', {}).keys())
                except Exception:
                    pass
        elif slot in ('Primary Specialization', 'Secondary Specialization'):
            candidates.extend(SPECIALIZATION_NAMES)
        elif 'Starship Trait' in slot:
            # cache.starship_traits = {name: {...}} flat dict
            try:
                candidates.extend(self._sets.cache.starship_traits.keys())
            except Exception:
                pass
        elif slot == 'Active Space Rep':
            try:
                candidates.extend(self._sets.cache.traits['space']['active_rep_traits'].keys())
            except Exception:
                pass
        elif slot == 'Space Reputation':
            try:
                candidates.extend(self._sets.cache.traits['space']['rep_traits'].keys())
            except Exception:
                pass
        elif slot == 'Active Ground Rep':
            try:
                candidates.extend(self._sets.cache.traits['ground']['active_rep_traits'].keys())
            except Exception:
                pass
        elif slot == 'Ground Reputation':
            try:
                candidates.extend(self._sets.cache.traits['ground']['rep_traits'].keys())
            except Exception:
                pass
        elif slot == 'Personal Space Traits':
            try:
                candidates.extend(self._sets.cache.traits['space']['traits'].keys())
            except Exception:
                pass
        elif slot == 'Personal Ground Traits':
            try:
                candidates.extend(self._sets.cache.traits['ground']['traits'].keys())
            except Exception:
                pass
        else:
            cache_key = self._SLOT_TO_CACHE_KEY.get(slot)
            try:
                if cache_key:
                    candidates.extend(self._sets.cache.equipment.get(cache_key, {}).keys())
                else:
                    for cat_items in self._sets.cache.equipment.values():
                        candidates.extend(cat_items.keys())
            except Exception:
                pass

        return sorted(set(candidates))

    def _populate_name_completer(self, slot: str):
        """Pre-populate the completer model for the given slot (called on slot change)."""
        all_names = self._build_search_candidates(slot)
        self._completer_model.clear()
        for name in all_names:
            self._completer_model.appendRow(QStandardItem(name))

    def _on_slot_changed(self, slot: str):
        is_tier = (slot == 'Ship Tier')
        is_ship_type = (slot == 'Ship Type')
        is_text = (slot in TEXT_SLOTS)
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
        # Clear item name field and reset completer state whenever slot changes
        self._suppress_next_focus_popup = True
        self._name_edit.blockSignals(True)
        self._name_edit.clear()
        self._name_edit.blockSignals(False)
        self._suppress_next_focus_popup = False
        # CRITICAL: reset QCompleter's internal completionPrefix so it doesn't
        # filter the new slot's list using the old slot's search text
        self._completer.setCompletionPrefix('')
        # Pre-populate completer with new slot's candidates
        if not is_tier and not is_ship_type and slot not in NON_ICON_SLOTS:
            self._populate_name_completer(slot)

    def _populate_ship_type_combo(self):
        if self._ship_type_combo.count() > 0:
            return
        names: list[str] = []
        if self._sets:
            try:
                names = sorted(self._sets.cache.ships.keys())
            except:
                pass
        if not names:
            self._ship_type_combo.lineEdit().setPlaceholderText('Cache not loaded')
            return
        for n in names:
            self._ship_type_combo.addItem(n)
        self._ship_type_combo.setCurrentIndex(-1)
        self._ship_type_combo.lineEdit().clear()

    def _on_name_focus_in(self, event):
        """On first focus (field was not focused before): open dropdown unless suppressed.
        Qt fires focusInEvent THEN mousePressEvent on the same click.
        We open here only when focus came from keyboard (tab) or programmatic setFocus.
        Mouse click is handled entirely by _on_name_mouse_press to avoid double-firing.
        """
        QLineEdit.focusInEvent(self._name_edit, event)
        if self._suppress_next_focus_popup:
            self._suppress_next_focus_popup = False
            return
        from PySide6.QtCore import Qt as _Qt
        if event.reason() == _Qt.FocusReason.MouseFocusReason:
            # Will be handled by _on_name_mouse_press — skip here to avoid double-open
            return
        self._show_name_dropdown()

    def _on_name_mouse_press(self, event):
        """Open/close dropdown on every mouse click in the field."""
        from PySide6.QtWidgets import QLineEdit as _QLE
        _QLE.mousePressEvent(self._name_edit, event)
        if self._suppress_next_focus_popup:
            self._suppress_next_focus_popup = False
            return
        self._show_name_dropdown()

    def _show_name_dropdown(self):
        """Toggle completer popup for the current slot."""
        slot = self._slot_combo.currentText()
        if slot in NON_ICON_SLOTS:
            return
        popup = self._completer.popup()
        if popup and popup.isVisible():
            popup.hide()
            return
        if self._completer_model.rowCount() == 0:
            self._populate_name_completer(slot)
        if self._completer_model.rowCount():
            self._completer.complete()

    def _on_name_edited(self, text: str):
        slot = self._slot_combo.currentText()
        if slot in NON_ICON_SLOTS:
            self._completer_model.clear()
            return
        query = text.strip().lower()
        all_names = self._build_search_candidates(slot)
        if not query:
            # Empty field — show full slot list (already in model from _populate_name_completer)
            # Just trigger the popup if model has items
            if self._completer_model.rowCount():
                self._completer.complete()
            return
        matches = [n for n in all_names if query in n.lower()][:60]
        self._completer_model.clear()
        for name in matches:
            self._completer_model.appendRow(QStandardItem(name))
        if matches:
            self._completer.complete()

    def _on_completer_activated(self, text: str):
        self._name_edit.setText(text)
        if text in self.BOFF_ABILITY_PROPERTIES:
            career, _ = self.BOFF_ABILITY_PROPERTIES[text]
            self._slot_combo.setCurrentText(f'Boff {career}')

    def _on_train(self):
        if self._train_worker and self._train_worker.isRunning():
            return
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox
        last_repeats = int(self._settings.value(_KEY_TRAIN_REPEATS, 1))
        dlg = QDialog(self)
        dlg.setWindowTitle('Train Model')
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel('How many passes over the data?'))
        hl = QHBoxLayout()
        hl.addWidget(QLabel('Passes:'))
        spin = QSpinBox()
        spin.setRange(1, 20)
        spin.setValue(last_repeats)
        hl.addWidget(spin)
        vl.addLayout(hl)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vl.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        train_repeats = spin.value()
        self._settings.setValue(_KEY_TRAIN_REPEATS, train_repeats)
        self._train_dlg = _TrainProgressDialog(parent=self)
        self._train_dlg.cancelled.connect(self._on_train_cancelled)
        self._train_dlg.show()
        data_root = self._sets_root / 'warp' / 'training_data'
        models_dir = self._sets_root / 'warp' / 'models'
        data_mgr = self._data_mgr
        sets_root = self._sets_root
        class _CombinedTrainWorker(QThread):
            progress = Signal(int, str)
            finished = Signal(bool, str)
            def run(self_):
                interrupted = self_.isInterruptionRequested
                from warp.trainer.screen_type_trainer import ScreenTypeTrainerWorker
                w = ScreenTypeTrainerWorker(data_root, models_dir)
                w.run(lambda p, m: self_.progress.emit(int(p*0.45), f'[Screen] {m}'), lambda ok, msg: None, interrupted)
                from warp.trainer.local_trainer import LocalTrainWorker as _LTW
                icon_worker = _LTW.__new__(_LTW)
                icon_worker._data_mgr = data_mgr
                icon_worker._sets_root = sets_root
                class _FakeSignal:
                    def __init__(self2, cb): self2._cb = cb
                    def emit(self2, *a): self2._cb(*a)
                results = {}
                icon_worker.progress = _FakeSignal(lambda pct, msg: self_.progress.emit(45 + int(pct * 0.55), f'[Icons] {msg}'))
                icon_worker.finished = _FakeSignal(lambda ok, msg: results.update({'ok': ok, 'msg': msg}))
                icon_worker._train()
                self_.finished.emit(results.get('ok', False), results.get('msg', 'Done'))
        self._train_worker = _CombinedTrainWorker(parent=self)
        self._train_worker.progress.connect(self._train_dlg.update_progress)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.start()

    def _on_train_cancelled(self):
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.requestInterruption()
            self._train_worker.wait(3000)
        if self._train_dlg:
            self._train_dlg.close()
            self._train_dlg = None
        self.statusBar().showMessage('Training cancelled.')

    def _on_train_finished(self, success: bool, message: str):
        if self._train_dlg:
            self._train_dlg.close()
            self._train_dlg = None
        if success:
            try:
                from warp.recognition.layout_detector import LayoutDetector
                detector = LayoutDetector()
                learned_count = 0
                for path in self._screenshots:
                    anns = self._data_mgr.get_annotations(path)
                    confirmed = [{'bbox': a.bbox, 'slot': a.slot} for a in anns if a.state == AnnotationState.CONFIRMED]
                    if confirmed:
                        import cv2
                        img = cv2.imread(str(path))
                        if img is not None:
                            detector.learn_layout(self._screen_types.get(path.name, 'UNKNOWN'), img.shape[:2], confirmed)
                            learned_count += 1
                log.info(f"Learned layouts from {learned_count} files")
            except Exception as e:
                log.warning(f"Layout learning failed: {e}")
            QMessageBox.information(self, 'Training Complete', message)
            self._recognition_cache.clear()
            try:
                from warp.recognition.icon_matcher import SETSIconMatcher
                SETSIconMatcher.reset_ml_session()
            except:
                pass
            if self._screenshots:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(200, lambda: self._start_screen_type_detection("train_finished"))
        else:
            QMessageBox.warning(self, 'Training Failed', message)
        self._train_worker = None

    def _auto_sync(self):
        try:
            token = (Path(__file__).parent.parent / 'warp' / 'hub_token.txt').read_text().strip()
            if not token or token == 'YOUR_HF_TOKEN_HERE' or (self._sync_worker and self._sync_worker.isRunning()):
                return
            self._sync_worker = SyncWorker(data_manager=self._data_mgr, hf_token=token, mode='upload')
            self._sync_worker.finished.connect(lambda ok: self.statusBar().showMessage('Synced.' if ok else 'Sync failed.'))
            self._sync_worker.start()
        except:
            pass

    def _update_progress(self):
        total = len(self._screenshots)
        annotated = sum(1 for p in self._screenshots if self._data_mgr.has_annotations(p))
        self._prog_lbl.setText(f'{annotated} / {total} annotated')
        self._prog_bar.setValue(int(100 * annotated / max(1, total)))
        for row, p in enumerate(self._screenshots):
            item = self._file_list.item(row)
            if item:
                stype = self._screen_types.get(p.name, 'UNKNOWN')
                icon = SCREEN_TYPE_ICONS.get(stype, '?')
                label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                item.setText(f'{icon} {label}\n  {p.name}')
                if self._data_mgr.has_annotations(p):
                    item.setForeground(QColor('#7effc8'))
                else:
                    item.setForeground(Qt.GlobalColor.white)

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(8):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')

    def closeEvent(self, event):
        if self._data_mgr:
            self._data_mgr.save()
        event.accept()
