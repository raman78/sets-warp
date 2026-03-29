# warp/trainer/trainer_window.py
# WARP CORE — Interactive ML trainer + recognition review.
# PySide6, integrated with SETS.

from __future__ import annotations

import logging
from pathlib import Path
import json
import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QComboBox, QLineEdit, QGroupBox,
    QProgressBar, QToolBar, QStatusBar, QMessageBox,
    QInputDialog, QSizePolicy, QFrame, QScrollArea,
    QAbstractItemView, QCompleter, QMenu, QPlainTextEdit,
    QCheckBox, QDoubleSpinBox, QStyledItemDelegate
)
from PySide6.QtCore import Qt, QSettings, QThread, Signal, QSortFilterProxyModel, QSize, QTimer
from PySide6.QtGui import QFont, QAction, QColor, QStandardItemModel, QStandardItem, QKeySequence, QShortcut, QBrush, QPalette


class _ColorPreservingDelegate(QStyledItemDelegate):
    """Keep item's ForegroundRole color visible even when the row is selected."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if isinstance(brush, QBrush) and brush.color().isValid():
            option.palette.setColor(QPalette.ColorRole.HighlightedText, brush.color())


from warp.trainer.annotation_widget import AnnotationWidget
from warp.trainer.training_data      import TrainingDataManager, AnnotationState
from warp.trainer.sync               import SyncWorker, HFTokenDialog
from warp.style import (
    apply_dark_style, primary_btn_style, secondary_btn_style,
    warning_btn_style, danger_btn_style, toggle_yellow_btn_style,
    ACCENT, FG, MFG, BG, MBG, LBG, BC, C_WARNING, C_SUCCESS, C_FAILURE,
)

log = logging.getLogger(__name__)

_KEY_LAST_DIR       = 'warp_core/last_dir'
_KEY_HF_TOKEN       = 'warp_core/hf_token'
_KEY_AUTO_ACCEPT    = 'warp_core/auto_accept_enabled'
_KEY_AUTO_CONF      = 'warp_core/auto_accept_conf'

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

FIXED_VALUE_SLOTS: frozenset[str] = frozenset(['Ship Tier', 'Ship Type'])
from warp.trainer.training_data import NON_ICON_SLOTS  # Ship Name/Type/Tier — no ML crops
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
        self._sets_root = self._find_sets_root() # Need sets_root for stats file
        
    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(8):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')

    def run(self):
        from src.setsdebug import log as _slog
        results: dict[str, str] = {}
        total = len(self._paths)
        classifier = None

        # Statistics collection
        stats_total_per_type = {st: 0 for st in SCREEN_TYPE_LABELS.keys()}
        stats_correct_per_type = {st: 0 for st in SCREEN_TYPE_LABELS.keys()}
        overall_correct = 0

        if self._models_dir is not None:
            try:
                from warp.recognition.screen_classifier import ScreenTypeClassifier
                classifier = ScreenTypeClassifier(self._models_dir)
                _slog.info(f'ScreenTypeDetector: classifier loaded from {self._models_dir}')
            except Exception as e:
                _slog.warning(f'ScreenTypeDetector: classifier unavailable — {e}')
                _slog.info('ScreenTypeDetector: will use UNKNOWN for all (no model trained yet)')
        else:
            _slog.warning('ScreenTypeDetector: no models_dir — all results will be UNKNOWN')
        import cv2
        _slog.info(f'ScreenTypeDetector: starting — {total} screenshot(s)')
        for idx, path in enumerate(self._paths):
            if self.isInterruptionRequested():
                _slog.info('ScreenTypeDetector: interrupted')
                break
            stype = 'UNKNOWN'
            is_correct = False
            try:
                img = cv2.imread(str(path))
                if img is None:
                    _slog.warning(f'ScreenTypeDetector: cannot read {path.name}')
                elif classifier is None:
                    # _slog.info(f'ScreenTypeDetector: [{idx+1}/{total}] {path.name} → UNKNOWN (no classifier)')
                    pass # Suppress per-file log
                else:
                    ml_stype, ml_conf = classifier.classify(img)
                    if ml_stype and ml_conf >= 0.70:
                        stype = ml_stype
                        is_correct = True
                        # _slog.info(f'ScreenTypeDetector: [{idx+1}/{total}] {path.name} → {stype} (conf={ml_conf:.2f})')
                    else:
                        # _slog.info(f'ScreenTypeDetector: [{idx+1}/{total}] {path.name} → UNKNOWN (best={ml_stype!r} conf={ml_conf:.2f} < 0.70)')
                        pass # Suppress per-file log
            except Exception as e:
                _slog.warning(f'ScreenTypeDetector: [{idx+1}/{total}] {path.name} → error: {e}')
            
            results[path.name] = stype
            stats_total_per_type[stype] = stats_total_per_type.get(stype, 0) + 1
            if is_correct:
                stats_correct_per_type[stype] = stats_correct_per_type.get(stype, 0) + 1
                overall_correct += 1

            self.progress.emit(idx + 1, total, path.name, stype)
        
        self._log_screen_type_stats(total, overall_correct, stats_total_per_type, stats_correct_per_type)
        _slog.info(f'ScreenTypeDetector: done — {len(results)} processed')
        self.finished.emit(results)

    def _log_screen_type_stats(self, total_files: int, overall_correct: int, 
                               stats_total_per_type: dict, stats_correct_per_type: dict):
        import json
        import datetime
        from src.setsdebug import log as _slog

        stats_path = self._sets_root / '.config' / 'screen_type_stats.json'
        
        # Load historical data
        try:
            history: list[dict] = json.loads(stats_path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        current_session_stats = {
            "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
            "total_files": total_files,
            "overall_accuracy": round(overall_correct / total_files, 2) if total_files > 0 else 0.0,
            "types": {}
        }

        summary_lines = []
        summary_lines.append(f'ScreenTypeDetector: Summary for {total_files} files:')
        summary_lines.append(f'  Overall Accuracy: {current_session_stats["overall_accuracy"]:.0%}')

        for stype_key in sorted(SCREEN_TYPE_LABELS.keys()):
            if stype_key == 'UNKNOWN': continue # Skip UNKNOWN for detailed stats
            total_for_type = stats_total_per_type.get(stype_key, 0)
            correct_for_type = stats_correct_per_type.get(stype_key, 0)
            
            accuracy = round(correct_for_type / total_for_type, 2) if total_for_type > 0 else 0.0
            current_session_stats["types"][stype_key] = {
                "total": total_for_type,
                "correct": correct_for_type,
                "accuracy": accuracy
            }

            # Calculate trend
            trend_icon = '→'
            prev_accuracies = [
                s["types"][stype_key]["accuracy"] for s in history 
                if stype_key in s["types"] and s["types"][stype_key]["total"] > 0
            ]
            if prev_accuracies:
                avg_prev_accuracy = sum(prev_accuracies) / len(prev_accuracies)
                if accuracy > avg_prev_accuracy + 0.02: # 2% improvement threshold
                    trend_icon = '↑'
                elif accuracy < avg_prev_accuracy - 0.02: # 2% degradation threshold
                    trend_icon = '↓'
            
            if total_for_type > 0:
                summary_lines.append(
                    f'  {SCREEN_TYPE_ICONS.get(stype_key, "?")} {SCREEN_TYPE_LABELS[stype_key]:<20}: '
                    f'{correct_for_type}/{total_for_type} ({accuracy:.0%}) {trend_icon}'
                )
        
        # Add UNKNOWN stats separately
        unknown_total = stats_total_per_type.get('UNKNOWN', 0)
        if unknown_total > 0:
            summary_lines.append(f'  {SCREEN_TYPE_ICONS.get("UNKNOWN", "?")} UNKNOWN             : {unknown_total} files')
            current_session_stats["types"]["UNKNOWN"] = {"total": unknown_total, "correct": 0, "accuracy": 0.0}


        # Log the summary
        for line in summary_lines:
            _slog.info(line)

        # Save current session stats to history (keep last 50 sessions)
        history.append(current_session_stats)
        history = history[-50:] # Keep only the last 50 sessions
        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
        except Exception as e:
            _slog.warning(f'ScreenTypeDetector: Failed to save stats history: {e}')


class OCRWorker(QThread):
    finished = Signal(int, str, float, object, str)  # row, text, conf, crop_bgr, ocr_raw

    def __init__(self, row: int, crop_bgr, slot: str, valid_tiers: list, valid_types: list, parent=None):
        super().__init__(parent)
        self.row = row
        self.crop_bgr = crop_bgr
        self.slot = slot
        self.valid_tiers = valid_tiers
        self.valid_types = valid_types

    def run(self):
        try:
            from warp.recognition.text_extractor import TextExtractor, RE_TIER
            import cv2
            
            # Upscale 2x for better small font recognition
            crop_proc = cv2.resize(self.crop_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            extractor = TextExtractor()
            result = extractor._get_ocr().readtext(crop_proc)
            
            if not result:
                self.finished.emit(self.row, '', 0.0, self.crop_bgr, '')
                return
                
            full_text = ' '.join([res[1] for res in result]).strip()
            best_conf = max(res[2] for res in result)
            ocr_raw = full_text
            
            # Typo correction (Learning)
            from pathlib import Path
            import json
            typo_file = Path('warp') / 'training_data' / 'ocr_typos.json'
            if typo_file.exists():
                try:
                    with open(typo_file, 'r') as f:
                        typos = json.load(f)
                    if full_text in typos:
                        full_text = typos[full_text]
                        best_conf = 1.0 # Force max conf since user previously corrected it
                except: pass
            
            text = full_text
            conf = best_conf
            
            if self.slot == 'Ship Tier':
                import re
                m = re.search(RE_TIER, full_text)
                if m:
                    extracted = m.group(1).upper()
                    import difflib
                    matches = difflib.get_close_matches(extracted, self.valid_tiers, n=1, cutoff=0.7)
                    if matches:
                        text = matches[0]
                        conf = 1.0
                    else:
                        text = ''
                        conf = 0.0
                else:
                    text = ''
                    conf = 0.0
            elif self.slot == 'Ship Type':
                import difflib
                matches = difflib.get_close_matches(full_text, self.valid_types, n=1, cutoff=0.6)
                if matches:
                    text = matches[0]
                    conf = 1.0
                else:
                    text = ''
                    conf = 0.0
            
            # Ship Name uses full_text directly
            from src.setsdebug import log as _slog
            _slog.info(f"ocr_worker slot={self.slot!r} raw={ocr_raw!r} → final={text!r} conf={conf:.2f}")
            self.finished.emit(self.row, text, conf, self.crop_bgr, ocr_raw)
        except Exception as e:
            from src.setsdebug import log as _slog
            _slog.warning(f'OCRWorker failed: {e}')
            self.finished.emit(self.row, '', 0.0, self.crop_bgr, '')


class MatchWorker(QThread):
    """Background icon matching (two-pass) to keep UI responsive during bbox draw."""
    finished = Signal(str, float, object, object, tuple)  # name, conf, thumb, crop_bgr, bbox

    def __init__(self, crop_bgr, bbox: tuple, candidate_names, sets_app, parent=None):
        super().__init__(parent)
        self._crop = crop_bgr
        self._bbox = bbox
        self._candidates = candidate_names
        self._sets = sets_app

    def run(self):
        name, conf, thumb = '', 0.0, None
        try:
            from warp.recognition.icon_matcher import SETSIconMatcher
            from src.setsdebug import log as _slog
            name, conf, thumb, _ = SETSIconMatcher(self._sets).match(
                self._crop, candidate_names=self._candidates)
            _slog.info(f'match_worker pass1 → name={name!r} conf={conf:.2f} '
                       f'(pool={len(self._candidates) if self._candidates else "all"})')
            if conf < 0.40 and self._candidates:
                name2, conf2, thumb2, _ = SETSIconMatcher(self._sets).match(
                    self._crop, candidate_names=None)
                _slog.info(f'match_worker pass2 (unrestricted) → name={name2!r} conf={conf2:.2f}')
                if conf2 > conf:
                    name, conf, thumb = name2, conf2, thumb2
            if conf < 0.40:
                _slog.info(f'match_worker: conf {conf:.2f} < 0.40 — treating as unmatched')
                name, conf, thumb = '', 0.0, None
        except Exception as e:
            from src.setsdebug import log as _slog
            _slog.warning(f'MatchWorker failed: {e}')
        self.finished.emit(name, conf, thumb, self._crop, self._bbox)


class RecognitionWorker(QThread):
    finished = Signal(list)
    error    = Signal(str)
    def __init__(self, path, stype: str, sets_app, parent=None):
        super().__init__(parent)
        self._path = path
        self._stype = stype
        self._sets_app = sets_app
    def run(self):
        import cv2
        from src.setsdebug import log as _slog
        from warp.warp_importer import WarpImporter

        # Load image once — reused for inference, recognition pipeline, and crop extraction
        img = cv2.imread(str(self._path))
        if img is None:
            _slog.warning(f'RecognitionWorker: cannot read image {self._path}')
            self.finished.emit([])
            return
        _slog.info(f'RecognitionWorker: image loaded {img.shape[1]}x{img.shape[0]} px')

        # Map trainer screen type → WarpImporter build_type
        _STYPE_MAP = {
            'SPACE_EQ':        'SPACE',
            'GROUND_EQ':       'GROUND',
            'TRAITS':          'SPACE_TRAITS',   # refined below via CNN
            'BOFFS':           'BOFFS',
            'SPECIALIZATIONS': 'SPEC',
            'SPACE_MIXED':     'SPACE',    # WarpImporter has no SPACE_MIXED order; confirmed_layout handles extras
            'GROUND_MIXED':    'GROUND',   # same rationale
        }
        importer_type = _STYPE_MAP.get(self._stype)   # None → UNKNOWN

        # UNKNOWN screens default to SPACE; TRAITS screens stay as SPACE_TRAITS
        if importer_type is None:
            importer_type = 'SPACE'
            _slog.info(f'RecognitionWorker: UNKNOWN screen — defaulting to SPACE')

        _slog.info(f'RecognitionWorker: start {self._path.name} stype={self._stype} → importer={importer_type}')

        # Build profile_override from confirmed annotations for this file
        profile_override = {}
        try:
            data_mgr = getattr(self._sets_app, '_warp_core_window', None)
            data_mgr = getattr(data_mgr, '_data_mgr', None)
            if data_mgr:
                anns = data_mgr.get_annotations(self._path)
                for a in anns:
                    if a.state.value == 'confirmed' and a.slot:
                        profile_override[a.slot] = profile_override.get(a.slot, 0) + 1
                if profile_override:
                    _slog.info(f'RecognitionWorker: profile_override from confirmed: {profile_override}')
        except Exception as _pe:
            _slog.debug(f'RecognitionWorker: profile_override failed: {_pe}')

        try:
            importer = WarpImporter(sets_app=self._sets_app, build_type=importer_type, from_trainer=True)
            result = importer._process_image(img, str(self._path), profile_override=profile_override or None)
            _slog.info(f'RecognitionWorker: pipeline done — {len(result.items)} items found')
            for e in result.errors:
                _slog.warning(f'RecognitionWorker: pipeline error: {e}')

            # Cross-check layout vs content
            cross_check_failed_items = set()
            try:
                xcheck = WarpImporter(sets_app=self._sets_app)
                for item in result.items:
                    if not xcheck._item_valid_for_slot(item.name, item.slot):
                        _slog.info(f'RecognitionWorker: cross-check warning: {item.name!r} invalid for {item.slot!r}')
                        cross_check_failed_items.add((item.slot, item.name))
            except:
                pass
        except Exception as e:
            _slog.warning(f'RecognitionWorker: exception — {e}')
            self.error.emit(str(e))
            return

        items = []
        for ri in result.items:
            crop_bgr = None
            if ri.bbox is not None:
                try:
                    x, y, w, h = ri.bbox
                    crop_bgr = img[y:y+h, x:x+w].copy()
                except:
                    pass
            _slog.info(f'RecognitionWorker:   slot={ri.slot!r:25} name={ri.name!r:40} conf={ri.confidence:.2f} bbox={ri.bbox}')
            cross_check = (ri.slot, ri.name) in cross_check_failed_items
            items.append({'name': ri.name, 'slot': ri.slot, 'conf': ri.confidence, 'bbox': ri.bbox,
                          'state': 'pending', 'thumb': ri.thumbnail, 'crop_bgr': crop_bgr,
                          'orig_name': ri.name, 'ship_name': result.ship_name,
                          'cross_check_failed': cross_check})
        _slog.info(f'RecognitionWorker: emitting {len(items)} items')
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
        apply_dark_style(self)
        self._title_lbl = QLabel('Classifying screenshots with ML model…')
        self._title_lbl.setFont(QFont('', 10, QFont.Weight.Bold))
        self._title_lbl.setStyleSheet(f'color:{ACCENT};')
        self._file_lbl = QLabel('')
        self._file_lbl.setStyleSheet(f'color:{MFG};font-size:10px;')
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
        apply_dark_style(self)
        title = QLabel('Matching icons against SETS library...')
        title.setFont(QFont('', 10, QFont.Weight.Bold))
        title.setStyleSheet(f'color:{ACCENT};')
        file_lbl = QLabel(f'{icon} {label}   {filename}')
        file_lbl.setStyleSheet(f'color:{MFG};font-size:10px;')
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
        self._loading_row = False
        self._sync_client = None
        self._sync_timer = None
        self._init_sync_client()
        self._detect_worker = None
        self._suppress_next_focus_popup = False  # set True after programmatic setFocus
        self._recog_worker = None
        self._detect_dlg = None
        self._recog_dlg = None
        self._sync_worker  = None
        self._pending_sync = False   # set True when confirmed data changes; timer flushes
        self._selection_just_changed = False
        self.setWindowTitle('WARP CORE — ML Trainer')
        self.setMinimumSize(1280, 740)
        apply_dark_style(self)
        self._build_ui()
        self._setup_shortcuts()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Ready — open a folder of STO screenshots to start annotating.')

        # Periodic HF sync — fires every 5 minutes, uploads only if data changed
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(5 * 60 * 1000)   # 5 minutes in ms
        self._sync_timer.timeout.connect(self._on_sync_timer)
        self._sync_timer.start()


    def showEvent(self, event):
        """Ensure canvas has focus once window is shown."""
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        if hasattr(self, '_ann_widget'):
            self._ann_widget.setFocus()

    def _set_popup_transient(self, popup) -> None:
        """Wayland fix: attach completer popup QWindow to main window so xdg_popup works."""
        wh = popup.windowHandle()
        mwh = self.windowHandle()
        if wh and mwh:
            wh.setTransientParent(mwh)

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
        sp.setSizes([400, 700, 400])
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 0)
        sp.setSizes([220, 700, 400])

    def _make_left_panel(self) -> QWidget:
        left = QWidget()
        left.setMinimumWidth(400)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)
        lbl = QLabel('Screenshots')
        lbl.setFont(QFont('', 10, QFont.Weight.Bold))
        ll.addWidget(lbl)
        self._file_list = QListWidget()
        self._file_list.setItemDelegate(_ColorPreservingDelegate(self._file_list))
        self._file_list.currentRowChanged.connect(self._load_screenshot)
        self._file_list.itemChanged.connect(self._on_file_item_changed)
        self._file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._file_list.customContextMenuRequested.connect(self._show_file_list_context_menu)
        ll.addWidget(self._file_list, 1)
        self._prog_lbl = QLabel('0 / 0 annotated')
        self._prog_lbl.setStyleSheet(f'color:{MFG};font-size:10px;')
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
        center.setMinimumWidth(400)
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self._ann_widget = AnnotationWidget(self._data_mgr)
        self._ann_widget.installEventFilter(self)
        self._ann_widget.annotation_added.connect(self._on_bbox_drawn)
        self._ann_widget.item_selected.connect(self._on_item_selected)
        self._ann_widget.item_deselected.connect(self._on_canvas_deselected)
        self._ann_widget.bbox_changed.connect(self._on_bbox_changed)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(self._ann_widget)
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setStyleSheet(f'QScrollArea {{ background: {BG}; border: none; }}')
        cl.addWidget(self._scroll_area, 1)
        # Fixed-height frame so progress bar never shifts the canvas or bottom panel
        _pf = QWidget()
        _pf.setFixedHeight(6)
        _pf_lay = QHBoxLayout(_pf)
        _pf_lay.setContentsMargins(0, 0, 0, 0)
        self._match_progress = QProgressBar()
        self._match_progress.setRange(0, 0)
        self._match_progress.setFixedHeight(6)
        self._match_progress.setTextVisible(False)
        self._match_progress.setVisible(False)
        _pf_lay.addWidget(self._match_progress)
        cl.addWidget(_pf)
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
        # Wayland: store popup widget and install event filter so we can set
        # transientParent on first Show (handle does not exist until then).
        self._completer_popup = self._completer.popup()
        self._completer_popup.installEventFilter(self)
        nc.addWidget(self._name_edit)
        self._tier_combo = QComboBox()
        for t in SHIP_TIER_VALUES:
            self._tier_combo.addItem(t)
        self._tier_combo.hide()
        self._tier_combo.textActivated.connect(
            lambda _: self._on_accept() if self._slot_combo.currentText() == 'Ship Tier' else None)
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
        self._ship_type_combo.textActivated.connect(
            lambda _: self._on_accept() if self._slot_combo.currentText() == 'Ship Type' else None)
        stc.activated.connect(
            lambda _: self._on_accept() if self._slot_combo.currentText() == 'Ship Type' else None)
        nc.addWidget(self._ship_type_combo)
        lay.addLayout(nc, 1)
        bc = QVBoxLayout()
        bc.addStretch()
        br = QHBoxLayout()
        self._btn_accept = QPushButton('Accept')
        self._btn_accept.setStyleSheet(primary_btn_style())
        self._btn_accept.clicked.connect(self._on_accept)
        self._btn_accept.setToolTip('Accept (Enter)')
        br.addWidget(self._btn_accept)
        self._chk_auto_accept = QCheckBox('Auto ≥')
        self._chk_auto_accept.setToolTip(
            'Auto-accept items where ML confidence meets threshold')
        self._chk_auto_accept.setChecked(False)
        self._spin_auto_conf = QDoubleSpinBox()
        self._spin_auto_conf.setRange(0.5, 1.0)
        self._spin_auto_conf.setSingleStep(0.05)
        self._spin_auto_conf.setValue(0.75)
        self._spin_auto_conf.setDecimals(2)
        self._spin_auto_conf.setFixedWidth(72)
        self._spin_auto_conf.setToolTip('Min confidence for auto-accept')
        br.addWidget(self._chk_auto_accept)
        br.addWidget(self._spin_auto_conf)
        bc.addLayout(br)
        lay.addLayout(bc)
        return g

    def _make_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(400)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(6, 8, 6, 8)
        pl.setSpacing(6)
        hdr = QLabel('Recognition Review')
        hdr.setFont(QFont('', 10, QFont.Weight.Bold))
        hdr.setStyleSheet(f'color:{ACCENT};')
        pl.addWidget(hdr)
        hint = QLabel('Green = high confidence\nYellow = uncertain  ·  Red = unmatched\nClick item to select on canvas.')
        hint.setWordWrap(True)
        hint.setStyleSheet(f'color:{MFG};font-size:10px;')
        pl.addWidget(hint)
        self._screen_type_badge = QLabel('Screen type: —')
        self._screen_type_badge.setStyleSheet(f'color:{C_WARNING};background:{MBG};border:1px solid {LBG};border-radius:3px;padding:2px 6px;font-size:11px;')
        pl.addWidget(self._screen_type_badge)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color:{BC};')
        pl.addWidget(sep)
        self._review_list = QListWidget()
        self._review_list.setItemDelegate(_ColorPreservingDelegate(self._review_list))
        self._review_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._review_list.currentRowChanged.connect(self._on_review_row_changed)
        self._review_list.installEventFilter(self)
        # ann_widget event filter set after creation in _make_center_panel
        self._review_list.itemClicked.connect(self._on_review_item_clicked)
        pl.addWidget(self._review_list, 1)
        self._review_summary = QLabel('')
        self._review_summary.setStyleSheet(f'color:{MFG};font-size:10px;')
        self._review_summary.setWordWrap(True)
        pl.addWidget(self._review_summary)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f'color:{BC};')
        pl.addWidget(sep2)
        self._btn_edit_bbox = QPushButton('Edit BBox')
        self._btn_edit_bbox.setStyleSheet(secondary_btn_style(checked_border=True))
        self._btn_edit_bbox.setCheckable(True)
        self._btn_edit_bbox.clicked.connect(self._on_edit_bbox_toggle)
        self._btn_edit_bbox.setVisible(False)  # Resize/move disabled — reserved for future
        pl.addWidget(self._btn_edit_bbox)
        mgmt = QHBoxLayout()
        self._btn_add_bbox = QPushButton('+ Add BBox')
        self._btn_add_bbox.setStyleSheet(warning_btn_style(checked_border=True))
        self._btn_add_bbox.setCheckable(True)
        self._btn_add_bbox.clicked.connect(self._on_add_bbox_toggle)
        self._btn_remove_item = QPushButton('- Remove BBox')
        self._btn_remove_item.setStyleSheet(danger_btn_style())
        self._btn_remove_item.clicked.connect(self._on_remove_item)
        mgmt.addWidget(self._btn_add_bbox)
        mgmt.addWidget(self._btn_remove_item)
        pl.addLayout(mgmt)
        self._manual_mode_lbl = QLabel('')
        self._manual_mode_lbl.setStyleSheet(f'color:{C_WARNING};font-size:10px;background:{MBG};border:1px solid {LBG};border-radius:3px;padding:3px;')
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

    def _on_open(self):
        last = self._settings.value(_KEY_LAST_DIR, '')
        dlg = QFileDialog(self)
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
                    self._file_list.blockSignals(True)
                    item.setText(f'{icon} {label}\n  {filename}')
                    item.setCheckState(Qt.CheckState.Unchecked)  # auto-detected = unconfirmed
                    self._file_list.blockSignals(False)
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
                confirmed = p.name in self._screen_types_manual
                self._file_list.blockSignals(True)
                item.setText(f'{icon} {label}\n  {p.name}')
                item.setCheckState(
                    Qt.CheckState.Checked if confirmed else Qt.CheckState.Unchecked)
                self._file_list.blockSignals(False)
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
        confirmed = p.name in self._screen_types_manual
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked if confirmed else Qt.CheckState.Unchecked)
        if self._data_mgr.has_annotations(p):
            item.setForeground(QColor('#7effc8'))
        else:
            item.setForeground(Qt.GlobalColor.white)
        return item

    def _update_file_item_check(self, row: int):
        """Sync checkbox state of file list item at row with _screen_types_manual."""
        item = self._file_list.item(row)
        if item is None or row >= len(self._screenshots):
            return
        fname = self._screenshots[row].name
        confirmed = fname in self._screen_types_manual
        self._file_list.blockSignals(True)
        item.setCheckState(
            Qt.CheckState.Checked if confirmed else Qt.CheckState.Unchecked)
        self._file_list.blockSignals(False)

    def _on_file_item_changed(self, item: QListWidgetItem):
        """Checkbox toggled by user on file list item."""
        row = self._file_list.row(item)
        if row < 0 or row >= len(self._screenshots):
            return
        path = self._screenshots[row]
        is_checked = item.checkState() == Qt.CheckState.Checked
        stype = self._screen_types.get(path.name, 'UNKNOWN')
        if is_checked:
            # User confirms the current ML-detected type
            self._screen_types_manual.add(path.name)
            self._save_screen_type_example(path, stype)
        else:
            # User un-confirms — remove from manual set
            # (type label stays, but will be re-detectable by ML next time)
            self._screen_types_manual.discard(path.name)
            self._data_mgr.remove_screen_type(path, stype)
            # save() already called inside remove_screen_type

    def _load_screenshot(self, row: int):
        if row < 0 or row >= len(self._screenshots): return
        self._current_idx = row; path = self._screenshots[row]; stype = self._screen_types.get(path.name, 'UNKNOWN')
        self._ann_widget.load_image(path); self._exit_manual_bbox_mode(); self._update_screen_type_ui(stype)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self._ann_widget.setFocus)
        # Clear Item Name and reset completer when switching to a new screenshot
        self._completer.setCompletionPrefix('')
        self._name_edit.blockSignals(True)
        self._name_edit.clear()
        self._name_edit.blockSignals(False)
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

    def _refresh_slot_combo(self, stype: str, keep_slot: str = ''):
        """Rebuild slot combo for screen type, hiding confirmed NON_ICON_SLOTS.

        keep_slot: slot of the currently displayed item — always kept visible
        so the user can read/edit an already-confirmed Ship Name/Type/Tier bbox.
        """
        group_key = SCREEN_TO_SLOT_GROUP.get(stype, 'SPACE_EQ')
        slots = SLOT_GROUPS.get(group_key, ALL_SLOTS)
        # Hide confirmed NON_ICON_SLOTS, but always show the currently active one
        confirmed_non_icon: set[str] = set()
        if self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            confirmed_non_icon = {
                ann.slot for ann in self._data_mgr.get_annotations(path)
                if ann.state == AnnotationState.CONFIRMED and ann.slot in NON_ICON_SLOTS
            }
        confirmed_non_icon.discard(keep_slot)
        current_slot = self._slot_combo.currentText()
        self._slot_combo.blockSignals(True)
        self._slot_combo.clear()
        for s in slots:
            if s not in confirmed_non_icon:
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
            self._file_list.blockSignals(True)
            item.setText(f'{icon} {label}\n  {path.name}')
            item.setCheckState(Qt.CheckState.Checked)
            self._file_list.blockSignals(False)
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

    def _seed_matcher_from_confirmed(self, path: Path):
        """
        Prime SETSIconMatcher with confirmed training-data crops before Auto-Detect.
        Delegates to seed_from_training_data (all confirmed crops, guarded against
        re-seeding). New in-session confirmations are already added live via
        add_session_example in _on_accept / _on_accept_all.
        """
        try:
            from warp.recognition.icon_matcher import SETSIconMatcher
            SETSIconMatcher.seed_from_training_data(self._sets_root / 'warp' / 'training_data')
        except Exception as e:
            log.warning(f'seed_matcher_from_confirmed failed: {e}')

    def _on_auto_detect(self):
        if self._current_idx < 0: return
        path = self._screenshots[self._current_idx]
        stype = self._screen_types.get(path.name, 'UNKNOWN')

        # Seed the icon matcher with all confirmed crops from this image
        # so Auto-Detect benefits from what the user has already confirmed
        self._seed_matcher_from_confirmed(path)

        # Remove only non-confirmed items from cache; keep confirmed intact
        existing = self._recognition_cache.get(path.name, [])
        confirmed_items = [ri for ri in existing if ri.get('state') == 'confirmed']
        self._recognition_cache.pop(path.name, None)
        self._start_recognition(path, stype, preserve_confirmed=confirmed_items)

    def _on_detect_screen_types(self):
        if not self._screenshots: return
        # Only detect files not yet manually labelled — never overwrite user choices
        self._start_screen_type_detection('detect_screen_types_button')

    def _start_recognition(self, path: Path, stype: str, preserve_confirmed: list | None = None):
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
        self._recog_worker.finished.connect(
            lambda items: self._on_recognition_done(path.name, stype, items,
                                                    preserve_confirmed=preserve_confirmed))
        self._recog_worker.error.connect(self._on_recognition_error)
        self._recog_worker.start()

    def _on_recognition_done(self, filename: str, stype: str, items: list,
                             preserve_confirmed: list | None = None):
        if self._recog_dlg:
            self._recog_dlg.close()
            self._recog_dlg = None
        # Merge: keep confirmed items, add new detections that don't overlap
        if preserve_confirmed:
            confirmed_bboxes = {ri['bbox'] for ri in preserve_confirmed if ri.get('bbox')}
            new_items = [ri for ri in items if ri.get('bbox') not in confirmed_bboxes]
            merged = preserve_confirmed + new_items
        else:
            merged = items
        self._recognition_cache[filename] = merged
        if self._current_idx >= 0 and self._screenshots[self._current_idx].name == filename:
            self._populate_review_panel(merged, stype)
            # Run auto-accept after panel is populated
            self._run_auto_accept()

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
                    log.debug(f'populate: confirmed from disk slot={ann.slot!r} '
                              f'bbox={ann.bbox} name={ann.name!r} ann_id={ann.ann_id}')
                    confirmed_by_id[ann.ann_id] = {
                    'name': ann.name, 'slot': ann.slot, 'bbox': ann.bbox,
                    'state': 'confirmed',
                    'conf': ann.ml_conf,          # real ML confidence, 0.0 if unknown
                    'orig_name': ann.ml_name or ann.name,  # what ML originally saw
                    'thumb': None, 'crop_bgr': None, 'ship_name': '', 'ann_id': ann.ann_id,
                }
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
        # Auto-accept high-conf items before drawing the list
        self._apply_auto_accept()
        self._review_list.clear()
        self._review_summary.setText('')
        self._set_review_buttons_enabled(False)
        for ri in self._recognition_items:
            self._add_review_row(ri['name'], ri['slot'], ri.get('conf', 0.0), confirmed=(ri.get('state') == 'confirmed'), cross_check_failed=ri.get('cross_check_failed', False))
        self._ann_widget.set_review_items(self._recognition_items)
        self._ann_widget.set_selected_row(-1)
        n = len(self._recognition_items)
        matched = sum(1 for i in self._recognition_items if i.get('name'))
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, stype)
        ship = (self._recognition_items[0].get('ship_name') or '--') if self._recognition_items else '--'
        self._review_summary.setText(f'{matched}/{n} identified  Ship: {ship}  {icon} {label}')
        self._set_review_buttons_enabled(n > 0)
        # Hide confirmed NON_ICON_SLOTS from slot combo for this image
        self._refresh_slot_combo(stype)
        if n > 0:
            self._review_list.setCurrentRow(0)

    def _add_review_row(self, name: str, slot: str, conf: float, confirmed: bool = False, cross_check_failed: bool = False):
        if confirmed:
            label = f'{slot}  ->  {name or "—"}  [confirmed]'
        elif cross_check_failed:
            label = f'⚠️ {slot}  ->  {name or "— unmatched —"}  [{conf:.0%}]'
        else:
            label = f'{slot}  ->  {name or "— unmatched —"}  [{conf:.0%}]'
        item = QListWidgetItem(label)
        if confirmed:
            if conf > 0.0:  # real confidence saved
                tooltip = (f'Slot: {slot}\nItem: {name or "—"}\n'
                           f'Status: confirmed by user\n'
                           f'ML recognition: {conf:.1%}')
            else:           # conf=0.0 — old annotation without saved confidence
                tooltip = (f'Slot: {slot}\nItem: {name or "—"}\n'
                           f'Status: confirmed by user\n'
                           f'ML recognition: unknown (previous session)')
        elif name:
            tooltip = f'Slot: {slot}\nItem: {name}\nConfidence: {conf:.1%}'
            if cross_check_failed:
                tooltip += '\n\n⚠️ WARNING: Item type does not match slot type!'
        else:
            tooltip = f'Slot: {slot}\nNo item recognised'
        item.setToolTip(tooltip)
        if confirmed:
            item.setForeground(QColor('#7effc8'))
        elif cross_check_failed:
            item.setForeground(QColor('#ffcc00')) # Orange/Gold warning
        elif not name:
            item.setForeground(QColor('#ff5555'))
        elif conf >= CONF_HIGH:
            item.setForeground(QColor('#ffaaaa'))
        elif conf >= CONF_MEDIUM:
            item.setForeground(QColor('#ff8888'))
        else:
            item.setForeground(QColor('#ff5555'))

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
        self._loading_row = True
        try:
            if 0 <= row < len(self._recognition_items):
                ri = self._recognition_items[row]
                is_confirmed = ri.get('state') == 'confirmed'
                self._btn_remove_item.setEnabled(True)
                # self._btn_edit_bbox.setEnabled(True)  # disabled
                # if is_confirmed:
                #     self._btn_edit_bbox.setChecked(False)
                #     self._ann_widget.set_draw_mode(False)
                slot = ri['slot']
                # Ensure this slot is visible in combo (confirmed NON_ICON_SLOTS
                # are normally hidden, but must show when the item is selected)
                if self._current_idx >= 0:
                    _stype = self._screen_types.get(
                        self._screenshots[self._current_idx].name, 'UNKNOWN')
                    self._refresh_slot_combo(_stype, keep_slot=slot)
                idx = self._slot_combo.findText(slot)
                if idx >= 0:
                    self._slot_combo.setCurrentIndex(idx)
                # Populate completer for this slot without triggering clear on name_edit
                # NON_ICON_SLOTS use their own widgets — skip completer (avoids iterating all equipment)
                if slot not in NON_ICON_SLOTS:
                    self._populate_name_completer(slot)
                # Set name field directly (slot already set above, skip _on_slot_changed clear)
                self._name_edit.blockSignals(True)
                self._name_edit.setText(ri['name'])
                self._name_edit.blockSignals(False)
                if ri.get('bbox'):
                    self._ann_widget.set_highlighted_row(row)
                else:
                    self._ann_widget.clear_highlight()
                if is_confirmed:
                    self._review_list.setFocus()
        finally:
            self._loading_row = False

    def _init_sync_client(self):
        try:
            from warp.knowledge.sync_client import WARPSyncClient
            self._sync_client = WARPSyncClient()
        except Exception as e:
            log.warning(f'WARP CORE: sync client init failed: {e}')

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
        pass  # Edit BBox disabled — reserved for future implementation

    def _setup_shortcuts(self):
        """Global keyboard shortcuts — work regardless of focus."""
        from PySide6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)
        QShortcut(QKeySequence('Alt+A'), self,
                  activated=lambda: self._btn_add_bbox.click())
        QShortcut(QKeySequence('Alt+R'), self,
                  activated=self._on_remove_item)
        QShortcut(QKeySequence('Return'), self,
                  activated=self._on_accept)
        QShortcut(QKeySequence('Delete'), self,
                  activated=self._on_remove_item)
        # Restore auto-accept settings
        self._chk_auto_accept.setChecked(
            self._settings.value(_KEY_AUTO_ACCEPT, True, type=bool))
        self._spin_auto_conf.setValue(
            float(self._settings.value(_KEY_AUTO_CONF, 0.75)))
        # Save on change
        self._chk_auto_accept.toggled.connect(
            lambda v: self._settings.setValue(_KEY_AUTO_ACCEPT, v))
        self._spin_auto_conf.valueChanged.connect(
            lambda v: self._settings.setValue(_KEY_AUTO_CONF, v))

    def eventFilter(self, obj, event):
        """Handle Delete key on review list/canvas, and forward Ctrl+wheel from scroll area to canvas."""
        from PySide6.QtCore import QEvent
        rl = getattr(self, '_review_list', None)
        aw = getattr(self, '_ann_widget', None)
        sa = getattr(self, '_scroll_area', None)
        if obj in (rl, aw) and obj is not None and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self._on_remove_item()
                return True
        # Forward wheel events from anywhere in scroll area to the canvas widget
        if event.type() == QEvent.Type.Wheel and sa and aw:
            from PySide6.QtGui import QCursor
            gpos = QCursor.pos()
            sa_pos = sa.mapFromGlobal(gpos)
            aw_pos = aw.mapFromGlobal(gpos)
            if sa.rect().contains(sa_pos) and not aw.rect().contains(aw_pos):
                aw.wheelEvent(event)
                return True
        # Wayland: set transient parent on any popup window (QComboBox, QMenu, QCompleter, …)
        from PySide6.QtWidgets import QWidget as _QW
        if (event.type() == QEvent.Type.Show
                and isinstance(obj, _QW)
                and (obj.windowFlags() & Qt.WindowType.Popup)):
            QTimer.singleShot(0, lambda o=obj: self._set_popup_transient(o))
        return super().eventFilter(obj, event)

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
        ri = self._recognition_items[row]
        if ri.get('state') == 'confirmed':
            name = ri.get('name') or ri.get('slot') or 'this item'
            reply = QMessageBox.question(
                self, 'Remove confirmed annotation',
                f'Remove confirmed bbox for "{name}"?\n\n'
                f'This will delete the saved annotation for this slot.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            # Also remove from TrainingDataManager
            if self._current_idx >= 0 and ri.get('bbox'):
                path = self._screenshots[self._current_idx]
                for ann in self._data_mgr.get_annotations(path):
                    if ann.bbox == ri['bbox']:
                        self._data_mgr.remove_annotation(path, ann)
                        self._data_mgr.save()
                        break
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
        # Restore removed NON_ICON_SLOT back to combo if it was confirmed
        if self._current_idx >= 0:
            _stype = self._screen_types.get(self._screenshots[self._current_idx].name, 'UNKNOWN')
            self._refresh_slot_combo(_stype)

    def _enter_manual_bbox_mode(self):
        pass  # Resize/move disabled — reserved for future implementation
        # self._manual_bbox_mode = True
        # self._btn_edit_bbox.setChecked(True)
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
        for btn in (self._btn_remove_item,):  # btn_edit_bbox disabled
            btn.setEnabled(enabled)

    def _update_add_bbox_btn(self):
        self._btn_add_bbox.setEnabled(self._current_idx >= 0)

    def _on_bbox_drawn(self, bbox: tuple):
        if self._manual_bbox_mode:
            row = self._review_list.currentRow()
            if 0 <= row < len(self._recognition_items):
                self._recognition_items[row]['bbox'] = bbox
            self._rematch_current_item(row, bbox)
        elif getattr(self, '_add_bbox_mode', False) \
                or getattr(self._ann_widget, '_alt_draw', False):
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
                        # ── P1: suggest slot from bbox position ──────────────
                        suggested = self._suggest_slot_from_position(bbox)
                        if suggested:
                            self._slot_combo.blockSignals(True)
                            self._slot_combo.setCurrentText(suggested)
                            self._slot_combo.blockSignals(False)
                            _slog.info(f'add_bbox: P1 slot suggestion → {suggested!r}')
                        _current_slot = self._slot_combo.currentText()
                        # If current slot is a NON_ICON_SLOT already confirmed for this image,
                        # advance to the next unconfirmed NON_ICON_SLOT to prevent
                        # SINGLE_INSTANCE step from silently deleting the earlier annotation.
                        if _current_slot in NON_ICON_SLOTS and self._current_idx >= 0:
                            _confirmed_slots = {
                                ann.slot for ann in self._data_mgr.get_annotations(path)
                                if ann.state == AnnotationState.CONFIRMED
                            }
                            if _current_slot in _confirmed_slots:
                                for _next in ('Ship Name', 'Ship Type', 'Ship Tier'):
                                    if _next not in _confirmed_slots:
                                        _current_slot = _next
                                        self._slot_combo.blockSignals(True)
                                        self._slot_combo.setCurrentText(_next)
                                        self._slot_combo.blockSignals(False)
                                        log.info(f'add_bbox: {_current_slot!r} already confirmed '
                                                 f'→ advanced slot to {_next!r}')
                                        break
                        if _current_slot not in NON_ICON_SLOTS:
                            _candidates = set(self._build_search_candidates(_current_slot)) or None
                            self._start_match_worker(crop_bgr, bbox, _candidates)
                            return
                        # NON_ICON_SLOT: icon matching skipped, fall through to _finish_bbox_drawn
                except Exception as _e:
                    from src.setsdebug import log as _slog
                    _slog.warning(f'add_bbox: error: {_e}')
            self._finish_bbox_drawn('', 0.0, None, crop_bgr, bbox)
        else:
            self._name_edit.setFocus()
            self._name_edit.clear()

    def _start_match_worker(self, crop_bgr, bbox: tuple, candidate_names) -> None:
        """Start async icon matching; show spinner after 500ms if still running."""
        if not hasattr(self, '_match_workers'):
            self._match_workers = []
        worker = MatchWorker(crop_bgr, bbox, candidate_names, self._sets, parent=self)
        worker.finished.connect(self._on_match_worker_done)
        self._match_workers.append(worker)
        QTimer.singleShot(500, lambda: self._match_progress.setVisible(
            any(w.isRunning() for w in self._match_workers)))
        worker.start()

    def _on_match_worker_done(self, name: str, conf: float, thumb, crop_bgr, bbox: tuple) -> None:
        self._match_progress.setVisible(False)
        self._finish_bbox_drawn(name, conf, thumb, crop_bgr, bbox)

    def _finish_bbox_drawn(self, name: str, conf: float, thumb, crop_bgr, bbox: tuple) -> None:
        """Finalise a drawn bbox: infer slot, add to review list, trigger OCR if needed."""
        slot = self._slot_combo.currentText()
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
            else:
                # Name found by matcher but doesn't belong to any allowed slot
                # for this screen type — discard to avoid wrong slot assignment
                from src.setsdebug import log as _slog2
                _slog2.info(f'add_bbox: discarding {name!r} — not valid for stype={stype}')
                name, conf, thumb = '', 0.0, None
        # NON_ICON_SLOTS (Ship Name/Type/Tier) — position only, always confirmed
        if slot in NON_ICON_SLOTS:
            _auto = False
        else:
            # Auto-accept before adding to list if conf >= threshold
            _auto = (name and conf > 0
                     and getattr(self, '_chk_auto_accept', None)
                     and self._chk_auto_accept.isChecked()
                     and conf >= self._spin_auto_conf.value())

        _cross_check = False
        try:
            if name:
                from warp.warp_importer import WarpImporter
                _cross_check = not WarpImporter(sets_app=self._sets)._item_valid_for_slot(name, slot)
        except: pass

        _state = 'confirmed' if _auto else 'pending'
        new_item = {'name': name, 'slot': slot, 'conf': conf, 'bbox': bbox, 'state': _state,
                    'thumb': thumb, 'crop_bgr': crop_bgr, 'orig_name': name, 'ship_name': '',
                    'cross_check_failed': _cross_check}
        self._recognition_items.append(new_item)
        if _auto and self._current_idx >= 0:
            _path = self._screenshots[self._current_idx]
            _saved = self._data_mgr.add_annotation(
                image_path=_path, bbox=bbox, slot=slot, name=name,
                state=AnnotationState.CONFIRMED, ml_conf=conf, ml_name=name)
            new_item['ann_id'] = _saved.ann_id
            if crop_bgr is not None:
                from warp.recognition.icon_matcher import SETSIconMatcher
                SETSIconMatcher.add_session_example(crop_bgr, name)
            self._data_mgr.save()
        self._add_review_row(name, slot, conf, confirmed=_auto, cross_check_failed=_cross_check)
        new_row = len(self._recognition_items) - 1
        self._review_list.setCurrentRow(new_row)
        self._set_review_buttons_enabled(True)
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        self._ann_widget.clear_pending()

        if slot in NON_ICON_SLOTS and crop_bgr is not None:
            if self._ship_type_combo.count() == 0:
                self._populate_ship_type_combo()
            v_tiers = [self._tier_combo.itemText(i) for i in range(self._tier_combo.count())]
            v_types = [self._ship_type_combo.itemText(i) for i in range(self._ship_type_combo.count())]
            worker = OCRWorker(new_row, crop_bgr, slot, v_tiers, v_types, parent=self)
            worker.finished.connect(self._on_ocr_finished)
            worker.start()
            if not hasattr(self, '_ocr_workers'): self._ocr_workers = []
            self._ocr_workers.append(worker)
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
        if not _auto:
            self._ann_widget.setFocus()
        else:
            self._review_list.setFocus()

    def _on_ocr_finished(self, row: int, text: str, conf: float, crop_bgr, ocr_raw: str = ''):
        if row < 0 or row >= len(self._recognition_items): return
        ri = self._recognition_items[row]
        slot = ri['slot']

        # Ship Name is position-only — discard OCR text, never store content
        if slot == 'Ship Name':
            text = ''
        ri['name'] = text
        ri['conf'] = conf
        ri['crop_bgr'] = crop_bgr
        ri['ocr_raw'] = ocr_raw
        
        cross_check = False
        try:
            if text:
                from warp.warp_importer import WarpImporter
                cross_check = not WarpImporter(sets_app=self._sets)._item_valid_for_slot(text, slot)
        except Exception:
            pass
        ri['cross_check_failed'] = cross_check
        
        litem = self._review_list.item(row)
        if litem:
            label = f'⚠️ {slot}  ->  {text or "— unmatched —"}  [{conf:.0%}]' if cross_check else f'{slot}  ->  {text or "— unmatched —"}  [{conf:.0%}]'
            litem.setText(label)
            if cross_check:
                litem.setForeground(QColor('#ffcc00'))
            elif not text:
                litem.setForeground(QColor('#ff5555'))
            elif conf >= 0.7:
                litem.setForeground(QColor('#7effc8'))
            elif conf >= 0.4:
                litem.setForeground(QColor('#ff8888'))
            else:
                litem.setForeground(QColor('#ff5555'))

        if self._review_list.currentRow() == row:
            self._on_item_selected(ri)

        if (text and conf >= 0.4
            and getattr(self, '_chk_auto_accept', None)
            and self._chk_auto_accept.isChecked()
            and conf >= self._spin_auto_conf.value()):
            if self._review_list.currentRow() != row:
                self._review_list.setCurrentRow(row)
            self._on_accept()


    def _rematch_current_item(self, row: int, bbox: tuple):  # noqa — kept for future use
        if row < 0 or self._current_idx < 0:
            return
        if 0 <= row < len(self._recognition_items):
            pass  # Removed early return for NON_ICON_SLOTS to allow OCR on edit
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
            name, conf, thumb, _used_sess = SETSIconMatcher(self._sets).match(crop)
            
            _cross_check = False
            try:
                if name:
                    from warp.warp_importer import WarpImporter
                    _cross_check = not WarpImporter(sets_app=self._sets)._item_valid_for_slot(name, slot)
            except: pass

            ri = self._recognition_items[row]
            
            if slot in NON_ICON_SLOTS:
                if self._ship_type_combo.count() == 0: self._populate_ship_type_combo()
                v_tiers = [self._tier_combo.itemText(i) for i in range(self._tier_combo.count())]
                v_types = [self._ship_type_combo.itemText(i) for i in range(self._ship_type_combo.count())]
                worker = OCRWorker(row, crop, slot, v_tiers, v_types, parent=self)
                worker.finished.connect(self._on_ocr_finished)
                worker.start()
                if not hasattr(self, '_ocr_workers'): self._ocr_workers = []
                self._ocr_workers.append(worker)
                return

            ri.update({'name': name, 'conf': conf, 'thumb': thumb, 'crop_bgr': crop, 'cross_check_failed': _cross_check})
            self._name_edit.setText(name)
            litem = self._review_list.item(row)
            if litem:
                colour = ('#ffcc00' if _cross_check else '#7effc8' if conf >= CONF_HIGH else '#e8c060' if conf >= CONF_MEDIUM else '#ff7e7e')
                prefix = '⚠️ ' if _cross_check else ''
                litem.setText(f'{prefix}{ri["slot"]}  ->  {name or "— unmatched —"}  [{conf:.0%}]')
                litem.setForeground(QColor(colour))
            # Auto-accept if conf >= threshold and checkbox enabled
            if (name and conf > 0
                    and getattr(self, '_chk_auto_accept', None)
                    and self._chk_auto_accept.isChecked()
                    and conf >= self._spin_auto_conf.value()
                    and ri.get('slot', '') not in NON_ICON_SLOTS):
                self._review_list.setCurrentRow(row)
                self._on_accept()
        except:
            pass

    def _suggest_slot_from_position(self, bbox: tuple) -> str:
        """
        P1 — Infer the most likely slot for a newly-drawn bbox based on its
        Y-position, comparing against:
          1. Already confirmed/pending annotations on this screenshot
          2. Learned layouts from anchors.json

        Returns the slot name or '' if no confident match.
        """
        bx, by, bw, bh = bbox
        cy = by + bh // 2  # center Y of the drawn bbox

        stype = 'UNKNOWN'
        if self._current_idx >= 0:
            stype = self._screen_types.get(
                self._screenshots[self._current_idx].name, 'UNKNOWN')
        group_key = SCREEN_TO_SLOT_GROUP.get(stype, 'ALL')
        allowed = set(SLOT_GROUPS.get(group_key, ALL_SLOTS))

        # ── Source 1: existing annotations on this screenshot ────────────────
        # Build a map: slot → average center-Y from current recognition items
        slot_y_map: dict[str, list[int]] = {}
        for ri in self._recognition_items:
            ri_bbox = ri.get('bbox')
            ri_slot = ri.get('slot', '')
            if ri_bbox and ri_slot and ri_slot in allowed:
                ri_cy = ri_bbox[1] + ri_bbox[3] // 2
                slot_y_map.setdefault(ri_slot, []).append(ri_cy)

        if slot_y_map:
            # Find the slot whose average Y is closest to our bbox
            best_slot = ''
            best_dist = float('inf')
            for slot, ys in slot_y_map.items():
                avg_y = sum(ys) / len(ys)
                dist = abs(cy - avg_y)
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot
            # Accept if within half an icon height (~30-40px typically)
            threshold = bh * 0.6
            if best_dist <= threshold:
                from src.setsdebug import log as _sl
                _sl.info(f'slot_suggest: bbox cy={cy} → {best_slot!r} (dist={best_dist:.0f}, '
                         f'threshold={threshold:.0f}, source=annotations)')
                return best_slot

        # ── Source 2: learned layouts from anchors.json ───────────────────────
        if self._current_idx >= 0:
            try:
                import cv2
                from warp.recognition.layout_detector import LayoutDetector
                build_type = self._STYPE_TO_BUILD.get(stype)
                if build_type:
                    img = cv2.imread(str(self._screenshots[self._current_idx]))
                    if img is not None:
                        h, w = img.shape[:2]
                        detector = LayoutDetector()
                        cal = detector._calibration
                        if cal and 'learned' in cal:
                            aspect = round(w / h, 3)
                            candidates = [
                                e for e in cal['learned']
                                if e['type'] == build_type
                                and abs(e['aspect'] - aspect) < 0.05
                            ]
                            if candidates:
                                layout = candidates[-1]  # most recent
                                best_slot = ''
                                best_dist = float('inf')
                                for slot_name, geo in layout['slots'].items():
                                    if slot_name not in allowed:
                                        continue
                                    if isinstance(geo, (int, float)):
                                        slot_cy = int(geo * h)
                                    else:
                                        slot_cy = int(geo['y_rel'] * h)
                                    dist = abs(cy - slot_cy)
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_slot = slot_name
                                threshold = bh * 0.8
                                if best_slot and best_dist <= threshold:
                                    from src.setsdebug import log as _sl
                                    _sl.info(f'slot_suggest: bbox cy={cy} → {best_slot!r} '
                                             f'(dist={best_dist:.0f}, threshold={threshold:.0f}, '
                                             f'source=anchors.json)')
                                    return best_slot
            except Exception as e:
                from src.setsdebug import log as _sl
                _sl.debug(f'slot_suggest: anchors lookup failed: {e}')

        return ''

    def _rematch_with_slot(self, row: int, slot: str, crop_bgr):
        """Re-run icon matching for an existing crop when the user changes the slot."""
        try:
            if slot in NON_ICON_SLOTS:
                if self._ship_type_combo.count() == 0:
                    self._populate_ship_type_combo()
                v_tiers = [self._tier_combo.itemText(i) for i in range(self._tier_combo.count())]
                v_types = [self._ship_type_combo.itemText(i) for i in range(self._ship_type_combo.count())]
                
                ri = self._recognition_items[row]
                ri['name'] = ''
                ri['slot'] = slot
                litem = self._review_list.item(row)
                if litem:
                    litem.setText(f'{slot}  ->  [Scanning...]')
                    litem.setForeground(QColor('#aaaaaa'))
                
                worker = OCRWorker(row, crop_bgr, slot, v_tiers, v_types, parent=self)
                worker.finished.connect(self._on_ocr_finished)
                worker.start()
                if not hasattr(self, '_ocr_workers'): self._ocr_workers = []
                self._ocr_workers.append(worker)
                return

            from warp.recognition.icon_matcher import SETSIconMatcher
            from src.setsdebug import log as _sl
            candidates = set(self._build_search_candidates(slot)) or None
            name, conf, thumb, _used_sess = SETSIconMatcher(self._sets).match(crop_bgr, candidate_names=candidates)
            _sl.info(f'rematch_slot slot={slot!r} candidates={len(candidates) if candidates else "all"} → name={name!r} conf={conf:.2f}')
            # No global fallback — if slot-scoped search can't match, show unmatched.
            # A global fallback would return items from wrong categories.
            if conf < 0.40:
                name, conf, thumb = '', 0.0, None
            _cross_check = False
            try:
                if name:
                    from warp.warp_importer import WarpImporter
                    _cross_check = not WarpImporter(sets_app=self._sets)._item_valid_for_slot(name, slot)
            except: pass

            ri = self._recognition_items[row]
            ri.update({'name': name, 'conf': conf, 'thumb': thumb, 'slot': slot, 'cross_check_failed': _cross_check})
            self._name_edit.blockSignals(True)
            self._name_edit.setText(name)
            self._name_edit.blockSignals(False)
            litem = self._review_list.item(row)
            if litem:
                prefix = '⚠️ ' if _cross_check else ''
                litem.setText(f'{prefix}{slot}  ->  {name or "— unmatched —"}  [{conf:.0%}]')
                if _cross_check:
                    litem.setForeground(QColor('#ffcc00'))
                elif conf >= CONF_HIGH:
                    litem.setForeground(QColor('#ffaaaa'))
                elif conf >= CONF_MEDIUM:
                    litem.setForeground(QColor('#ff8888'))
                else:
                    litem.setForeground(QColor('#ff5555'))
            # Auto-accept if threshold met after rematch
            if (name and conf >= 0.40
                    and getattr(self, '_chk_auto_accept', None)
                    and self._chk_auto_accept.isChecked()
                    and conf >= self._spin_auto_conf.value()
                    and slot not in NON_ICON_SLOTS):
                self._review_list.setCurrentRow(row)
                self._on_accept()
        except Exception as e:
            from src.setsdebug import log as _sl
            _sl.warning(f'rematch_with_slot failed: {e}')

    def _on_canvas_deselected(self):
        """Canvas click on already-selected bbox or empty area → deselect everything."""
        self._review_list.blockSignals(True)
        self._review_list.setCurrentRow(-1)
        self._review_list.blockSignals(False)
        self._set_review_buttons_enabled(False)
        self._ann_widget.clear_highlight()

    def _on_bbox_changed(self, row: int, new_bbox: tuple):
        """Shift+LMB move/resize finished — persist the new bbox and re-classify."""
        if row < 0 or row >= len(self._recognition_items): return
        ri = self._recognition_items[row]
        ri['bbox'] = new_bbox
        
        # Update underlying annotation if it exists
        ann_id = ri.get('ann_id', '')
        if ann_id and self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            for ann in self._data_mgr.get_annotations(path):
                if ann.ann_id == ann_id:
                    self._data_mgr.update_annotation(path, ann, bbox=new_bbox)
                    self._data_mgr.save()
                    break
        
        # Re-run recognition for this specific crop if not already confirmed
        if ri.get('state') != 'confirmed' and self._current_idx >= 0:
            try:
                import cv2
                from warp.warp_importer import WarpImporter
                path = self._screenshots[self._current_idx]
                img = cv2.imread(str(path))
                if img is not None:
                    stype = self._screen_types.get(path.name, 'UNKNOWN')
                    importer_type = {'SPACE_EQ': 'SPACE', 'GROUND_EQ': 'GROUND', 'TRAITS': 'SPACE_TRAITS',
                                     'BOFFS': 'BOFFS', 'SPECIALIZATIONS': 'SPEC',
                                     'SPACE_MIXED': 'SPACE', 'GROUND_MIXED': 'GROUND'}.get(stype, 'SPACE_EQ')
                    importer = WarpImporter(sets_app=self._sets, build_type=importer_type, from_trainer=True)
                    matcher = importer._get_matcher()
                    
                    x, y, w, h = new_bbox
                    # Ensure bbox is within image bounds
                    y = max(0, min(y, img.shape[0]-1))
                    x = max(0, min(x, img.shape[1]-1))
                    h = max(1, min(h, img.shape[0]-y))
                    w = max(1, min(w, img.shape[1]-x))
                    
                    crop = img[y:y+h, x:x+w]
                    if crop.size > 0:
                        # Optional: limit candidates by slot type
                        # For now, just match against full index for better flexibility in trainer
                        name, conf, thumb, _used_sess = matcher.match(crop)
                        ri['name'] = name
                        ri['conf'] = conf
                        # Refresh visual row
                        self._review_list.takeItem(row)
                        self._add_review_row(name, ri['slot'], conf, confirmed=False)
                        self._review_list.insertItem(row, self._review_list.takeItem(self._review_list.count()-1))
                        self._review_list.setCurrentRow(row)
            except Exception as e:
                log.warning(f'Re-classification failed: {e}')

        self._ann_widget.set_review_items(self._recognition_items)

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

        # Ensure slot is visible in combo (confirmed NON_ICON_SLOTS may be hidden)
        if self._current_idx >= 0:
            _stype = self._screen_types.get(
                self._screenshots[self._current_idx].name, 'UNKNOWN')
            self._refresh_slot_combo(_stype, keep_slot=slot)
        # Set slot without triggering _on_slot_changed's clear() on name_edit
        self._slot_combo.blockSignals(True)
        self._slot_combo.setCurrentText(slot)
        self._slot_combo.blockSignals(False)
        if slot not in NON_ICON_SLOTS:
            self._populate_name_completer(slot)

        # Set name fields
        self._configure_name_field(slot)
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
            self._name_edit.blockSignals(True)
            self._name_edit.setText(name)
            self._name_edit.blockSignals(False)

    # ── Auto-accept ───────────────────────────────────────────────────────────

    def _apply_auto_accept(self):
        """Auto-accept pending items with conf >= threshold.
        Called before populating review list — items are marked confirmed
        in-place so _add_review_row renders them as confirmed directly."""
        from src.setsdebug import log as _sl
        if not getattr(self, '_chk_auto_accept', None): return
        if not self._chk_auto_accept.isChecked(): return
        threshold = self._spin_auto_conf.value()
        accepted = 0
        path = self._screenshots[self._current_idx] if self._current_idx >= 0 else None
        for ri in self._recognition_items:
            if ri.get('state') != 'pending': continue
            if ri.get('slot', '') in NON_ICON_SLOTS: continue
            conf = ri.get('conf', 0.0)
            if conf < threshold: continue
            slot = ri.get('slot', '')
            name = ri.get('name', '') or ri.get('orig_name', '')
            if not slot or not name: continue
            ri['state'] = 'confirmed'
            if ri.get('bbox') and path:
                _saved = self._data_mgr.add_annotation(
                    image_path=path, bbox=ri['bbox'], slot=slot, name=name,
                    state=AnnotationState.CONFIRMED,
                    ml_conf=conf, ml_name=name,
                )
                ri['ann_id'] = _saved.ann_id
            if ri.get('crop_bgr') is not None:
                from warp.recognition.icon_matcher import SETSIconMatcher
                SETSIconMatcher.add_session_example(ri['crop_bgr'], name)
            accepted += 1
        if accepted:
            self._data_mgr.save()
            _sl.info(f'TrainerWindow: auto-accepted {accepted} items '
                     f'(conf>={threshold:.2f})')

    def _run_auto_accept(self):
        """Legacy: called after panel is drawn. Now a no-op since
        _apply_auto_accept runs before _add_review_row."""
        pass

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
            # Check for overlapping bbox with different slot (likely user error)
            if ri.get('bbox') and self._current_idx >= 0:
                path = self._screenshots[self._current_idx]
                existing = self._data_mgr.get_annotations(path)
                new_bbox = ri['bbox']
                for ann in existing:
                    if ann.state.value != 'confirmed': continue
                    if ann.slot == slot: continue  # same slot = ok
                    if ann.ann_id == ri.get('ann_id', ''): continue
                    # Ship Type and Ship Tier intentionally overlap (tier is part of type line)
                    pair = {ann.slot, slot}
                    if pair == {'Ship Type', 'Ship Tier'}: continue
                    # Check overlap
                    ox, oy, ow, oh = ann.bbox
                    nx, ny, nw, nh = new_bbox
                    ix = max(0, min(ox+ow, nx+nw) - max(ox, nx))
                    iy = max(0, min(oy+oh, ny+nh) - max(oy, ny))
                    overlap = ix * iy
                    area = min(ow*oh, nw*nh)
                    if area > 0 and overlap / area > 0.7:
                        from PySide6.QtWidgets import QMessageBox
                        ans = QMessageBox.warning(self, 'Possible duplicate',
                            f'This bbox overlaps {ann.slot!r} → {ann.name!r}\n'
                            f'Are you sure you want to confirm as {slot!r}?',
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
                        if ans != QMessageBox.StandardButton.Yes:
                            return
                        break
            prev_name = ri.get('name', '')
            ri['name'] = name
            ri['slot'] = slot
            ri['state'] = 'confirmed'
            if ri.get('bbox') and self._current_idx >= 0:
                path = self._screenshots[self._current_idx]
                log.debug(f'accept: row={row} slot={slot!r} name={name!r} bbox={ri["bbox"]}')
                saved = self._data_mgr.add_annotation(
                    image_path=path, bbox=ri['bbox'], slot=slot, name=name,
                    state=AnnotationState.CONFIRMED,
                    ml_conf=ri.get('conf', 0.0),
                    ml_name=ri.get('orig_name', ''),
                )
                ri['ann_id'] = saved.ann_id  # track for future edits on this bbox
                self._ann_widget.refresh_annotations(path)
            litem = self._review_list.item(row)
            if litem:
                litem.setText(f'{slot}  ->  {name or "—"}  [confirmed]')
                litem.setForeground(QColor('#7effc8'))
            if name and ri.get('crop_bgr') is not None and slot not in NON_ICON_SLOTS:
                from warp.recognition.icon_matcher import SETSIconMatcher
                SETSIconMatcher.add_session_example(ri['crop_bgr'], name)
                self._contribute(ri, name)
            elif name and slot in NON_ICON_SLOTS:
                ocr_raw = ri.get('ocr_raw', '')
                # Only log OCR typo if the user actually chose a name different from what OCR settled on
                if ocr_raw and prev_name != name:
                    from src.setsdebug import log as _slog
                    _slog.info(f"Learned new OCR typo correction: {ocr_raw!r} -> {name!r}")
                    from pathlib import Path
                    import json
                    typo_file = Path('warp') / 'training_data' / 'ocr_typos.json'
                    try:
                        typos = {}
                        if typo_file.exists():
                            with open(typo_file, 'r') as f:
                                typos = json.load(f)
                        if typos.get(ocr_raw) != name:
                            typos[ocr_raw] = name
                            with open(typo_file, 'w') as f:
                                json.dump(typos, f, indent=2)
                    except Exception as e:
                        from src.setsdebug import log as _slog2
                        _slog2.warning(f"Failed to save ocr_typos: {e}")
        else:
            self._ann_widget.confirm_current(slot=slot, name=name)
        # Keep name_edit showing the accepted value — don't clear after accept
        self._update_progress()
        self._advance_to_next_unconfirmed(row)
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        self._ann_widget.clear_pending()
        self._ann_widget.set_review_items(self._recognition_items)
        self._data_mgr.save()
        self._auto_sync()
        # Update learned layout for this screenshot after each confirm
        if self._current_idx >= 0:
            self._learn_layout_for(self._screenshots[self._current_idx])
        # Refresh slot combo: hide confirmed NON_ICON_SLOTS except the one
        # currently displayed (after advance, so keep_slot reflects new row)
        if self._current_idx >= 0:
            _stype = self._screen_types.get(self._screenshots[self._current_idx].name, 'UNKNOWN')
            _cur = self._review_list.currentRow()
            _keep = (self._recognition_items[_cur]['slot']
                     if 0 <= _cur < len(self._recognition_items) else '')
            self._refresh_slot_combo(_stype, keep_slot=_keep)
        # Deferred focus — after all signals settle, return focus to list
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._review_list.setFocus)

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
            # trait_type:  'personal' | 'rep' | 'active_rep'  (migrated by datafunctions.py at load)
            if hasattr(self._sets.cache, 'starship_traits') and item_name in self._sets.cache.starship_traits:
                return _allowed('Starship Traits')
            if hasattr(self._sets.cache, 'traits'):
                t = self._sets.cache.traits
                trait_slot_map = [
                    ('space',  'rep',        'Space Reputation'),
                    ('space',  'active_rep', 'Active Space Rep'),
                    ('ground', 'rep',        'Ground Reputation'),
                    ('ground', 'active_rep', 'Active Ground Rep'),
                    ('space',  'personal',   'Personal Space Traits'),
                    ('ground', 'personal',   'Personal Ground Traits'),
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
                candidates.extend(self._sets.cache.traits['space']['active_rep'].keys())
            except Exception:
                pass
        elif slot == 'Space Reputation':
            try:
                candidates.extend(self._sets.cache.traits['space']['rep'].keys())
            except Exception:
                pass
        elif slot == 'Active Ground Rep':
            try:
                candidates.extend(self._sets.cache.traits['ground']['active_rep'].keys())
            except Exception:
                pass
        elif slot == 'Ground Reputation':
            try:
                candidates.extend(self._sets.cache.traits['ground']['rep'].keys())
            except Exception:
                pass
        elif slot == 'Personal Space Traits':
            try:
                candidates.extend(self._sets.cache.traits['space']['personal'].keys())
            except Exception:
                pass
        elif slot == 'Personal Ground Traits':
            try:
                candidates.extend(self._sets.cache.traits['ground']['personal'].keys())
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

    def _configure_name_field(self, slot: str) -> None:
        """Single point of truth for name-input widget state.

        Controls visibility, editability, label and placeholder based on slot.
        Call from both _on_slot_changed and _on_item_selected so the rules
        are never duplicated.
        """
        is_tier      = (slot == 'Ship Tier')
        is_ship_type = (slot == 'Ship Type')
        is_non_icon  = slot in NON_ICON_SLOTS  # Ship Name / Type / Tier

        self._tier_combo.setVisible(is_tier)
        self._ship_type_combo.setVisible(is_ship_type)
        self._name_edit.setVisible(not is_tier and not is_ship_type)
        self._name_edit.setEnabled(not is_non_icon)

        if is_tier:
            self._name_label.setText('Tier:')
        elif is_ship_type:
            self._name_label.setText('Ship Type:')
        elif is_non_icon:
            self._name_label.setText('Ship Name:')
            self._name_edit.setPlaceholderText('Position only — OCR reads this automatically')
        else:
            self._name_label.setText('Item name:')
            self._name_edit.setPlaceholderText("Item name (or leave blank for 'Unknown')")

    def _on_slot_changed(self, slot: str):
        self._configure_name_field(slot)
        if slot == 'Ship Type':
            self._populate_ship_type_combo()
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
        # Re-run icon matching or OCR with new slot's candidates (user-initiated change only)
        if not self._loading_row:
            row = self._review_list.currentRow()
            if 0 <= row < len(self._recognition_items):
                ri = self._recognition_items[row]
                crop_bgr = ri.get('crop_bgr')
                
                # If bbox was loaded from saved annotations, crop_bgr is initially None. Fetch it now.
                if crop_bgr is None and self._current_idx >= 0:
                    import cv2
                    img = cv2.imread(str(self._screenshots[self._current_idx]))
                    if img is not None:
                        bbox = ri.get('bbox')
                        if bbox:
                            x, y, w, h = bbox
                            crop_bgr = img[y:y+h, x:x+w].copy()
                            ri['crop_bgr'] = crop_bgr
                            
                if crop_bgr is not None:
                    if ri.get('state') == 'confirmed':
                        ri['state'] = 'pending'
                    self._rematch_with_slot(row, slot, crop_bgr)

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
        # Selection from dropdown = immediate confirm, no need to click Accept
        self._on_accept()
        self._review_list.setFocus()

    def _auto_sync(self):
        """Mark that confirmed data changed — actual upload happens on next timer tick."""
        self._pending_sync = True

    def _on_sync_timer(self):
        """Called every 5 minutes — uploads crops to HF and checks for a newer central model."""
        # Refresh community knowledge (pHash overrides)
        if self._sync_client:
            try:
                self._sync_client.refresh_knowledge()
            except Exception as e:
                log.debug(f'WARP CORE: knowledge refresh error: {e}')

        # Check for newer central model (rate-limited to once per 15 min internally)
        try:
            from warp.trainer.model_updater import ModelUpdater
            ModelUpdater().check_and_update(self._sets_root)
        except Exception as e:
            log.debug(f'WARP CORE: model update check error: {e}')

        if not self._pending_sync:
            return
        try:
            token = (Path(__file__).parent.parent / 'hub_token.txt').read_text().strip()
            if not token or token == 'YOUR_HF_TOKEN_HERE':
                return
            if self._sync_worker and self._sync_worker.isRunning():
                log.debug('HF Sync: previous upload still running, will retry next tick')
                return
            log.info('HF Sync: timer tick — uploading pending changes to HuggingFace…')
            self._pending_sync = False
            self._sync_worker = SyncWorker(data_manager=self._data_mgr, hf_token=token, mode='upload')
            self._sync_worker.progress.connect(lambda pct, msg: log.debug(f'HF Sync [{pct}%]: {msg}'))
            self._sync_worker.finished.connect(self._on_sync_finished)
            self._sync_worker.start()
        except Exception as e:
            log.warning(f'HF Sync: init error: {e}')
            self._pending_sync = True  # restore flag so next tick retries

    def _on_sync_finished(self, ok: bool):
        msg = 'Synced.' if ok else 'Sync failed.'
        self.statusBar().showMessage(msg)
        log.info(f'HF Sync: finished — {"OK" if ok else "ERROR"}')

    # Screen type → importer build_type mapping (same as RecognitionWorker)
    _STYPE_TO_BUILD: dict[str, str] = {
        'SPACE_EQ':        'SPACE',
        'GROUND_EQ':       'GROUND',
        'TRAITS':          'SPACE_TRAITS',
        'BOFFS':           'BOFFS',
        'SPECIALIZATIONS': 'SPEC',
        'SPACE_MIXED':     'SPACE',
        'GROUND_MIXED':    'GROUND',
    }

    def _learn_layout_for(self, path: Path) -> bool:
        """Save confirmed layout for one screenshot to anchors.json. Returns True if saved."""
        try:
            anns = self._data_mgr.get_annotations(path)
            confirmed = [{'bbox': a.bbox, 'slot': a.slot} for a in anns
                         if a.state == AnnotationState.CONFIRMED]
            if not confirmed:
                return False
            stype = self._screen_types.get(path.name, 'UNKNOWN')
            build_type = self._STYPE_TO_BUILD.get(stype)
            if not build_type:
                log.debug(f'Layout learn: {path.name} — stype={stype!r} unknown, skipping')
                return False
            import cv2
            from warp.recognition.layout_detector import LayoutDetector
            img = cv2.imread(str(path))
            if img is None:
                return False
            LayoutDetector().learn_layout(build_type, img.shape[:2], confirmed)
            log.info(f'Layout learn: {path.name} [{build_type}] — {len(confirmed)} slots saved to anchors.json')
            return True
        except Exception as e:
            log.warning(f'Layout learn: error for {path.name}: {e}')
            return False

    def _learn_all_layouts(self):
        """Save confirmed layouts for all screenshots to anchors.json."""
        import cv2
        from warp.recognition.layout_detector import LayoutDetector
        detector = LayoutDetector()
        learned_count = 0
        skipped_unknown = 0
        for path in self._screenshots:
            anns = self._data_mgr.get_annotations(path)
            confirmed = [{'bbox': a.bbox, 'slot': a.slot} for a in anns
                         if a.state == AnnotationState.CONFIRMED]
            if not confirmed:
                continue
            stype = self._screen_types.get(path.name, 'UNKNOWN')
            build_type = self._STYPE_TO_BUILD.get(stype)
            if not build_type:
                skipped_unknown += 1
                log.debug(f'Layout learn: {path.name} — stype={stype!r} unknown, skipping')
                continue
            img = cv2.imread(str(path))
            if img is not None:
                detector.learn_layout(build_type, img.shape[:2], confirmed)
                log.info(f'Layout learn: {path.name} [{build_type}] — {len(confirmed)} slots')
                learned_count += 1
        log.info(f'Layout learning: saved {learned_count} layouts'
                 + (f', skipped {skipped_unknown} (unknown screen type)' if skipped_unknown else '')
                 + ' → anchors.json')

    def _update_progress(self):
        total = len(self._screenshots)
        annotated = sum(1 for p in self._screenshots if self._data_mgr.has_annotations(p))
        confirmed_types = self._data_mgr.get_screen_type_counts()
        confirmed_total = sum(confirmed_types.values())
        # Build compact per-type summary for status bar
        if confirmed_types:
            type_summary = '  |  Screen types: ' + '  '.join(
                f'{SCREEN_TYPE_ICONS.get(k,"?")}{v}' for k, v in sorted(confirmed_types.items()))
        else:
            type_summary = '  |  No confirmed screen types yet'
        self._prog_lbl.setText(
            f'{annotated}/{total} annotated  ·  {confirmed_total} confirmed screen types{type_summary}')
        self._prog_bar.setValue(int(100 * annotated / max(1, total)))
        self._file_list.blockSignals(True)
        for row, p in enumerate(self._screenshots):
            item = self._file_list.item(row)
            if item:
                stype = self._screen_types.get(p.name, 'UNKNOWN')
                icon = SCREEN_TYPE_ICONS.get(stype, '?')
                label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                confirmed = p.name in self._screen_types_manual
                item.setText(f'{icon} {label}\n  {p.name}')
                item.setCheckState(
                    Qt.CheckState.Checked if confirmed else Qt.CheckState.Unchecked)
                if self._data_mgr.has_annotations(p):
                    item.setForeground(QColor('#7effc8'))
                else:
                    item.setForeground(Qt.GlobalColor.white)
        self._file_list.blockSignals(False)

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(8):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')

    def closeEvent(self, event):
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().removeEventFilter(self)
        except Exception:
            pass
        if hasattr(self, '_sync_timer') and self._sync_timer:
            self._sync_timer.stop()

        # Stop sync worker if it's currently running to prevent "QThread destroyed" abort
        if hasattr(self, '_sync_worker') and self._sync_worker and self._sync_worker.isRunning():
            self._sync_worker.requestInterruption()
            self._sync_worker.wait(1000)

        # Stop active OCR workers
        if hasattr(self, '_ocr_workers'):
            for w in self._ocr_workers:
                if w.isRunning():
                    w.requestInterruption()
                    w.wait(500)

        if hasattr(self, '_data_mgr') and self._data_mgr:
            self._data_mgr.save()
        event.accept()
