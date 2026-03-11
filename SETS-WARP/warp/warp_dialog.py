# warp/warp_dialog.py
# Multi-step WARP import dialog.
# Fully integrated with SETS cache, build data model, and UI patterns.
#
# Uses PySide6 (same as SETS).
# Accesses sets_app.cache.equipment, sets_app.cache.traits,
#           sets_app.cache.starship_traits, sets_app.cache.images
# Writes to  sets_app.build   via slot_equipment_item / slot_trait_item
#            sets_app.widgets via the same functions
# Then calls sets_app.autosave()

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QGroupBox, QFileDialog,
    QProgressBar, QStackedWidget, QWidget, QScrollArea,
    QMessageBox, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QFont, QPixmap, QImage

from warp.warp_importer import WarpImporter, ImportResult, RecognisedItem

log = logging.getLogger(__name__)

# ── SETS build key mappings ────────────────────────────────────────────────────
# Maps WARP slot names → (SETS build_key, environment, is_equipment)
# Derived from SETS empty_build() and EQUIPMENT_TYPES in constants.py
SLOT_MAP = {
    # SPACE equipment
    'Fore Weapons':          ('fore_weapons',   'space', True),
    'Aft Weapons':           ('aft_weapons',    'space', True),
    'Experimental Weapon':   ('experimental',   'space', True),
    'Devices':               ('devices',        'space', True),
    'Hangars':               ('hangars',        'space', True),
    'Deflector':             ('deflector',      'space', True),
    'Sec-Def':               ('sec_def',        'space', True),
    'Engines':               ('engines',        'space', True),
    'Warp Core':             ('core',           'space', True),
    'Shield':                ('shield',         'space', True),
    'Universal Consoles':    ('uni_consoles',   'space', True),
    'Engineering Consoles':  ('eng_consoles',   'space', True),
    'Science Consoles':      ('sci_consoles',   'space', True),
    'Tactical Consoles':     ('tac_consoles',   'space', True),
    # SPACE traits
    'Personal Space Traits': ('traits',         'space', False),
    'Starship Traits':       ('starship_traits','space', False),
    'Reputation Traits':     ('rep_traits',     'space', False),
    'Active Rep Traits':     ('active_rep_traits','space',False),
    # GROUND equipment
    'Body Armor':            ('armor',          'ground', True),
    'EV Suit':               ('ev_suit',        'ground', True),
    'Personal Shield':       ('personal_shield','ground', True),
    'Weapons':               ('weapons',        'ground', True),
    'Kit':                   ('kit',            'ground', True),
    'Kit Modules':           ('kit_modules',    'ground', True),
    'Ground Devices':        ('ground_devices', 'ground', True),
    # GROUND traits
    'Personal Ground Traits':('traits',         'ground', False),
    'Ground Rep Traits':     ('rep_traits',     'ground', False),
}

BUILD_TYPES = [
    ('SPACE',         'Space Build',
     'Equipment, consoles, boffs and traits for a space loadout'),
    ('GROUND',        'Ground Build',
     'Equipment, boffs and traits for a ground loadout'),
    ('SPACE_SKILLS',  'Space Skills',
     'Space skill tree point allocation'),
    ('GROUND_SKILLS', 'Ground Skills',
     'Ground skill tree point allocation'),
]

_SETTINGS_KEY_LAST_DIR = 'warp/last_import_dir'

SCREENSHOT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


# ── Background worker ──────────────────────────────────────────────────────────

class _ImportWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)   # ImportResult
    error    = Signal(str)

    def __init__(self, folder: Path, build_type: str, sets_app):
        super().__init__()
        self._folder     = folder
        self._build_type = build_type
        self._sets_app   = sets_app

    def run(self):
        try:
            importer = WarpImporter(
                build_type=self._build_type,
                sets_app=self._sets_app,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            result = importer.process_folder(self._folder)
            self.finished.emit(result)
        except Exception as exc:
            log.exception('WARP import worker error')
            self.error.emit(str(exc))


# ── Main dialog ────────────────────────────────────────────────────────────────

class WarpDialog(QDialog):
    """
    3-page import dialog:
      Page 0 — build type + folder selection
      Page 1 — progress
      Page 2 — review / per-item accept/reject
    """

    def __init__(self, sets_app, parent=None):
        super().__init__(parent)
        self._sets = sets_app
        self._settings = QSettings()
        self._folder: Path | None = None
        self._build_type = 'SPACE'
        self._worker: _ImportWorker | None = None
        self._import_result: ImportResult | None = None
        self._review_rows: list[_ReviewRow] = []

        self.setWindowTitle('WARP — Screenshot Import')
        self.setMinimumWidth(580)
        self.setMinimumHeight(420)
        self.setModal(True)
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        title = QLabel('⚡  WARP — Build Importer')
        title.setFont(QFont('', 13, QFont.Weight.Bold))
        title.setStyleSheet('color: #7ec8e3;')
        root.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color: #3a6a9c;')
        root.addWidget(sep)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        self._stack.addWidget(self._make_page_select())
        self._stack.addWidget(self._make_page_progress())
        self._stack.addWidget(self._make_page_review())

        # Footer
        footer = QHBoxLayout()
        self._btn_back   = QPushButton('← Back')
        self._btn_next   = QPushButton('Analyse →')
        self._btn_accept = QPushButton('✓  Import Build')
        self._btn_cancel = QPushButton('Cancel')
        self._btn_back.setVisible(False)
        self._btn_accept.setVisible(False)
        self._btn_accept.setStyleSheet(
            'QPushButton {background:#1a5c3a;color:#7effc8;'
            'border:1px solid #3aac6a;border-radius:3px;padding:5px 14px;font-weight:bold;}'
            'QPushButton:hover{background:#2a8c5a;}')
        footer.addWidget(self._btn_back)
        footer.addStretch()
        footer.addWidget(self._btn_cancel)
        footer.addWidget(self._btn_next)
        footer.addWidget(self._btn_accept)
        root.addLayout(footer)

        self._btn_cancel.clicked.connect(self.reject)
        self._btn_next.clicked.connect(self._on_next)
        self._btn_back.clicked.connect(self._on_back)
        self._btn_accept.clicked.connect(self._on_accept)

    def _make_page_select(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setSpacing(12)

        # Build type
        g1 = QGroupBox('1.  Select Build Type')
        g1_lay = QVBoxLayout(g1)
        self._type_bg = QButtonGroup(self)
        for i, (key, label, tip) in enumerate(BUILD_TYPES):
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            rb.setProperty('build_key', key)
            if i == 0:
                rb.setChecked(True)
            g1_lay.addWidget(rb)
            self._type_bg.addButton(rb, i)
        lay.addWidget(g1)

        # Folder
        g2 = QGroupBox('2.  Select Screenshots Folder')
        g2_lay = QVBoxLayout(g2)
        notice = QLabel(
            '⚠  The folder should contain screenshots of ONE build only.\n'
            '   Multiple builds in the same folder will produce incorrect results.')
        notice.setWordWrap(True)
        notice.setStyleSheet('color:#e8c060;font-size:11px;')
        g2_lay.addWidget(notice)

        row = QHBoxLayout()
        self._folder_label = QLabel('No folder selected')
        self._folder_label.setStyleSheet(
            'color:#aaa;background:#111;border:1px solid #333;'
            'border-radius:3px;padding:3px 7px;')
        self._folder_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        btn_browse = QPushButton('Browse…')
        btn_browse.setFixedWidth(88)
        btn_browse.clicked.connect(self._browse_folder)
        row.addWidget(self._folder_label, 1)
        row.addWidget(btn_browse)
        g2_lay.addLayout(row)
        lay.addWidget(g2)
        lay.addStretch()
        return page

    def _make_page_progress(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label = QLabel('Preparing…')
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setWordWrap(True)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        lay.addStretch()
        lay.addWidget(self._progress_label)
        lay.addSpacing(10)
        lay.addWidget(self._progress_bar)
        lay.addStretch()
        return page

    def _make_page_review(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        hdr = QLabel(
            'Review recognised items — correct any mistakes before importing.\n'
            'Green = high confidence, Yellow = uncertain, Red = low confidence.')
        hdr.setWordWrap(True)
        hdr.setStyleSheet('color:#aaa;font-size:11px;')
        lay.addWidget(hdr)

        self._review_scroll = QScrollArea()
        self._review_scroll.setWidgetResizable(True)
        self._review_inner  = QWidget()
        self._review_layout = QVBoxLayout(self._review_inner)
        self._review_layout.setSpacing(3)
        self._review_layout.addStretch()
        self._review_scroll.setWidget(self._review_inner)
        lay.addWidget(self._review_scroll, 1)

        self._review_summary = QLabel('')
        self._review_summary.setStyleSheet('color:#7effc8;font-size:11px;')
        lay.addWidget(self._review_summary)
        return page

    # ── Navigation ─────────────────────────────────────────────────────────

    def _on_next(self):
        if self._stack.currentIndex() == 0:
            if not self._folder or not self._folder.is_dir():
                QMessageBox.warning(self, 'WARP', 'Please select a valid screenshots folder.')
                return
            checked = self._type_bg.checkedButton()
            self._build_type = checked.property('build_key') if checked else 'SPACE'
            self._start_import()

    def _on_back(self):
        self._stack.setCurrentIndex(0)
        self._btn_back.setVisible(False)
        self._btn_next.setVisible(True)
        self._btn_accept.setVisible(False)
        self._btn_next.setEnabled(True)

    def _on_accept(self):
        """Apply all accepted items to SETS build."""
        if self._import_result is None:
            return
        self._apply_to_sets()
        self.accept()

    # ── Import pipeline ────────────────────────────────────────────────────

    def _start_import(self):
        self._stack.setCurrentIndex(1)
        self._btn_next.setEnabled(False)
        self._btn_back.setVisible(False)
        self._progress_bar.setValue(0)
        self._progress_label.setText('Starting…')

        self._worker = _ImportWorker(self._folder, self._build_type, self._sets)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_import_done)
        self._worker.error.connect(self._on_import_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_import_done(self, result: ImportResult):
        self._import_result = result
        self._populate_review(result)
        self._stack.setCurrentIndex(2)
        self._btn_back.setVisible(True)
        self._btn_next.setVisible(False)
        self._btn_accept.setVisible(True)
        self._btn_cancel.setEnabled(True)

        n    = len(result.items)
        low  = sum(1 for i in result.items if i.confidence < 0.70)
        self._review_summary.setText(
            f'Found {n} item(s) — {low} with low confidence (highlighted in yellow/red).\n'
            f'Ship: {result.ship_name}  {result.ship_type}  {result.ship_tier}')

    def _on_import_error(self, msg: str):
        self._stack.setCurrentIndex(0)
        self._btn_next.setEnabled(True)
        QMessageBox.critical(self, 'WARP — Analysis Error', msg)

    # ── Review population ──────────────────────────────────────────────────

    def _populate_review(self, result: ImportResult):
        # Remove old rows
        while self._review_layout.count() > 1:
            item = self._review_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._review_rows = []

        for ri in result.items:
            # Verify item name exists in SETS cache before showing
            exists = self._item_in_cache(ri)
            row = _ReviewRow(ri, item_exists=exists, parent=self._review_inner)
            self._review_layout.insertWidget(self._review_layout.count() - 1, row)
            self._review_rows.append(row)

    def _item_in_cache(self, ri: RecognisedItem) -> bool:
        """Check whether recognised item name exists in SETS equipment/trait caches."""
        if not ri.name:
            return False
        slot_info = SLOT_MAP.get(ri.slot)
        if not slot_info:
            return True   # unknown slot — assume OK, let SETS validate
        build_key, env, is_equipment = slot_info
        if is_equipment:
            return ri.name in self._sets.cache.equipment.get(build_key, {})
        elif build_key == 'starship_traits':
            return ri.name in self._sets.cache.starship_traits
        else:
            return ri.name in self._sets.cache.traits.get(env, {}).get(build_key, {})

    # ── Apply to SETS ──────────────────────────────────────────────────────

    def _apply_to_sets(self):
        """
        Iterates accepted review rows and writes each item into the SETS build
        using the same functions the existing item picker uses.
        """
        from src.buildupdater import slot_equipment_item, slot_trait_item

        # Set ship info if recognised
        r = self._import_result
        if r.ship_name and not self._sets.build['space']['ship_name']:
            self._sets.build['space']['ship_name'] = r.ship_name
            self._sets.widgets.ship['name'].setText(r.ship_name)

        accepted = [row for row in self._review_rows if row.is_accepted()]

        for row in accepted:
            ri    = row.get_item()
            slot_info = SLOT_MAP.get(ri.slot)
            if not slot_info:
                log.warning(f'WARP: Unknown slot "{ri.slot}" — skipping')
                continue

            build_key, env, is_equipment = slot_info
            idx  = ri.slot_index

            if is_equipment:
                item_data = self._make_equipment_item(ri, build_key, env)
                if item_data:
                    slot_equipment_item(self._sets, item_data, env, build_key, idx)
            else:
                item_data = {'item': ri.name}
                slot_trait_item(self._sets, item_data, env, build_key, idx)

        # Switch to the correct build tab
        tab_map = {
            'SPACE':         0,
            'GROUND':        1,
            'SPACE_SKILLS':  2,
            'GROUND_SKILLS': 3,
        }
        tab_idx = tab_map.get(self._build_type, 0)
        self._sets.switch_main_tab(tab_idx)
        self._sets.autosave()

    def _make_equipment_item(
        self, ri: RecognisedItem, build_key: str, env: str
    ) -> dict | None:
        """
        Build an equipment item dict compatible with slot_equipment_item().
        Format: {'item': name, 'rarity': rarity, 'modifiers': [...]}
        Uses SETS default_rarity and default_mark from settings.
        """
        eq_cache = self._sets.cache.equipment.get(build_key, {})
        entry    = eq_cache.get(ri.name)
        if not entry:
            log.warning(f'WARP: Item "{ri.name}" not in cache[{build_key}] — skipping')
            return None

        default_rarity = self._sets.settings.value('default_rarity', 'Epic')
        default_mark   = self._sets.settings.value('default_mark', 'XV')

        rarity = entry.get('rarity') or default_rarity
        return {
            'item':      ri.name,
            'rarity':    rarity,
            'modifiers': [default_mark] + [None] * 4,
        }

    # ── Folder picker ──────────────────────────────────────────────────────

    def _browse_folder(self):
        last = self._settings.value(_SETTINGS_KEY_LAST_DIR, '')
        dlg  = QFileDialog(self)
        dlg.setWindowTitle('Select Screenshots Folder')
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        if last and Path(last).is_dir():
            dlg.setDirectory(last)

        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                self._folder = Path(files[0])
                self._folder_label.setText(str(self._folder))
                self._folder_label.setStyleSheet(
                    'color:#eee;background:#111;border:1px solid #3a6a9c;'
                    'border-radius:3px;padding:3px 7px;')
                self._settings.setValue(_SETTINGS_KEY_LAST_DIR, str(self._folder))


# ── Review row widget ──────────────────────────────────────────────────────────

class _ReviewRow(QWidget):
    """One item in the review list."""

    def __init__(self, item: RecognisedItem, item_exists: bool = True, parent=None):
        super().__init__(parent)
        self._item    = item
        self._accepted = True
        self._setup_ui(item_exists)

    def _setup_ui(self, item_exists: bool):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 2, 3, 2)
        lay.setSpacing(7)

        conf = self._item.confidence
        if conf >= 0.85 and item_exists:
            badge = f'color:#7effc8;background:#1a3a2a;border:1px solid #3aac6a;'
        elif conf >= 0.70 and item_exists:
            badge = f'color:#e8c060;background:#3a2a0a;border:1px solid #9c7a20;'
        else:
            badge = f'color:#ff7e7e;background:#3a1a1a;border:1px solid #ac3a3a;'

        # Icon thumbnail
        icon_lbl = QLabel()
        icon_lbl.setFixedSize(40, 40)
        icon_lbl.setStyleSheet('background:#222;border:1px solid #444;')
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self._item.thumbnail is not None:
            try:
                # thumbnail is a QImage from warp_importer
                pix = QPixmap.fromImage(self._item.thumbnail).scaled(
                    38, 38, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                icon_lbl.setPixmap(pix)
            except Exception:
                icon_lbl.setText('?')
        else:
            icon_lbl.setText('?')

        # Slot label
        slot_lbl = QLabel(f'{self._item.slot}\n[{self._item.slot_index}]')
        slot_lbl.setFixedWidth(130)
        slot_lbl.setStyleSheet('color:#7ec8e3;font-size:10px;')
        slot_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Item name
        self._name_lbl = QLabel(self._item.name or '— unknown —')
        if not item_exists:
            self._name_lbl.setStyleSheet('color:#ff9999;')
        else:
            self._name_lbl.setStyleSheet('color:#eee;')
        self._name_lbl.setWordWrap(True)
        self._name_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Confidence badge
        conf_lbl = QLabel(f'{conf:.0%}')
        conf_lbl.setFixedWidth(42)
        conf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_lbl.setStyleSheet(
            badge + 'border-radius:3px;padding:1px 3px;font-size:11px;')

        # Accept toggle
        self._btn_acc = QPushButton('✓')
        self._btn_acc.setFixedSize(26, 26)
        self._btn_acc.setCheckable(True)
        self._btn_acc.setChecked(True)
        self._btn_acc.setStyleSheet(
            'QPushButton:checked{background:#1a5c3a;color:#7effc8;'
            'border:1px solid #3aac6a;border-radius:3px;}'
            'QPushButton:!checked{background:#3a1a1a;color:#888;'
            'border:1px solid #5a2a2a;border-radius:3px;}')
        self._btn_acc.toggled.connect(self._toggle)

        lay.addWidget(icon_lbl)
        lay.addWidget(slot_lbl)
        lay.addWidget(self._name_lbl, 1)
        lay.addWidget(conf_lbl)
        lay.addWidget(self._btn_acc)

        self._update_bg()

    def _toggle(self, checked: bool):
        self._accepted = checked
        self._btn_acc.setText('✓' if checked else '✗')
        self._update_bg()

    def _update_bg(self):
        self.setStyleSheet(
            'background:#1a2530;' if self._accepted else 'background:#2a1a1a;')
        self.setAutoFillBackground(True)

    def is_accepted(self) -> bool:
        return self._accepted

    def get_item(self) -> RecognisedItem:
        return self._item
