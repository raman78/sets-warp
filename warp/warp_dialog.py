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
from warp.trainer.training_data import VIRTUAL_ITEM_NAMES
from warp.style import (
    apply_dark_style, primary_btn_style, secondary_btn_style,
    ACCENT, FG, MFG, BG, MBG, LBG, BC, C_WARNING,
)

log = logging.getLogger(__name__)
try:
    from src.setsdebug import log as _slog
except Exception:
    _slog = log

# ── SETS build key mappings ────────────────────────────────────────────────────
# Maps WARP slot names → (SETS build_key, environment, is_equipment)
# Derived from SETS empty_build() and EQUIPMENT_TYPES in constants.py
SLOT_MAP = {
    # SPACE equipment
    'Fore Weapons':          ('fore_weapons',   'space', True),
    'Aft Weapons':           ('aft_weapons',    'space', True),
    'Experimental Weapon':   ('experimental',   'space', True),
    'Experimental':          ('experimental',   'space', True),  # alias used by WARP CORE annotations
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
    'Personal Space Traits': ('traits',          'space', False),
    'Starship Traits':       ('starship_traits', 'space', False),
    'Space Reputation':      ('rep_traits',      'space', False),  # name from SPACE_TRAITS_SLOT_ORDER
    'Reputation Traits':     ('rep_traits',      'space', False),  # legacy alias
    'Active Space Rep':      ('active_rep_traits','space', False),  # name from SPACE_TRAITS_SLOT_ORDER
    'Active Rep Traits':     ('active_rep_traits','space', False),  # legacy alias
    # GROUND equipment
    'Body Armor':            ('armor',          'ground', True),
    'EV Suit':               ('ev_suit',        'ground', True),
    'Personal Shield':       ('personal_shield','ground', True),
    'Weapons':               ('weapons',        'ground', True),
    'Kit':                   ('kit',            'ground', True),
    'Kit Modules':           ('kit_modules',    'ground', True),
    'Ground Devices':        ('ground_devices', 'ground', True),
    # GROUND traits
    'Personal Ground Traits': ('traits',          'ground', False),
    'Ground Reputation':      ('rep_traits',      'ground', False),  # name from GROUND_TRAITS_SLOT_ORDER
    'Ground Rep Traits':      ('rep_traits',      'ground', False),  # legacy alias
    'Active Ground Rep':      ('active_rep_traits','ground', False),  # name from GROUND_TRAITS_SLOT_ORDER
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

# ── Configuration Constants ───────────────────────────────────────────────────
_BOFF_Y_THRESHOLD_PX = 30
_BOFF_X_THRESHOLD_PX = 50
_FUZZY_MATCH_CUTOFF = 0.68
_MIN_CONFIDENCE_PROFESSION = 0.40


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
            importer.set_interrupt_check(self.isInterruptionRequested)
            result = importer.process_folder(self._folder)
            if not self.isInterruptionRequested():
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
        last = self._settings.value(_SETTINGS_KEY_LAST_DIR, '')
        self._folder: Path | None = Path(last) if last and Path(last).is_dir() else None
        self._build_type = 'SPACE'
        self._worker: _ImportWorker | None = None
        self._import_result: ImportResult | None = None

        self.setWindowTitle('WARP — Screenshot Import')
        self.setMinimumWidth(580)
        self.setMinimumHeight(420)
        self.setModal(True)
        apply_dark_style(self)
        self._build_ui()
        if self._folder:
            self._folder_label.setText(str(self._folder))
            self._folder_label.setStyleSheet(
                f'color:{FG};background:{MBG};border:1px solid {ACCENT};'
                f'border-radius:3px;padding:3px 7px;')

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        title = QLabel('⚡  WARP — Build Importer')
        title.setFont(QFont('', 13, QFont.Weight.Bold))
        title.setStyleSheet(f'color:{ACCENT};')
        root.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color:{BC};')
        root.addWidget(sep)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        self._stack.addWidget(self._make_page_select())
        self._stack.addWidget(self._make_page_progress())

        # Footer
        footer = QHBoxLayout()
        self._btn_next   = QPushButton('Analyse & Import →')
        self._btn_cancel = QPushButton('Cancel')
        self._btn_next.setStyleSheet(primary_btn_style())
        footer.addStretch()
        footer.addWidget(self._btn_cancel)
        footer.addWidget(self._btn_next)
        root.addLayout(footer)

        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_next.clicked.connect(self._on_next)

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
        notice.setStyleSheet(f'color:{C_WARNING};font-size:11px;')
        g2_lay.addWidget(notice)

        row = QHBoxLayout()
        self._folder_label = QLabel('No folder selected')
        self._folder_label.setStyleSheet(
            f'color:{MFG};background:{BG};border:1px solid {BC};'
            f'border-radius:3px;padding:3px 7px;')
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
        from warp.ui_helpers import time_spent_counter
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label = QLabel('Preparing…')
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setWordWrap(True)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._time_lbl, self._clock = time_spent_counter(page)
        time_row = QHBoxLayout()
        time_row.addWidget(self._time_lbl)
        time_row.addStretch()
        lay.addStretch()
        lay.addWidget(self._progress_label)
        lay.addSpacing(10)
        lay.addWidget(self._progress_bar)
        lay.addSpacing(4)
        lay.addLayout(time_row)
        lay.addStretch()
        return page


    def _on_next(self):
        if self._stack.currentIndex() == 0:
            if not self._folder or not self._folder.is_dir():
                QMessageBox.warning(self, 'WARP', 'Please select a valid screenshots folder.')
                return
            checked = self._type_bg.checkedButton()
            self._build_type = checked.property('build_key') if checked else 'SPACE'
            self._start_import()

    # ── Import pipeline ────────────────────────────────────────────────────

    def _start_import(self):
        self._stack.setCurrentIndex(1)
        self._btn_next.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_label.setText('Starting…')
        self._clock.start(1000)

        self._worker = _ImportWorker(self._folder, self._build_type, self._sets)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_import_done)
        self._worker.error.connect(self._on_import_error)
        self._worker.start()

    def _on_cancel(self):
        self._clock.stop()
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
        self.reject()

    def _on_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_import_done(self, result: ImportResult):
        self._clock.stop()
        self._import_result = result
        self._apply_to_sets()
        n_detected   = len(result.items)
        n_imported   = sum(1 for i in result.items if i.name)
        n_unmatched  = n_detected - n_imported
        ship_info    = result.ship_type or 'Unknown ship'
        msg = (
            f"Detected {n_detected} item slot(s)\n"
            f"Identified and imported: {n_imported}\n"
            f"Unmatched (skipped): {n_unmatched}\n\n"
            f"Ship: {ship_info} {result.ship_tier}"
        )
        QMessageBox.information(self, "WARP — Import Complete", msg)
        self.accept()

    def _on_import_error(self, msg: str):
        self._clock.stop()
        self._stack.setCurrentIndex(0)
        self._btn_next.setEnabled(True)
        QMessageBox.critical(self, 'WARP — Analysis Error', msg)


    # ── Apply to SETS ──────────────────────────────────────────────────────

    def _apply_to_sets(self):
        """
        Writes all recognised items directly into the SETS build (no review step).
        Only items with a non-empty name are imported.
        """
        r = self._import_result
        if r is None:
            return

        ship_data = self._resolve_and_apply_ship(r.ship_type, r.ship_tier)

        boff_items, overflow_consoles, written_indices = self._import_equipment_and_traits(r.items)

        if overflow_consoles:
            self._handle_console_overflow(overflow_consoles, written_indices)

        if boff_items:
            is_ground = (getattr(r, 'build_type', '') == 'GROUND_BOFFS')
            self._write_boffs_to_build(boff_items, ship_data, is_ground=is_ground)

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

    def _resolve_and_apply_ship(self, ship_type: str | None, ship_tier: str | None) -> dict | None:
        """
        Matches ship type against SETS cache and applies it to the UI and build.
        Returns the matched ship_data dict if successful, otherwise None.
        """
        _slog.info(f'WARP: resolving ship_type={ship_type!r} ship_tier={ship_tier!r}')
        try:
            from src.buildupdater import clear_ship
            widgets = self._sets.widgets
            
            if not ship_type:
                _slog.info('WARP: no ship type detected — clearing ship selection')
                clear_ship(self._sets)
                return None

            from difflib import get_close_matches
            ships = getattr(self._sets.cache, 'ships', {})
            candidates = list(ships.keys())
            
            match = None
            if ship_type in ships:
                match = ship_type
            else:
                ocr_words = set(ship_type.lower().split())
                subset_hits = [c for c in candidates if ocr_words.issubset(set(c.lower().split()))]
                if len(subset_hits) == 1:
                    match = subset_hits[0]
                elif len(subset_hits) > 1:
                    match = min(subset_hits, key=lambda c: len(set(c.lower().split()) - ocr_words))
                else:
                    hits = get_close_matches(ship_type, candidates, n=1, cutoff=_FUZZY_MATCH_CUTOFF)
                    if hits:
                        match = hits[0]

            if not match:
                _slog.info(f'WARP: ship {ship_type!r} not found in cache — clearing ship')
                clear_ship(self._sets)
                return None

            _slog.info(f'WARP: auto-selecting ship {match!r} from {ship_type!r}')
            try:
                from src.callbacks import _save_session_slots, _restore_session_slots, align_space_frame
                from src.widgets import exec_in_thread
                sets = self._sets
                ship_data = ships[match]
                
                sets.building = True
                widgets.ship['button'].setText(match)
                
                image_filename = ship_data['image'][5:]
                def _on_ship_image(img, _w=widgets):
                    _w.ship['image'].set_image(img[0])
                exec_in_thread(sets, sets.images.get_ship_image, image_filename, result=_on_ship_image)
                
                tier = ship_data['tier']
                widgets.ship['tier'].clear()
                if tier == 6:
                    widgets.ship['tier'].addItems(('T6', 'T6-X', 'T6-X2'))
                elif tier == 5:
                    widgets.ship['tier'].addItems(('T5', 'T5-U', 'T5-X', 'T5-X2'))
                else:
                    widgets.ship['tier'].addItem(f'T{tier}')
                    
                target_tier = ship_tier or f'T{tier}'
                idx = widgets.ship['tier'].findText(target_tier)
                if idx >= 0:
                    widgets.ship['tier'].setCurrentIndex(idx)
                    
                sets.build['space']['ship'] = match
                sets.build['space']['tier'] = target_tier
                if ship_data.get('equipcannons') == 'yes':
                    widgets.ship['dc'].show()
                else:
                    widgets.ship['dc'].hide()
                    
                _save_session_slots(sets)
                align_space_frame(sets, ship_data, clear=False)
                _restore_session_slots(sets)
                sets.building = False
                
                return ship_data
                
            except Exception as _se:
                _slog.warning(f'WARP: ship select failed: {_se}')
                return None
                
        except Exception as _e:
            _slog.warning(f'WARP: ship info widget update failed: {_e}')
            return None

    def _import_equipment_and_traits(self, items: list[RecognisedItem]) -> tuple[list, list, dict]:
        """
        Iterates over recognised items, sorting BOFF abilities and processing equipment/traits.
        Returns (boff_items, overflow_consoles, written_indices).
        """
        from src.buildupdater import slot_equipment_item, slot_trait_item
        boff_items = []
        overflow_consoles = []
        written_indices = {}

        for ri in items:
            if not ri.name or ri.name in VIRTUAL_ITEM_NAMES:
                continue

            if ri.slot.startswith('Boff '):
                boff_items.append(ri)
                continue

            slot_info = SLOT_MAP.get(ri.slot)
            if not slot_info:
                log.warning(f'WARP: Unknown slot "{ri.slot}" — skipping')
                continue

            build_key, env, is_equipment = slot_info
            idx = ri.slot_index

            try:
                if not is_equipment:
                    item_data = {'item': ri.name}
                    slot_trait_item(self._sets, item_data, env, build_key, idx)
                    continue

                item_data = self._make_equipment_item(ri, build_key, env)
                if not item_data:
                    continue

                build_list = self._sets.build[env].get(build_key, [])
                if idx >= len(build_list):
                    if build_key == 'uni_consoles':
                        overflow_consoles.append((ri, item_data, env))
                    else:
                        log.warning(f'WARP: slot {ri.slot}[{idx}] out of range — skipping "{ri.name}"')
                    continue

                slot_equipment_item(self._sets, item_data, env, build_key, idx)
                if 'console' in build_key:
                    written_indices[build_key] = max(written_indices.get(build_key, 0), idx + 1)

            except Exception as e:
                log.warning(f'WARP: Failed to import "{ri.name}" into {ri.slot}[{idx}]: {e}')

        return boff_items, overflow_consoles, written_indices

    def _handle_console_overflow(self, overflow_items: list, written_indices: dict):
        """
        Redistributes overflow uni_consoles items into available eng/sci/tac slots.
        """
        from src.buildupdater import slot_equipment_item
        overflow_items.sort(key=lambda x: x[0].bbox[1] if x[0].bbox else 0)
        
        for ri, item_data, env in overflow_items:
            placed = False
            for ck in ('eng_consoles', 'sci_consoles', 'tac_consoles'):
                next_idx = written_indices.get(ck, 0)
                bl = self._sets.build[env].get(ck, [])
                if next_idx < len(bl):
                    slot_equipment_item(self._sets, item_data, env, ck, next_idx)
                    written_indices[ck] = next_idx + 1
                    log.info(f'WARP: {ri.slot}[{ri.slot_index}] overflow → {ck}[{next_idx}] "{ri.name}"')
                    placed = True
                    break
            if not placed:
                log.warning(f'WARP: {ri.slot}[{ri.slot_index}] overflow — no console slot available for "{ri.name}"')

    def _write_boffs_to_build(self, boff_items: list, ship_data: dict | None, is_ground: bool = False) -> None:
        """
        Orchestrates writing recognized BOFF abilities to the SETS build.
        Divides the process into preparation, clustering, matching, and UI updating.
        """
        seats_visual, visual_to_seat_id, boffs_build = self._prepare_boff_seats(ship_data, is_ground)
        if not seats_visual:
            return

        seat_clusters = self._cluster_boff_items(boff_items)
        if not seat_clusters:
            return

        assigned, cluster_info = self._match_clusters_to_seats(seat_clusters, seats_visual)
        if not assigned:
            _slog.warning('WARP boff: No BOFF clusters matched to seats.')
            return

        self._write_abilities_to_ui(assigned, cluster_info, seats_visual, visual_to_seat_id, boffs_build, is_ground)

    def _prepare_boff_seats(self, ship_data: dict | None, is_ground: bool) -> tuple[list, dict, list]:
        """
        Extracts BOFF seat configuration based on build type (ground or space).
        Returns (seats_visual, visual_to_seat_id, boffs_build).
        """
        from src.buildupdater import get_boff_spec
        
        if is_ground:
            boffs_build = self._sets.build['ground']['boffs']
            _boff_profs = self._sets.build['ground'].get('boff_profs', {})
            _boff_specs = self._sets.build['ground'].get('boff_specs', {})
            n_seats = len(boffs_build)
            seats_visual = [(4, _boff_profs.get(i, 'Universal'), _boff_specs.get(i)) for i in range(n_seats)]
            visual_to_seat_id = {i: i for i in range(n_seats)}
            return seats_visual, visual_to_seat_id, boffs_build

        if not ship_data or not ship_data.get('boffs'):
            _slog.warning('WARP boff: space build requested but ship_data has no boffs field.')
            return [], {}, []
            
        try:
            seats_visual = [get_boff_spec(self._sets, s) for s in ship_data['boffs']]
        except Exception as e:
            _slog.warning(f'WARP boff: could not parse seat specs: {e}')
            return [], {}, []
            
        sorted_ix = sorted(enumerate(seats_visual), key=lambda p: p[1], reverse=True)
        visual_to_seat_id = {vis_i: seat_id for seat_id, (vis_i, _) in enumerate(sorted_ix)}
        boffs_build = self._sets.build['space']['boffs']
        return seats_visual, visual_to_seat_id, boffs_build

    def _cluster_boff_items(self, boff_items: list) -> list[list]:
        """
        Groups recognized BOFF items by Y-position, then splits by large X gaps.
        Returns a list of item clusters, one per detected seat.
        """
        items_with_bbox = [ri for ri in boff_items if ri.bbox]
        if not items_with_bbox:
            _slog.warning('WARP boff: no boff items with bbox — cannot route')
            return []

        items_sorted_y = sorted(items_with_bbox, key=lambda ri: ri.bbox[1])
        y_bands = []
        for ri in items_sorted_y:
            if not y_bands or ri.bbox[1] - y_bands[-1][-1].bbox[1] > _BOFF_Y_THRESHOLD_PX:
                y_bands.append([ri])
            else:
                y_bands[-1].append(ri)

        seat_clusters = []
        for band in y_bands:
            x_sorted = sorted(band, key=lambda ri: ri.bbox[0])
            cluster = [x_sorted[0]]
            for ri in x_sorted[1:]:
                prev_right = cluster[-1].bbox[0] + cluster[-1].bbox[2]
                if ri.bbox[0] - prev_right > _BOFF_X_THRESHOLD_PX:
                    seat_clusters.append(cluster)
                    cluster = [ri]
                else:
                    cluster.append(ri)
            seat_clusters.append(cluster)
            
        _slog.info(f'WARP boff: {len(items_with_bbox)} items → {len(y_bands)} Y-bands → {len(seat_clusters)} clusters')
        return seat_clusters

    def _match_clusters_to_seats(self, seat_clusters: list[list], seats_visual: list) -> tuple[dict, list]:
        """
        Matches BOFF item clusters to available UI seats based on profession alignment.
        Returns (assigned_dict, cluster_info_list).
        """
        from warp.recognition.boff_keys import parse_seat_profession, parse_seat_spec
        from collections import Counter
        
        _SPEC_TO_PROF = {
            'Temporal Operative': 'Temporal',
            'Command':            'Command',
            'Miracle Worker':     'Miracle Worker',
            'Intelligence':       'Intelligence',
            'Pilot':              'Pilot',
        }

        cluster_info = []
        for c in seat_clusters:
            cluster_slot = getattr(c[0], 'seat_key', '') or c[0].slot
            base_prof = parse_seat_profession(cluster_slot)
            spec_prof = parse_seat_spec(cluster_slot)
            
            content_profs = []
            for ri in c:
                if ri.confidence >= _MIN_CONFIDENCE_PROFESSION and ri.name:
                    found = False
                    for domain in ['space', 'ground']:
                        for career, ranks in self._sets.cache.boff_abilities.get(domain, {}).items():
                            for rank_dict in ranks:
                                if isinstance(rank_dict, dict) and ri.name in rank_dict:
                                    content_profs.append(career)
                                    found = True
                                    break
                            if found: break
                        if found: break
            
            prof_set = set(content_profs)
            if base_prof is None:
                base_prof = Counter(content_profs).most_common(1)[0][0] if content_profs else 'Unknown'
            cluster_info.append([c, base_prof, prof_set, spec_prof])

        unmatched = list(range(len(cluster_info)))
        assigned = {}

        def _find_cluster(primary_prof: str, target_spec: str | None) -> int | None:
            for ci in unmatched:
                c_primary, c_profs, c_spec = cluster_info[ci][1], cluster_info[ci][2], cluster_info[ci][3]
                if c_primary != primary_prof:
                    continue
                if target_spec:
                    if c_spec:
                        if c_spec != target_spec:
                            continue
                    elif target_spec not in c_profs:
                        continue
                return ci
            return None

        # Pass 1 & 2: explicit-profession seats
        for vis_i, (rank, prof, spec) in enumerate(seats_visual):
            if prof == 'Universal' or vis_i in assigned:
                continue
            spec_prof = _SPEC_TO_PROF.get(spec) if spec else None
            ci = _find_cluster(prof, spec_prof)
            if ci is None and spec_prof:
                ci = _find_cluster(spec_prof, None)
            if ci is not None:
                assigned[vis_i] = ci
                unmatched.remove(ci)

        # Pass 3: Universal seats with spec
        for vis_i, (rank, prof, spec) in enumerate(seats_visual):
            if prof != 'Universal' or not spec or vis_i in assigned:
                continue
            spec_prof = _SPEC_TO_PROF.get(spec)
            ci = _find_cluster(spec_prof, None) if spec_prof else None
            if ci is not None:
                assigned[vis_i] = ci
                unmatched.remove(ci)

        # Pass 4: Universal seats without spec
        for vis_i, (rank, prof, spec) in enumerate(seats_visual):
            if prof != 'Universal' or spec or vis_i in assigned:
                continue
            if unmatched:
                ci = unmatched.pop(0)
                assigned[vis_i] = ci

        return assigned, cluster_info

    def _write_abilities_to_ui(self, assigned: dict, cluster_info: list, seats_visual: list,
                               visual_to_seat_id: dict, boffs_build: list, is_ground: bool) -> None:
        """
        Updates the UI elements for matched BOFF seats and writes the abilities into the SETS build structure.
        """
        from src.buildupdater import load_boff_stations
        
        _BASE_PROFS = {'Tactical', 'Engineering', 'Science'}
        env_key = 'ground' if is_ground else 'space'
        all_boff_cache = self._sets.cache.boff_abilities.get('all', {})
        written = 0

        _boff_profs = self._sets.build['ground'].get('boff_profs', {}) if is_ground else None
        _boff_specs = self._sets.build['space'].get('boff_specs', {}) if not is_ground else None

        for vis_i, ci in assigned.items():
            seat_id = visual_to_seat_id.get(vis_i)
            if seat_id is None or seat_id >= len(boffs_build):
                continue

            rank, profession, spec = seats_visual[vis_i]
            primary_prof = cluster_info[ci][1]
            cluster_items = sorted(cluster_info[ci][0], key=lambda ri: ri.bbox[0])

            if profession == 'Universal' and primary_prof in _BASE_PROFS:
                if is_ground:
                    _boff_profs[seat_id] = primary_prof
                    try:
                        self._sets.widgets.build['ground']['boff_profs'][seat_id].setCurrentText(primary_prof)
                    except Exception as _e:
                        _slog.warning(f'WARP boff: ground seat[{seat_id}] dropdown switch failed: {_e}')
                elif _boff_specs is not None and seat_id < len(_boff_specs) and isinstance(_boff_specs[seat_id], list):
                    seat_spec = spec or ''
                    target_text = f'{primary_prof} / {seat_spec}' if seat_spec else primary_prof
                    try:
                        self._sets.widgets.build['space']['boff_labels'][seat_id].setCurrentText(target_text)
                    except Exception as _e:
                        _slog.warning(f'WARP boff: seat[{seat_id}] dropdown switch failed: {_e}')
                        _boff_specs[seat_id][0] = primary_prof

            direct = [ri.slot_index for ri in cluster_items]
            if all(0 <= si < rank for si in direct) and len(set(direct)) == len(direct):
                slot_indices = direct
            else:
                slot_indices = self._slot_indices_from_x(cluster_items, rank)
                
            _slog.info(f'WARP boff: seat[{seat_id}] {profession}/{spec or "-"} rank={rank} '
                       f'← {[ri.name for ri in cluster_items]} slots={slot_indices}')

            for ri, slot_idx in zip(cluster_items, slot_indices):
                if slot_idx >= rank or ri.name in VIRTUAL_ITEM_NAMES:
                    continue
                if ri.name not in all_boff_cache:
                    continue
                boffs_build[seat_id][slot_idx] = {'item': ri.name}
                written += 1

        if written:
            load_boff_stations(self._sets, env_key)
            _slog.info(f'WARP: wrote {written} boff abilities to build')
        else:
            _slog.warning('WARP boff: 0 abilities written — check cache or cluster matching')

    def _slot_indices_from_x(self, items: list, rank: int) -> list[int]:
        """
        Maps abilities to slot indices using X-position gaps for cases where definitive slot_index is unreliable.
        """
        xs = [ri.bbox[0] for ri in items]
        if len(xs) <= 1:
            return [0]
        gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        step = min(gaps)
        if step <= 0:
            return list(range(len(xs)))
        indices = [0]
        for gap in gaps:
            jump = max(1, round(gap / step))
            indices.append(min(indices[-1] + jump, rank - 1))
        return indices

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
            'mark':      default_mark,
            'modifiers': [None] * 4,
        }

    # ── Folder picker ──────────────────────────────────────────────────────

    def _browse_folder(self):
        from warp.folder_picker import pick_folder
        last = self._settings.value(_SETTINGS_KEY_LAST_DIR, '')
        folder = pick_folder(self, title='Select Screenshots Folder', start_dir=last)
        if folder is None:
            return
        self._folder = folder
        self._folder_label.setText(str(folder))
        self._folder_label.setStyleSheet(
            f'color:{FG};background:{MBG};border:1px solid {ACCENT};'
            f'border-radius:3px;padding:3px 7px;')
        self._settings.setValue(_SETTINGS_KEY_LAST_DIR, str(folder))
