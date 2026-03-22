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
        self._build_ui()
        if self._folder:
            self._folder_label.setText(str(self._folder))
            self._folder_label.setStyleSheet(
                'color:#eee;background:#111;border:1px solid #3a6a9c;'
                'border-radius:3px;padding:3px 7px;')

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

        # Footer
        footer = QHBoxLayout()
        self._btn_next   = QPushButton('Analyse & Import →')
        self._btn_cancel = QPushButton('Cancel')
        self._btn_next.setStyleSheet(
            'QPushButton {background:#1a5c3a;color:#7effc8;'
            'border:1px solid #3aac6a;border-radius:3px;padding:5px 14px;font-weight:bold;}'
            'QPushButton:hover{background:#2a8c5a;}')
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
        ship_info    = result.ship_name or 'Unknown ship'
        msg = (
            f"Detected {n_detected} item slot(s)\n"
            f"Identified and imported: {n_imported}\n"
            f"Unmatched (skipped): {n_unmatched}\n\n"
            f"Ship: {ship_info} {result.ship_type} {result.ship_tier}"
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
        from src.buildupdater import slot_equipment_item, slot_trait_item

        r = self._import_result
        if r is None:
            return

        _ship_data = None  # captured for boff seat mapping after ship selection

        # Set ship info if recognised
        # Set ship name, ship selection and tier from recognised result
        _slog.info(f'WARP: _apply_to_sets ship_name={r.ship_name!r} ship_type={r.ship_type!r} ship_tier={r.ship_tier!r}')
        if r.ship_name or r.ship_type or r.ship_tier:
            try:
                build   = self._sets.build['space']
                widgets = self._sets.widgets
                # Set free-text ship name (U.S.S. Barbarossa)
                if r.ship_name and not build.get('ship_name'):
                    build['ship_name'] = r.ship_name
                    widgets.ship['name'].setText(r.ship_name)
                # Try to select ship class from cache using ship_type string
                # cache.ships is keyed by wiki Page name
                if r.ship_type and build.get('ship', '<Pick Ship>') in ('<Pick Ship>', '', None):
                    from difflib import get_close_matches
                    ships = getattr(self._sets.cache, 'ships', {})
                    candidates = list(ships.keys())
                    # Try exact then fuzzy match
                    match = None
                    if r.ship_type in ships:
                        match = r.ship_type
                    else:
                        # Word-subset: OCR may omit subtype words (e.g. 'Nautilus')
                        ocr_words = set(r.ship_type.lower().split())
                        subset_hits = [c for c in candidates
                                       if ocr_words.issubset(set(c.lower().split()))]
                        if len(subset_hits) == 1:
                            match = subset_hits[0]
                        elif len(subset_hits) > 1:
                            # Pick fewest extra words
                            match = min(subset_hits,
                                        key=lambda c: len(set(c.lower().split()) - ocr_words))
                        else:
                            hits = get_close_matches(r.ship_type, candidates, n=1, cutoff=0.68)
                            if hits:
                                match = hits[0]
                    if match:
                        _slog.info(f'WARP: auto-selecting ship {match!r} from {r.ship_type!r}')
                        # Use select_ship logic directly — handles image, tier, slots
                        try:
                            from src.callbacks import (
                                _save_session_slots, _restore_session_slots,
                                align_space_frame)
                            from src.widgets import exec_in_thread
                            sets = self._sets
                            ship_data = ships[match]
                            _ship_data = ship_data  # capture for boff writing below
                            sets.building = True
                            widgets.ship['button'].setText(match)
                            # Load image
                            image_filename = ship_data['image'][5:]
                            def _on_ship_image(img, _w=widgets):
                                ship_img = img[0]
                                _w.ship['image'].set_image(ship_img)
                            exec_in_thread(sets, sets.images.get_ship_image,
                                           image_filename, result=_on_ship_image)
                            # Tier combo
                            tier = ship_data['tier']
                            widgets.ship['tier'].clear()
                            if tier == 6:
                                widgets.ship['tier'].addItems(('T6', 'T6-X', 'T6-X2'))
                            elif tier == 5:
                                widgets.ship['tier'].addItems(('T5', 'T5-U', 'T5-X', 'T5-X2'))
                            else:
                                widgets.ship['tier'].addItem(f'T{tier}')
                            # Set tier from recognised result (e.g. T6-X2)
                            ship_tier = r.ship_tier or f'T{tier}'
                            idx = widgets.ship['tier'].findText(ship_tier)
                            if idx >= 0:
                                widgets.ship['tier'].setCurrentIndex(idx)
                            sets.build['space']['ship'] = match
                            sets.build['space']['tier'] = ship_tier
                            if ship_data.get('equipcannons') == 'yes':
                                widgets.ship['dc'].show()
                            else:
                                widgets.ship['dc'].hide()
                            _save_session_slots(sets)
                            align_space_frame(sets, ship_data, clear=False)
                            _restore_session_slots(sets)
                            sets.building = False
                        except Exception as _se:
                            _slog.warning(f'WARP: ship select failed: {_se}')
                    else:
                        _slog.info(f'WARP: ship {r.ship_type!r} not found in cache')
            except Exception as _e:
                _slog.warning(f'WARP: ship info widget update failed: {_e}')

        boff_items = []
        for ri in r.items:
            if not ri.name:
                continue
            # Boff abilities are handled separately after equipment/traits
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
                if is_equipment:
                    item_data = self._make_equipment_item(ri, build_key, env)
                    if item_data:
                        build_list = self._sets.build[env].get(build_key, [])
                        if idx >= len(build_list):
                            log.warning(
                                f'WARP: slot {ri.slot}[{idx}] out of range '
                                f'(build has {len(build_list)} slots) — skipping "{ri.name}"'
                            )
                        else:
                            slot_equipment_item(self._sets, item_data, env, build_key, idx)
                else:
                    item_data = {'item': ri.name}
                    slot_trait_item(self._sets, item_data, env, build_key, idx)
            except Exception as e:
                log.warning(f'WARP: Failed to import "{ri.name}" into {ri.slot}[{idx}]: {e}')

        if boff_items:
            self._write_boffs_to_build(boff_items, _ship_data)

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

    def _write_boffs_to_build(self, boff_items: list, ship_data: dict | None) -> None:
        """Write recognized boff abilities to the SETS build.

        Two-phase approach:
        1. Y/X clustering: group boff items by position into per-seat clusters.
           STO's BOFFS screen can show 2 seats on the same visual row (same Y band)
           separated by a large X gap — handled by pass-2 X splitting.
        2. Profession matching: each cluster is matched to the correct ship seat by
           comparing the dominant ability profession against seat definitions.
           This is robust regardless of ship_data['boffs'] ordering.
        """
        from collections import Counter
        from src.buildupdater import get_boff_spec, load_boff_stations

        if not ship_data:
            _slog.debug(f'WARP boff: no ship_data — {len(boff_items)} items ignored')
            return

        ship_seats_raw = ship_data.get('boffs', [])
        if not ship_seats_raw:
            _slog.warning('WARP boff: ship_data has no boffs field')
            return

        # Parse all seats; build SETS seat_id mapping (rank-descending sort,
        # same as align_space_frame).
        try:
            seats_visual = [get_boff_spec(self._sets, s) for s in ship_seats_raw]
        except Exception as e:
            _slog.warning(f'WARP boff: could not parse seat specs: {e}')
            return

        # visual_to_seat_id[ship_data_idx] = SETS seat_id (rank-desc position)
        sorted_ix = sorted(enumerate(seats_visual), key=lambda p: p[1], reverse=True)
        visual_to_seat_id = {vis_i: seat_id for seat_id, (vis_i, _) in enumerate(sorted_ix)}

        # ── Phase 1: cluster boff items into per-seat groups ─────────────────
        # Pass A: Y-bands (new band if Y gap > threshold)
        # Pass B: within each Y-band, split on large X gaps between item edges
        Y_THRESHOLD = 30   # px
        X_THRESHOLD = 50   # px — within-seat gap ~2-5px, between-seat ~100px+

        items_with_bbox = [ri for ri in boff_items if ri.bbox]
        if not items_with_bbox:
            _slog.warning('WARP boff: no boff items with bbox — cannot route')
            return

        items_sorted_y = sorted(items_with_bbox, key=lambda ri: ri.bbox[1])

        y_bands: list[list] = []
        for ri in items_sorted_y:
            y = ri.bbox[1]
            if not y_bands or y - y_bands[-1][-1].bbox[1] > Y_THRESHOLD:
                y_bands.append([ri])
            else:
                y_bands[-1].append(ri)

        seat_clusters: list[list] = []
        for band in y_bands:
            x_sorted = sorted(band, key=lambda ri: ri.bbox[0])
            cluster = [x_sorted[0]]
            for ri in x_sorted[1:]:
                prev_right = cluster[-1].bbox[0] + cluster[-1].bbox[2]
                if ri.bbox[0] - prev_right > X_THRESHOLD:
                    seat_clusters.append(cluster)
                    cluster = [ri]
                else:
                    cluster.append(ri)
            seat_clusters.append(cluster)

        _slog.info(f'WARP boff: {len(items_with_bbox)} items → '
                   f'{len(y_bands)} Y-bands → {len(seat_clusters)} seat clusters '
                   f'(ship has {len(ship_seats_raw)} seats)')

        # ── Phase 2: match each cluster to a ship seat by profession ─────────
        # Map spec name → profession name (for Universal-spec seats and spec-prof seats)
        _SPEC_TO_PROF = {
            'Temporal Operative': 'Temporal',
            'Command':            'Command',
            'Miracle Worker':     'Miracle Worker',
            'Intelligence':       'Intelligence',
            'Pilot':              'Pilot',
        }

        # For each cluster, compute the profession distribution from slot annotations.
        # primary_prof = most-common profession; prof_set = all professions in cluster.
        ClusterInfo = list[tuple]   # (cluster_items, primary_prof, prof_set, rank_count)
        cluster_info: list = []
        for c in seat_clusters:
            prof_counts = Counter(ri.slot[5:] for ri in c)   # 'Boff Temporal' → 'Temporal'
            primary = prof_counts.most_common(1)[0][0]
            cluster_info.append([c, primary, set(prof_counts.keys())])

        # Match clusters to seats in three passes:
        # 1. Named profession seats (non-Universal) with specialization
        # 2. Named profession seats (non-Universal) without specialization
        # 3. Universal seats with specialization (match by spec-profession)
        # 4. Universal seats without specialization (leftover clusters)
        unmatched = list(range(len(cluster_info)))      # indices into cluster_info
        assigned: dict[int, int] = {}                   # ship_data_idx → cluster_info_idx

        def _find_cluster(primary_prof: str, spec_prof: str | None) -> int | None:
            for ci in unmatched:
                c_primary, c_profs = cluster_info[ci][1], cluster_info[ci][2]
                if c_primary != primary_prof:
                    continue
                if spec_prof and spec_prof not in c_profs:
                    continue
                return ci
            return None

        # Pass 1 & 2: explicit-profession seats
        for vis_i, (rank, prof, spec) in enumerate(seats_visual):
            if prof == 'Universal' or vis_i in assigned:
                continue
            spec_prof = _SPEC_TO_PROF.get(spec) if spec else None
            ci = _find_cluster(prof, spec_prof)
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

        # Pass 4: Universal seats without spec — assign remaining clusters in order
        for vis_i, (rank, prof, spec) in enumerate(seats_visual):
            if prof != 'Universal' or spec or vis_i in assigned:
                continue
            if unmatched:
                ci = unmatched.pop(0)
                assigned[vis_i] = ci

        # ── Phase 3: write abilities to build ────────────────────────────────
        # Base professions a Universal seat can be set to (same options as the
        # dropdown the user sees in the SETS UI).
        _BASE_PROFS = {'Tactical', 'Engineering', 'Science'}

        boffs_build    = self._sets.build['space']['boffs']
        boff_specs     = self._sets.build['space']['boff_specs']
        all_boff_cache = self._sets.cache.boff_abilities.get('all', {})
        written        = 0

        for vis_i, ci in assigned.items():
            seat_id = visual_to_seat_id.get(vis_i)
            if seat_id is None or seat_id >= len(boffs_build):
                continue

            rank, profession, spec = seats_visual[vis_i]
            primary_prof = cluster_info[ci][1]
            cluster_items = sorted(cluster_info[ci][0], key=lambda ri: ri.bbox[0])

            # For Universal seats: set the profession dropdown to match what was
            # recognised — exactly as the user would do manually before picking abilities.
            # Only applies to base professions (Tactical/Engineering/Science).
            # Spec-based professions (Temporal, Command, …) don't change the base.
            if profession == 'Universal' and primary_prof in _BASE_PROFS:
                if seat_id < len(boff_specs) and isinstance(boff_specs[seat_id], list):
                    boff_specs[seat_id][0] = primary_prof
                    _slog.info(f'WARP boff: seat[{seat_id}] Universal → set to {primary_prof}')

            _slog.info(f'WARP boff: seat[{seat_id}] {profession}/{spec or "-"} rank={rank} '
                       f'← {[ri.name for ri in cluster_items]}')

            for rank_slot, ri in enumerate(cluster_items):
                if rank_slot >= rank:
                    break
                if ri.name not in all_boff_cache:
                    _slog.info(f'WARP boff: {ri.name!r} not in boff cache — skip')
                    continue
                boffs_build[seat_id][rank_slot] = {'item': ri.name}
                written += 1

        if written:
            load_boff_stations(self._sets, 'space')
            _slog.info(f'WARP: wrote {written} boff abilities to build')
        else:
            _slog.warning('WARP boff: 0 abilities written — check cache or cluster matching')

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
        last = self._settings.value(_SETTINGS_KEY_LAST_DIR, '')
        dlg  = QFileDialog(self)
        dlg.setWindowTitle('Select Screenshots Folder')
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        # Show image files for context but keep directory-only selection
        dlg.setNameFilter('Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)')
        if last and Path(last).is_dir():
            dlg.setDirectory(last)
        # Make file list read-only: user can see images but only dirs are selectable
        from PySide6.QtWidgets import QListView, QTreeView
        for view in dlg.findChildren(QListView) + dlg.findChildren(QTreeView):
            view.setSelectionMode(view.SelectionMode.NoSelection)
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                self._folder = Path(files[0])
                self._folder_label.setText(str(self._folder))
                self._folder_label.setStyleSheet(
                    'color:#eee;background:#111;border:1px solid #3a6a9c;'
                    'border-radius:3px;padding:3px 7px;')
                self._settings.setValue(_SETTINGS_KEY_LAST_DIR, str(self._folder))
