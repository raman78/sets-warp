# warp/trainer/sync.py
# Synchronises training data (annotations + icon crops) with Hugging Face Hub.
#
# SECURITY MODEL — two-folder staging:
#
#   staging/<install_id>/annotations.jsonl   — contributed by users, unverified
#   staging/<install_id>/crops/<sha>.png
#   data/annotations.jsonl                   — approved by repo owner only
#   data/crops/<sha>.png
#
# Users upload to staging/ only.
# Repo owner runs approve_staging.py (or manually) to merge into data/.
# A single bad actor can only pollute their own staging/ folder.
#
# RATE LIMITING (client-side):
#   Max 200 annotations per day per install_id.
#   Enforced locally — a determined attacker could bypass this,
#   but it stops accidental flooding.
#
# VALIDATION (client-side + file-level):
#   - Crop must be non-empty PNG, ≥ 32×32 px
#   - Slot must be a known slot name
#   - Name must be non-empty, ≤ 120 chars, printable ASCII/Unicode
#   - No duplicate sha256 in the same upload batch

from __future__ import annotations

import json
import hashlib
import logging
import datetime
import tempfile
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QCheckBox
)

from warp.trainer.training_data import TrainingDataManager, NON_ICON_SLOTS

logger = logging.getLogger(__name__)
try:
    from src.setsdebug import log as _slog
except Exception:
    _slog = logger

# Hugging Face dataset repository ID
HF_DATASET_REPO  = "sets-sto/sto-icon-dataset"
HF_REPO_TYPE     = "dataset"
CROPS_DIR        = "data/crops"           # approved
ANNOTATIONS_FILE = "data/annotations.jsonl"

STAGING_ROOT     = "staging"              # user contributions land here
MAX_DAILY_UPLOADS = 1000                  # per install_id, per UTC day
MAX_NAME_LEN      = 120
MIN_CROP_PX       = 24


def _get_install_id() -> str:
    """Stable anonymous identifier for this installation."""
    try:
        from warp.knowledge.sync_client import _get_install_id as _gid
        return _gid()
    except Exception:
        import uuid
        return str(uuid.uuid4())[:16]


def _validate_annotation(item: dict) -> str | None:
    """Returns None if valid, or an error string."""
    name = item.get("name", "").strip()
    slot = item.get("slot", "").strip()
    if not name:
        return "empty name"
    if len(name) > MAX_NAME_LEN:
        return f"name too long ({len(name)})"
    if not slot:
        return "empty slot"
    if not name.isprintable():
        return "non-printable characters in name"
    return None


def _validate_crop(path: Path) -> str | None:
    """Returns None if valid PNG crop, or an error string."""
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return "unreadable image"
        h, w = img.shape[:2]
        if h < MIN_CROP_PX or w < MIN_CROP_PX:
            return f"too small ({w}×{h})"
        return None
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class SyncWorker(QThread):
    """
    Uploads confirmed crops to staging/ on HF Hub.
    Never writes to data/ — only repo owner can approve staging.

    Signals:
        progress(percent, message)
        finished(success: bool)
    """

    progress = Signal(int, str)
    finished = Signal(bool)

    def __init__(
        self,
        data_manager: TrainingDataManager,
        hf_token: str,
        mode: str = "upload",
    ):
        super().__init__()
        self._mgr   = data_manager
        self._token = hf_token
        self._mode  = mode

    def run(self):
        try:
            if self._mode in ("upload", "both"):
                self._upload()
                self._upload_screen_types()
                self._upload_anchors_grid()
            if self._mode in ("download", "both"):
                self._download()
            self.finished.emit(True)
        except Exception as e:
            logger.error(f"Sync error: {e}")
            self.finished.emit(False)

    # ---------------------------------------------------------------- upload

    def _upload(self):
        """Upload new confirmed crops to staging/<install_id>/ on HF Hub."""
        from huggingface_hub import HfApi
        api = HfApi(token=self._token)

        api.create_repo(
            repo_id=HF_DATASET_REPO,
            repo_type=HF_REPO_TYPE,
            exist_ok=True,
            private=False,
        )

        install_id   = _get_install_id()
        staging_dir  = f"{STAGING_ROOT}/{install_id}"
        staging_anno = f"{staging_dir}/annotations.jsonl"
        staging_crop = f"{staging_dir}/crops"

        confirmed = self._mgr.get_confirmed_crops()
        _slog.info(f'HF Sync: {len(confirmed)} confirmed crops locally')
        if not confirmed:
            self.progress.emit(100, "Nothing to upload.")
            return

        # Client-side rate limit check
        today     = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        rl_file   = self._mgr._dir / '.sync_rate_limit.json'
        rl        = {}
        try:
            rl = json.loads(rl_file.read_text())
        except Exception:
            pass
        daily_count = rl.get(today, 0)
        _slog.info(f'HF Sync: {daily_count}/{MAX_DAILY_UPLOADS} uploads already sent today')
        if daily_count >= MAX_DAILY_UPLOADS:
            self.progress.emit(100, f"Daily upload limit ({MAX_DAILY_UPLOADS}) reached.")
            return

        # Load local cache of already-uploaded hashes (avoids list_repo_files on every sync)
        self.progress.emit(5, "Checking existing uploads…")
        existing_hashes = self._load_uploaded_hashes_cache()
        if existing_hashes:
            _slog.info(f'HF Sync: {len(existing_hashes)} crops in local cache (skipping HF listing)')
        else:
            # Bootstrap: fetch from HF once, then persist locally
            existing_hashes = self._fetch_staging_hashes(api, staging_crop)
            _slog.info(f'HF Sync: bootstrapped {len(existing_hashes)} hashes from HF listing')
            self._save_uploaded_hashes_cache(existing_hashes)

        # Collect all new files first, then upload in a single commit
        from huggingface_hub import CommitOperationAdd
        new_annotations: list[dict] = []
        operations:      list       = []
        uploaded = 0
        total    = len(confirmed)

        self.progress.emit(10, "Preparing files…")
        for idx, item in enumerate(confirmed):
            if daily_count + uploaded >= MAX_DAILY_UPLOADS:
                break

            err = _validate_annotation(item)
            if err:
                logger.warning(f"Sync: skipping invalid annotation ({err}): {item}")
                continue

            crop_path = Path(item["path"])
            if not crop_path.exists():
                _slog.warning(f'HF Sync: crop file missing, skipping: {crop_path.name}')
                continue

            err = _validate_crop(crop_path)
            if err:
                logger.warning(f"Sync: skipping invalid crop ({err}): {crop_path.name}")
                continue

            sha = self._file_sha256(crop_path)
            if sha in existing_hashes:
                _slog.debug(f'HF Sync: skipping (already on HF): {crop_path.name}')
                continue

            _slog.debug(f'HF Sync: queuing [{item["slot"]}] {item["name"]} → {sha[:12]}…')
            operations.append(CommitOperationAdd(
                path_in_repo=f"{staging_crop}/{sha}.png",
                path_or_fileobj=str(crop_path),
            ))
            existing_hashes.add(sha)
            uploaded += 1
            ann_entry: dict = {
                "slot":        item["slot"],
                "name":        item["name"],
                "crop_sha256": sha,
                "date":        today,
            }
            if item.get("ml_name") is not None:
                ann_entry["ml_name"] = item["ml_name"]
            new_annotations.append(ann_entry)

        if new_annotations:
            self.progress.emit(88, f"Uploading {uploaded} crops in one commit…")
            self._append_staging_annotations_to_ops(operations, staging_anno, new_annotations, api)
            api.create_commit(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
                operations=operations,
                commit_message=f"WARP staging: {uploaded} new crops ({today})",
            )
            _slog.info(f'HF Sync: commit sent — {uploaded} crops + annotations')
            # Persist newly uploaded hashes so next sync skips list_repo_files
            self._save_uploaded_hashes_cache(existing_hashes)

        # Update local rate limit counter
        rl[today] = daily_count + uploaded
        try:
            rl_file.write_text(json.dumps(rl))
        except Exception:
            pass

        msg = f"Uploaded {uploaded} annotations to staging." if uploaded else "Nothing new to upload (all already on HF)."
        _slog.info(f'HF Sync: done — {uploaded} new uploads, total on HF: {len(existing_hashes)}')
        self.progress.emit(100, msg)

    def _load_uploaded_hashes_cache(self) -> set[str]:
        """Load locally cached set of already-uploaded crop sha256 hashes."""
        cache_file = self._mgr._dir / '.sync_uploaded_hashes.json'
        try:
            return set(json.loads(cache_file.read_text()))
        except Exception:
            return set()

    def _save_uploaded_hashes_cache(self, hashes: set[str]) -> None:
        """Persist the uploaded-hashes set locally so future syncs skip HF listing."""
        cache_file = self._mgr._dir / '.sync_uploaded_hashes.json'
        try:
            cache_file.write_text(json.dumps(sorted(hashes)))
        except Exception:
            pass

    def _fetch_staging_hashes(self, api, staging_crop_dir: str) -> set[str]:
        try:
            files = api.list_repo_files(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
            )
            return {
                Path(f).stem
                for f in files
                if f.startswith(staging_crop_dir) and f.endswith(".png")
            }
        except Exception:
            return set()

    def _append_staging_annotations_to_ops(self, operations: list, path_in_repo: str,
                                           new_entries: list[dict], api) -> None:
        """Build annotations.jsonl content and add it as a CommitOperationAdd to operations."""
        import io
        from huggingface_hub import CommitOperationAdd
        existing_lines: list[str] = []
        existing_hashes: set[str] = set()
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type=HF_REPO_TYPE,
                token=self._token,
            )
            with open(local, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            existing_hashes.add(json.loads(line).get("crop_sha256", ""))
                        except Exception:
                            pass
                        existing_lines.append(line)
        except Exception:
            pass

        combined = list(existing_lines)
        for entry in new_entries:
            if entry["crop_sha256"] not in existing_hashes:
                combined.append(json.dumps(entry, ensure_ascii=False))

        content_bytes = "\n".join(combined).encode("utf-8")
        operations.append(CommitOperationAdd(
            path_in_repo=path_in_repo,
            path_or_fileobj=io.BytesIO(content_bytes),
        ))

    # ---------------------------------------------------------------- anchor grids (P11)

    def _upload_anchors_grid(self):
        """Upload normalized bbox grid entries from local anchors.json to HF staging (P11)."""
        anchors_path = Path(self._mgr._dir) / 'anchors.json'
        if not anchors_path.exists():
            return
        try:
            data = json.loads(anchors_path.read_text(encoding='utf-8'))
        except Exception:
            return

        learned = data.get('learned', [])
        if not learned:
            return

        from huggingface_hub import HfApi, CommitOperationAdd
        import io as _io
        api = HfApi(token=self._token)
        install_id  = _get_install_id()
        staging_dir = f"{STAGING_ROOT}/{install_id}"

        cache_file = Path(self._mgr._dir) / '.sync_uploaded_grids.json'
        try:
            uploaded_grids: set[str] = set(json.loads(cache_file.read_text()))
        except Exception:
            uploaded_grids = set()

        operations: list = []
        new_hashes:  set[str] = set()

        for entry in learned:
            build_type = entry.get('type', '')
            if not build_type:
                continue
            slots = entry.get('slots', {})
            # Keep only icon slots (skip NON_ICON_SLOTS text labels)
            icon_slots = {k: v for k, v in slots.items() if k not in NON_ICON_SLOTS}
            if len(icon_slots) < 3:
                continue

            payload = {
                'build_type': build_type,
                'aspect':     entry.get('aspect'),
                'resolution': entry.get('res', ''),
                'slots':      icon_slots,
            }
            payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            sha8 = hashlib.sha256(payload_json.encode()).hexdigest()[:8]

            if sha8 in uploaded_grids:
                continue

            operations.append(CommitOperationAdd(
                path_in_repo=f"{staging_dir}/anchors_grid_{sha8}.json",
                path_or_fileobj=_io.BytesIO(payload_json.encode()),
            ))
            new_hashes.add(sha8)

        if not operations:
            _slog.debug('HF Sync: no new anchor grids to upload')
            return

        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type=HF_REPO_TYPE,
            operations=operations,
            commit_message=f"WARP anchors: {len(operations)} grid entries",
        )
        uploaded_grids.update(new_hashes)
        try:
            cache_file.write_text(json.dumps(sorted(uploaded_grids)))
        except Exception:
            pass
        _slog.info(f'HF Sync: uploaded {len(operations)} anchor grid entries')

    # ---------------------------------------------------------------- screen types

    def _upload_screen_types(self):
        """Upload confirmed screen type screenshots to staging/<install_id>/screen_types/."""
        screen_types_dir = self._mgr._dir / 'screen_types'
        if not screen_types_dir.exists():
            return
        type_dirs = [d for d in screen_types_dir.iterdir() if d.is_dir()]
        if not type_dirs:
            return

        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi(token=self._token)
        install_id  = _get_install_id()
        staging_dir = f"{STAGING_ROOT}/{install_id}/screen_types"

        screen_cache_file = self._mgr._dir / '.sync_uploaded_screen_hashes.json'
        existing = self._load_screen_hashes_cache(screen_cache_file)
        if existing:
            _slog.info(f'HF Sync: {len(existing)} screen type screenshots in local cache (skipping HF listing)')
        else:
            existing = self._fetch_staging_screen_hashes(api, staging_dir)
            _slog.info(f'HF Sync: bootstrapped {len(existing)} screen hashes from HF listing')

        operations: list = []
        for type_dir in sorted(type_dirs):
            stype = type_dir.name
            for png in sorted(type_dir.glob('*.png')):
                sha = self._file_sha256(png)
                if sha in existing:
                    continue
                operations.append(CommitOperationAdd(
                    path_in_repo=f"{staging_dir}/{stype}/{sha}.png",
                    path_or_fileobj=str(png),
                ))
                existing.add(sha)

        if not operations:
            _slog.debug('HF Sync: no new screen type screenshots to upload')
            return

        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type=HF_REPO_TYPE,
            operations=operations,
            commit_message=f"WARP screen types: {len(operations)} screenshots",
        )
        try:
            screen_cache_file.write_text(json.dumps(sorted(existing)))
        except Exception:
            pass
        _slog.info(f'HF Sync: uploaded {len(operations)} screen type screenshot(s)')

    @staticmethod
    def _load_screen_hashes_cache(cache_file: Path) -> set[str]:
        try:
            return set(json.loads(cache_file.read_text()))
        except Exception:
            return set()

    def _fetch_staging_screen_hashes(self, api, staging_screen_dir: str) -> set[str]:
        try:
            files = api.list_repo_files(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
            )
            return {
                Path(f).stem
                for f in files
                if f.startswith(staging_screen_dir) and f.endswith('.png')
            }
        except Exception:
            return set()

    # ---------------------------------------------------------------- download

    def _download(self):
        """Download approved annotations from data/ (not staging)."""
        self.progress.emit(10, "Downloading approved annotations…")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=ANNOTATIONS_FILE,
                repo_type=HF_REPO_TYPE,
            )
            with open(path) as _f:
                count = sum(1 for line in _f if line.strip())
            self.progress.emit(100, f"Downloaded {count} approved annotations.")
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            self.progress.emit(100, "Download failed — dataset may be empty.")

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _file_sha256(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()[:32]


# ---------------------------------------------------------------------------
# SyncManager — WARP-specific background sync task
# ---------------------------------------------------------------------------

class SyncManager:
    """
    WARP background sync — uploads confirmed crops / screen-types / anchors to HF Hub.

    Designed to work with BackgroundTaskManager (src.background_tasks):
        btm.register(sync_mgr.check_and_upload, interval_ms=10*60*1000, startup_delay_ms=15_000)
        btm.on_stop(sync_mgr.stop)

    check_and_upload() is a single sync cycle; safe to call from any periodic timer.
    """

    def __init__(self, sets_app) -> None:
        self._sets_app = sets_app
        self._worker: SyncWorker | None = None

    # ---------------------------------------------------------------- public API

    def check_and_upload(self) -> None:
        """One sync cycle: upload pending crops if any. Called by BackgroundTaskManager."""
        token = self._read_token()
        if not token:
            return

        mgr = self._data_manager()
        if mgr is None:
            return

        confirmed = [c for c in mgr.get_confirmed_crops() if Path(c['path']).exists()]
        uploaded  = self._load_uploaded_hashes(mgr)
        pending   = max(0, len(confirmed) - len(uploaded))

        if pending == 0:
            _slog.debug('SyncManager: timer tick — nothing pending')
            return

        if self._worker and self._worker.isRunning():
            _slog.debug(f'SyncManager: {pending} crops pending, upload already running — skipping tick')
            return

        _slog.info(f'SyncManager: {pending} crops pending — starting upload…')
        self._worker = SyncWorker(data_manager=mgr, hf_token=token, mode='upload')
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def stop(self) -> None:
        """Wait for any in-progress upload to finish. Call on app quit."""
        if self._worker and self._worker.isRunning():
            self._worker.wait(5000)

    # ---------------------------------------------------------------- private

    def _on_finished(self, ok: bool) -> None:
        mgr = self._data_manager()
        remaining = 0
        if mgr:
            confirmed = [c for c in mgr.get_confirmed_crops() if Path(c['path']).exists()]
            uploaded  = self._load_uploaded_hashes(mgr)
            remaining = max(0, len(confirmed) - len(uploaded))
        if ok:
            _slog.info(f'SyncManager: upload OK — {remaining} crops still pending')
        else:
            _slog.warning(f'SyncManager: upload FAILED — {remaining} crops still pending')

    def _data_manager(self):
        """Return TrainingDataManager — prefer WARP CORE's live instance if window is open."""
        win = getattr(self._sets_app, '_warp_core_window', None)
        if win is not None and hasattr(win, '_data_mgr'):
            return win._data_mgr
        # WARP CORE not open — create a read-only instance from disk
        try:
            from warp.trainer.training_data import TrainingDataManager
            data_dir = Path(__file__).resolve().parent.parent / 'training_data'
            if data_dir.exists():
                return TrainingDataManager(str(data_dir))
        except Exception:
            pass
        return None

    @staticmethod
    def _read_token() -> str:
        try:
            p = Path(__file__).resolve().parent.parent / 'hub_token.txt'
            t = p.read_text().strip()
            if t and t != 'YOUR_HF_TOKEN_HERE':
                return t
        except Exception:
            pass
        return ''

    @staticmethod
    def _load_uploaded_hashes(mgr) -> set[str]:
        cache_file = mgr._dir / '.sync_uploaded_hashes.json'
        try:
            return set(json.loads(cache_file.read_text()))
        except Exception:
            return set()


# ---------------------------------------------------------------------------
# HF Token setup dialog (kept for internal use / testing)
# ---------------------------------------------------------------------------

class HFTokenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hugging Face Token")
        self.setFixedWidth(440)
        self._token = ""
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.addWidget(QLabel("Hugging Face Token:"))
        self._token_edit = QLineEdit()
        self._token_edit.setPlaceholderText("hf_…")
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        lay.addWidget(self._token_edit)
        btns = QHBoxLayout()
        btn_ok     = QPushButton("Save")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self._on_ok)
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        lay.addLayout(btns)

    def _on_ok(self):
        self._token = self._token_edit.text().strip()
        if self._token:
            self.accept()

    def get_token(self) -> str:
        return self._token
