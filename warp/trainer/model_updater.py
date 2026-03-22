# warp/trainer/model_updater.py
#
# Background model updater — checks the WARP backend for a newer centrally-trained
# EfficientNet model and downloads it from HF when available.
#
# Flow:
#   1. Read local warp/models/model_version.json (if it exists)
#   2. Call backend GET /model/version  (rate-limited: at most once per 24 h)
#   3. Compare 'trained_at' timestamps — remote wins only if strictly newer
#   4. Download icon_classifier.pt + label_map.json + icon_classifier_meta.json
#      directly from HF (sets-sto/warp-knowledge, public read)
#   5. Write model_version.json to record the installed version
#   6. Call SETSIconMatcher.reset_ml_session() so the new model is loaded on
#      the next match request
#
# Local model always takes priority:
#   - If local model was trained more recently (WARP CORE → Train Model),
#     its trained_at will be newer than the remote version → no update.
#   - The remote update ONLY installs when the remote is strictly newer.
#
# All network calls are non-blocking (background thread).
# Failures are silently logged — the update is skipped, never crashes the app.

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_BACKEND_URL          = 'https://sets-warp-backend.onrender.com'
_CHECK_INTERVAL_HOURS = 24          # minimum hours between remote checks
_VERSION_CACHE_FILE   = 'warp/models/model_version_remote_cache.json'
_CONNECT_TIMEOUT      = 5           # seconds
_READ_TIMEOUT         = 15          # seconds
_MODEL_FILES          = [           # files to download from HF knowledge repo
    ('models/icon_classifier.pt',            'icon_classifier.pt'),
    ('models/label_map.json',               'label_map.json'),
    ('models/icon_classifier_meta.json',    'icon_classifier_meta.json'),
    ('models/model_version.json',           'model_version.json'),
    ('models/screen_classifier.pt',          'screen_classifier.pt'),
    ('models/screen_classifier_labels.json', 'screen_classifier_labels.json'),
]
# Used only for the one-time "download if missing" fallback
_SCREEN_CLASSIFIER_FILES = [
    ('models/screen_classifier.pt',          'screen_classifier.pt'),
    ('models/screen_classifier_labels.json', 'screen_classifier_labels.json'),
]


class ModelUpdater:
    """
    Non-blocking remote model update checker.

    Usage:
        ModelUpdater().check_and_update(sets_root, on_updated=lambda: log.info('reloaded'))
    """

    def check_and_update(
        self,
        sets_root: Path,
        on_updated: Callable[[], None] | None = None,
    ) -> None:
        """
        Check for a newer centrally-trained model and download it if available.
        Non-blocking — returns immediately; the check runs in a daemon thread.
        """
        threading.Thread(
            target=self._bg_check,
            args=(Path(sets_root), on_updated),
            daemon=True,
            name='warp-model-update',
        ).start()

    # ── background worker ─────────────────────────────────────────────────────

    def _bg_check(self, sets_root: Path, on_updated: Callable | None) -> None:
        try:
            models_dir = sets_root / 'warp' / 'models'

            # Always ensure screen_classifier is present (static, not version-managed)
            self._ensure_screen_classifier(models_dir)

            if not self._due_for_check(sets_root):
                return

            log.info('ModelUpdater: checking for remote model update...')
            remote = self._fetch_remote_version()
            if not remote or not remote.get('available'):
                log.debug('ModelUpdater: no model published on remote yet')
                self._save_check_timestamp(sets_root)
                return

            local_ts  = self._read_local_trained_at(models_dir)
            remote_ts = remote.get('trained_at', '')

            if local_ts and remote_ts <= local_ts:
                log.info(
                    f'ModelUpdater: local model is current '
                    f'(local={local_ts[:10]}, remote={remote_ts[:10]})'
                )
                self._save_check_timestamp(sets_root)
                return

            log.info(
                f'ModelUpdater: remote model is newer '
                f'(remote={remote_ts[:10]}, local={local_ts[:10] if local_ts else "none"}) '
                f'— downloading...'
            )

            if self._download_model(models_dir, remote):
                log.info(
                    f'ModelUpdater: model updated — '
                    f'{remote.get("n_classes")} classes, '
                    f'val_acc={remote.get("val_acc", 0):.1%}'
                )
                # Reload icon matcher ML session
                try:
                    from warp.recognition.icon_matcher import SETSIconMatcher
                    SETSIconMatcher.reset_ml_session()
                    log.info('ModelUpdater: icon matcher reloaded with new model')
                except Exception as e:
                    log.warning(f'ModelUpdater: matcher reload failed: {e}')

                if on_updated:
                    try:
                        on_updated()
                    except Exception:
                        pass

            self._save_check_timestamp(sets_root)

        except Exception as e:
            log.warning(f'ModelUpdater: update check failed: {e}')

    # ── network ───────────────────────────────────────────────────────────────

    def _fetch_remote_version(self) -> dict | None:
        """GET /model/version from backend. Returns dict or None on failure."""
        try:
            import urllib.request
            req = urllib.request.Request(
                f'{_BACKEND_URL}/model/version',
                headers={'User-Agent': 'WARP/0.4.0'},
            )
            with urllib.request.urlopen(req, timeout=_CONNECT_TIMEOUT + _READ_TIMEOUT) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            log.debug(f'ModelUpdater: /model/version fetch failed: {e}')
            return None

    def _download_model(self, models_dir: Path, remote_meta: dict) -> bool:
        """
        Download model files from HF knowledge repo directly.
        Uses hf_hub_download — no token needed for public repo.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            log.warning('ModelUpdater: huggingface_hub not installed — cannot download model')
            return False

        hf_repo = 'sets-sto/warp-knowledge'
        models_dir.mkdir(parents=True, exist_ok=True)
        tmp_files: list[tuple[Path, Path]] = []  # (tmp_path, final_path)

        # Use upload token if available — suppresses HF unauthenticated warning
        # and gets higher rate limits. Falls back to anonymous (public repo).
        token = self._read_hub_token(models_dir.parent)

        for hf_path, local_name in _MODEL_FILES:
            final_path = models_dir / local_name
            try:
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=hf_path,
                    repo_type='dataset',
                    token=token or None,
                )
                tmp_files.append((Path(downloaded), final_path))
            except Exception as e:
                # icon_classifier.pt and label_map.json are required
                if local_name in ('icon_classifier.pt', 'label_map.json'):
                    log.warning(f'ModelUpdater: required file {hf_path} unavailable: {e}')
                    return False
                log.debug(f'ModelUpdater: optional file {hf_path} unavailable: {e}')

        # Copy all files atomically (only after all required files downloaded OK)
        import shutil
        for src, dst in tmp_files:
            shutil.copy2(src, dst)
            log.debug(f'ModelUpdater: installed {dst.name}')

        return True

    def _ensure_screen_classifier(self, models_dir: Path) -> None:
        """Download screen_classifier.pt from HF if it's missing (one-time, silent)."""
        pt_path = models_dir / 'screen_classifier.pt'
        if pt_path.exists():
            return
        log.info('ModelUpdater: screen_classifier.pt missing — downloading from HF...')
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return
        import shutil
        hf_repo = 'sets-sto/warp-knowledge'
        token = self._read_hub_token(models_dir.parent)
        models_dir.mkdir(parents=True, exist_ok=True)
        for hf_path, local_name in _SCREEN_CLASSIFIER_FILES:
            try:
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=hf_path,
                    repo_type='dataset',
                    token=token or None,
                )
                shutil.copy2(downloaded, models_dir / local_name)
                log.info(f'ModelUpdater: downloaded {local_name}')
            except Exception as e:
                log.warning(f'ModelUpdater: could not download {hf_path}: {e}')

    # ── rate-limiting (check at most once per 24 h) ───────────────────────────

    def _due_for_check(self, sets_root: Path) -> bool:
        cache_path = self._cache_path(sets_root)
        if not cache_path.exists():
            return True
        try:
            data     = json.loads(cache_path.read_text(encoding='utf-8'))
            last_ts  = data.get('last_check', 0)
            elapsed  = (time.time() - last_ts) / 3600   # hours
            return elapsed >= _CHECK_INTERVAL_HOURS
        except Exception:
            return True

    def _save_check_timestamp(self, sets_root: Path) -> None:
        cache_path = self._cache_path(sets_root)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({'last_check': time.time()}),
                encoding='utf-8',
            )
        except Exception:
            pass

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _read_hub_token(warp_dir: Path) -> str:
        """Read hub_token.txt from warp/ directory. Returns '' if not found."""
        try:
            return (warp_dir / 'hub_token.txt').read_text().strip()
        except Exception:
            return ''

    @staticmethod
    def _read_local_trained_at(models_dir: Path) -> str:
        """Return 'trained_at' from local model_version.json, or '' if not present."""
        version_file = models_dir / 'model_version.json'
        if not version_file.exists():
            return ''
        try:
            return json.loads(version_file.read_text(encoding='utf-8')).get('trained_at', '')
        except Exception:
            return ''

    @staticmethod
    def _cache_path(sets_root: Path) -> Path:
        return sets_root / _VERSION_CACHE_FILE
