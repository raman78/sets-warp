# warp/knowledge/sync_client.py
#
# Client for the WARP Knowledge Backend.
#
# Responsibilities:
#   1. UPLOAD: send crop images + confirmed labels to the backend
#      (backend holds the HF write token — never exposed to users)
#
#   2. DOWNLOAD: fetch the community knowledge base at startup
#      knowledge.json = {phash_hex: item_name, ...}
#      Used by icon_matcher as an override layer before template matching.
#
# All network calls are:
#   - Non-blocking (background thread)
#   - Silent on failure (log warning, never crash the app)
#   - Rate-limited locally (max MAX_CONTRIBUTIONS_PER_DAY per installation)
#
# Configuration (warp/knowledge/config.json or hardcoded defaults):
#   BACKEND_URL   — base URL of the backend service
#   INSTALL_ID    — random UUID generated on first run, stored locally
#                   used for deduplication on the backend (not for user identity)

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_BACKEND_URL       = 'https://sets-warp-backend.onrender.com'
KNOWLEDGE_CACHE_FILE      = Path('warp') / 'knowledge' / 'knowledge_cache.json'
INSTALL_ID_FILE           = Path('warp') / 'knowledge' / 'install_id.txt'
RATE_LIMIT_FILE           = Path('warp') / 'knowledge' / 'rate_limit.json'
MAX_CONTRIBUTIONS_PER_DAY = 200    # per installation, per day
KNOWLEDGE_MAX_AGE_HOURS   = 24     # re-download knowledge base after this
CONNECT_TIMEOUT           = 5      # seconds
READ_TIMEOUT              = 15     # seconds — knowledge download (has cache fallback)
CONTRIBUTE_TIMEOUT        = 60     # seconds — longer: covers Render cold-start (~50 s)

WARP_VERSION = '0.4.0'


class WARPSyncClient:
    """
    Non-blocking client for WARP community knowledge sync.

    Usage:
        client = WARPSyncClient()                 # start background download
        overrides = client.get_knowledge()        # {phash_hex: item_name}

        client.contribute(
            crop_bgr    = np.ndarray,             # icon crop (BGR)
            item_name   = 'Fleet Deflector...',   # confirmed name
            wrong_name  = 'Other Item...',        # what WARP guessed (optional)
            confirmed   = True,                   # user explicitly confirmed
        )
    """

    def __init__(self, backend_url: str | None = None):
        self._url       = (backend_url or self._load_config_url()
                           or DEFAULT_BACKEND_URL).rstrip('/')
        self._install_id = self._get_or_create_install_id()
        self._knowledge: dict[str, str] = {}   # phash_hex → item_name
        self._knowledge_lock = threading.Lock()

        # Start background knowledge download immediately
        threading.Thread(
            target=self._download_knowledge_bg,
            daemon=True, name='warp-knowledge-dl'
        ).start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, str]:
        """
        Return current community knowledge dict {phash_hex: item_name}.
        Always returns instantly (uses cached value, updated in background).
        """
        with self._knowledge_lock:
            return dict(self._knowledge)

    def contribute(
        self,
        crop_bgr:   np.ndarray,
        item_name:  str,
        wrong_name: str = '',
        confirmed:  bool = True,
        on_done:    Callable[[bool], None] | None = None,
    ) -> None:
        """
        Upload a crop + label to the community knowledge base.

        Non-blocking — fires a background thread and returns immediately.
        on_done(success: bool) is called when the upload completes (optional).
        """
        if not item_name.strip():
            log.debug('WARPSync: contribute called with empty item_name, skipped')
            return

        if not self._check_rate_limit():
            log.debug('WARPSync: rate limit reached for today, skipping contribution')
            if on_done:
                on_done(False)
            return

        threading.Thread(
            target=self._contribute_bg,
            args=(crop_bgr.copy(), item_name, wrong_name, confirmed, on_done),
            daemon=True, name='warp-contribute'
        ).start()

    def refresh_knowledge(self) -> None:
        """Force re-download of knowledge base (ignores cache age)."""
        threading.Thread(
            target=self._download_knowledge_bg,
            args=(True,),
            daemon=True, name='warp-knowledge-refresh'
        ).start()

    # ── Background workers ─────────────────────────────────────────────────────

    def _download_knowledge_bg(self, force: bool = False) -> None:
        """Download knowledge.json from backend; update self._knowledge."""

        # Check cache freshness
        cache_path = self._resolve_path(KNOWLEDGE_CACHE_FILE)
        if not force and cache_path.exists():
            try:
                mtime = cache_path.stat().st_mtime
                age_h = (time.time() - mtime) / 3600
                if age_h < KNOWLEDGE_MAX_AGE_HOURS:
                    data = json.loads(cache_path.read_text(encoding='utf-8'))
                    with self._knowledge_lock:
                        self._knowledge = data.get('knowledge', {})
                    log.debug(f'WARPSync: loaded {len(self._knowledge)} knowledge entries from cache')
                    return
            except Exception as e:
                log.debug(f'WARPSync: cache read failed: {e}')

        try:
            import urllib.request
            req = urllib.request.Request(
                f'{self._url}/knowledge',
                headers={'User-Agent': f'WARP/{WARP_VERSION}'},
            )
            with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT + READ_TIMEOUT) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            knowledge = data.get('knowledge', {})

            with self._knowledge_lock:
                self._knowledge = knowledge

            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({'knowledge': knowledge, 'fetched_at': time.time()},
                           ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            log.info(f'WARPSync: downloaded {len(knowledge)} knowledge entries')

        except Exception as e:
            log.warning(f'WARPSync: knowledge download failed: {e}')
            # Fall back to cached data even if stale
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding='utf-8'))
                    with self._knowledge_lock:
                        self._knowledge = data.get('knowledge', {})
                    log.debug('WARPSync: using stale cache as fallback')
                except Exception:
                    pass

    def _contribute_bg(
        self,
        crop_bgr:   np.ndarray,
        item_name:  str,
        wrong_name: str,
        confirmed:  bool,
        on_done:    Callable[[bool], None] | None,
    ) -> None:
        """Send crop + label to backend (runs in background thread)."""
        try:
            import cv2

            # Encode crop as PNG → base64
            ok, buf = cv2.imencode('.png', crop_bgr)
            if not ok:
                raise ValueError('imencode failed')
            crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

            # Compute phash for deduplication
            icon64 = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
            phash  = _compute_phash(icon64)

            payload = json.dumps({
                'install_id':    self._install_id,
                'phash':         phash,
                'crop_png_b64':  crop_b64,
                'item_name':     item_name,
                'wrong_name':    wrong_name,
                'confirmed':     confirmed,
                'warp_version':  WARP_VERSION,
                'timestamp':     time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }, ensure_ascii=False).encode('utf-8')

            import urllib.request

            def _post() -> dict:
                req = urllib.request.Request(
                    f'{self._url}/contribute',
                    data=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent':   f'WARP/{WARP_VERSION}',
                    },
                    method='POST',
                )
                with urllib.request.urlopen(req, timeout=CONTRIBUTE_TIMEOUT) as resp:
                    return json.loads(resp.read().decode('utf-8'))

            # One retry — Render free tier needs ~50 s to cold-start
            try:
                result = _post()
            except Exception as first_err:
                log.debug(f'WARPSync: contribution attempt 1 failed ({first_err}), retrying...')
                time.sleep(3)
                result = _post()   # raises on second failure → caught below

            success = result.get('ok', False)
            if success:
                self._increment_rate_limit()
                log.debug(f'WARPSync: contribution accepted id={result.get("contribution_id")}')
            else:
                log.warning(f'WARPSync: contribution rejected: {result.get("error")}')

            if on_done:
                on_done(success)

        except Exception as e:
            log.warning(f'WARPSync: contribution failed: {e}')
            if on_done:
                on_done(False)

    # ── Rate limiting ──────────────────────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        """Return True if we're below the daily contribution limit."""
        try:
            p = self._resolve_path(RATE_LIMIT_FILE)
            if not p.exists():
                return True
            data = json.loads(p.read_text())
            today = str(date.today())
            if data.get('date') != today:
                return True
            return data.get('count', 0) < MAX_CONTRIBUTIONS_PER_DAY
        except Exception:
            return True

    def _increment_rate_limit(self) -> None:
        try:
            p = self._resolve_path(RATE_LIMIT_FILE)
            p.parent.mkdir(parents=True, exist_ok=True)
            today = str(date.today())
            data = {}
            if p.exists():
                data = json.loads(p.read_text())
            if data.get('date') != today:
                data = {'date': today, 'count': 0}
            data['count'] = data.get('count', 0) + 1
            p.write_text(json.dumps(data))
        except Exception:
            pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_or_create_install_id(self) -> str:
        p = self._resolve_path(INSTALL_ID_FILE)
        if p.exists():
            try:
                return p.read_text().strip()
            except Exception:
                pass
        install_id = str(uuid.uuid4())
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(install_id)
        except Exception:
            pass
        return install_id

    def _load_config_url(self) -> str | None:
        try:
            p = self._resolve_path(Path('warp') / 'knowledge' / 'config.json')
            if p.exists():
                return json.loads(p.read_text()).get('backend_url')
        except Exception:
            pass
        return None

    def _resolve_path(self, rel: Path) -> Path:
        """Resolve a relative path from the SETS-WARP root."""
        base = Path(__file__).resolve().parent
        for _ in range(6):
            if (base / 'pyproject.toml').exists():
                return base / rel
            base = base.parent
        return Path(rel)


# ── pHash helper (standalone, mirrors icon_matcher) ───────────────────────────

def _compute_phash(icon64_bgr: np.ndarray) -> str:
    """
    Compute 64-bit perceptual hash of a 64×64 BGR icon.
    Returns lowercase hex string (16 chars).
    """
    import cv2
    gray  = cv2.cvtColor(icon64_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (32, 32))
    dct   = np.fft.rfft2(small)
    low   = np.abs(dct[:8, :8]).flatten()
    mean  = low.mean()
    bits  = (low > mean).astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return format(h & 0xFFFFFFFFFFFFFFFF, '016x')
