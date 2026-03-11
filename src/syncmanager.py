"""
SyncManager — detects and downloads changed/new assets from GitHub.

Uses GitHub Tree API (SHA1 + size) to find changed/missing files,
then downloads only what's needed using a bounded thread pool.

Download discipline:
  - Max 5 concurrent threads (numbered T-1..T-5)
  - 404 = permanent failure, no retry
  - Other errors retried up to MAX_RETRIES times with RETRY_DELAY_S pause
  - Stall timeout: STALL_TIMEOUT_S seconds of no data = abort attempt

Logging:
  - Terminal (stderr): one progress bar line per group, overwritten with \\r
  - Log file: only final result per file (OK / FAILED), never per-attempt noise
  - Log prefix: always includes asset group name for easy grep

Asset groups (GitHub-backed, 3 groups):
  1. Item Icons    — images/
  2. Ship Images   — ship_images/
  3. Cargo Data    — cargo/

Wiki-only groups (no GitHub mirror, downloaded on demand):
  4. Boff Abilities — wiki suffix _icon_(Federation).png
  5. Skill Icons    — wiki suffix .png
"""

from __future__ import annotations

import hashlib
import queue
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable
from urllib.parse import quote_plus, unquote_plus

from .constants import GITHUB_CACHE_URL, WIKI_IMAGE_URL
from .setsdebug import log

GITHUB_API_TREE     = 'https://api.github.com/repos/STOCD/SETS-Data/git/trees/main?recursive=1'
TREE_CACHE_FILENAME = 'github_tree_cache.json'
TREE_CACHE_MAX_AGE  = 60 * 60   # 1 hour

MAX_RETRIES     = 1
RETRY_DELAY_S   = 3
STALL_TIMEOUT_S = 10
MAX_THREADS     = 5

BAR_WIDTH = 20   # characters for the progress bar fill

ASSET_GROUPS = [
    ('Item Icons',  'images/',      'icon'),
    ('Ship Images', 'ship_images/', 'ship'),
    ('Cargo Data',  'cargo/',       'cargo'),
]


# ---------------------------------------------------------------------------
# Terminal progress bar
# ---------------------------------------------------------------------------

class _TermProgress:
    """
    Renders a single overwritten line on stderr for one asset group.

    While active, normal log.info/warning output to stderr is suppressed
    (written to file only) so the \\r line is not torn by interleaved text.
    After finish() the suppression is lifted and a final newline is printed.

    Thread-safe: update() may be called from any thread.
    """

    # class-level flag: when True, setsdebug._write skips the stderr print
    _suppress_stderr = False
    _suppress_lock   = Lock()

    def __init__(self, label: str, total: int):
        self._label   = f'[{label:<13}]'
        self._total   = max(total, 1)
        self._current = 0
        self._current_file = ''
        self._lock    = Lock()
        self._started = False

    def _bar(self, done: int) -> str:
        filled = int(BAR_WIDTH * done / self._total)
        empty  = BAR_WIDTH - filled
        return '█' * filled + '░' * empty

    def _render(self, current: int, suffix: str) -> str:
        pct  = self._bar(current)
        line = f'{self._label}  {pct}  {current}/{self._total}  {suffix}'
        try:
            width = max(40, __import__('shutil').get_terminal_size().columns - 1)
        except Exception:
            width = 119
        if len(line) < width:
            line = line + ' ' * (width - len(line))
        else:
            line = line[:width]
        return '\r' + line

    def start(self):
        with _TermProgress._suppress_lock:
            _TermProgress._suppress_stderr = True
        line = self._render(0, 'starting…')
        sys.stderr.write(line)
        sys.stderr.flush()
        self._started = True

    def update(self, current: int, filename: str = ''):
        with self._lock:
            self._current      = current
            self._current_file = filename
        # truncate filename so line stays on one terminal row
        short = filename[:40] + '…' if len(filename) > 41 else filename
        line  = self._render(current, short)
        with _TermProgress._suppress_lock:
            sys.stderr.write(line)
            sys.stderr.flush()

    def finish(self, summary: str):
        """Print final line (no \\r overwrite), re-enable stderr logging."""
        line = self._render(self._total, summary)
        with _TermProgress._suppress_lock:
            # final line: overwrite progress, then newline
            sys.stderr.write(line + '\n')
            sys.stderr.flush()
            _TermProgress._suppress_stderr = False


# Patch setsdebug to honour suppression flag
def _patch_setsdebug():
    import src.setsdebug as _sd
    _orig_write = _sd._write

    def _patched_write(level, msg):
        # Always write to file (original behaviour handles that).
        # Skip stderr print while terminal progress bar is active.
        if _TermProgress._suppress_stderr:
            # Write to file only — replicate the file-write path directly
            from datetime import datetime
            import threading, os
            ts  = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            tid = threading.current_thread().name
            line = f'{ts}  [{level}]  [{tid}]  {msg}'
            if _sd._file_ok and _sd._fh:
                with _sd._lock:
                    try:
                        _sd._fh.write(line + '\n')
                        _sd._fh.flush()
                        os.fsync(_sd._fh.fileno())
                    except Exception:
                        pass
        else:
            _orig_write(level, msg)

    _sd._write = _patched_write

_patch_setsdebug()


# ---------------------------------------------------------------------------
# SHA helpers
# ---------------------------------------------------------------------------

def _git_sha1(filepath: Path) -> str | None:
    try:
        data = filepath.read_bytes()
        header = f'blob {len(data)}\0'.encode()
        return hashlib.sha1(header + data).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# GitHub tree fetch / cache
# ---------------------------------------------------------------------------

def _fetch_github_tree(session) -> list[dict] | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(GITHUB_API_TREE, timeout=(10, STALL_TIMEOUT_S))
            if resp.ok:
                data  = resp.json()
                blobs = [e for e in data.get('tree', []) if e['type'] == 'blob']
                log.info(f'SyncManager: tree fetched — {len(blobs)} files')
                return blobs
            if resp.status_code == 404:
                log.warning('SyncManager: tree 404 — repo not found?')
                return None
            error = f'HTTP {resp.status_code}'
        except Exception as e:
            error = str(e)
        log.warning(f'SyncManager: tree attempt {attempt}/{MAX_RETRIES} — {error}')
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_S)
    return None


def _load_tree_cache(cache_path: Path) -> list[dict] | None:
    import json
    if not cache_path.exists():
        return None
    age = time.time() - cache_path.stat().st_mtime
    if age > TREE_CACHE_MAX_AGE:
        log.info(f'SyncManager: tree cache {age/60:.0f}min old — refreshing')
        return None
    try:
        data = json.loads(cache_path.read_text(encoding='utf-8'))
        log.info(f'SyncManager: tree cache hit ({age/60:.0f}min old, {len(data)} files)')
        return data
    except Exception:
        return None


def _save_tree_cache(cache_path: Path, tree: list[dict]):
    import json
    try:
        cache_path.write_text(json.dumps(tree), encoding='utf-8')
    except Exception as e:
        log.warning(f'SyncManager: could not save tree cache: {e}')


# ---------------------------------------------------------------------------
# SyncManager
# ---------------------------------------------------------------------------

class SyncManager:

    def __init__(self, images_dir, ship_images_dir, cargo_dir, cache_dir, downloader):
        self._images_dir       = Path(images_dir)
        self._ship_images_dir  = Path(ship_images_dir)
        self._cargo_dir        = Path(cargo_dir)
        self._cache_dir        = Path(cache_dir)
        self._downloader       = downloader
        self._tree_cache_path  = Path(cache_dir) / TREE_CACHE_FILENAME

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, on_progress: Callable[[str, int, int], None] | None = None) -> dict:
        prog = on_progress or (lambda t, c, n: None)
        prog('Checking for updates…', 0, 0)

        session = self._downloader._session

        tree = _load_tree_cache(self._tree_cache_path)
        if tree is None:
            prog('Fetching update manifest…', 0, 0)
            tree = _fetch_github_tree(session)
            if tree is None:
                log.warning('SyncManager: cannot reach GitHub — skipping sync')
                prog('Update check failed (offline?)', 0, 0)
                return {'checked': 0, 'updated': 0, 'failed': 0, 'cargo_updated': False}
            _save_tree_cache(self._tree_cache_path, tree)

        total_updated     = 0
        total_failed      = 0
        cargo_was_updated = False

        for (label, prefix, type_tag) in ASSET_GROUPS:
            entries   = [e for e in tree if e['path'].startswith(prefix)]
            to_update = self._diff_group(entries, type_tag)
            count     = len(to_update)

            log.info(f'SyncManager [{label}]: {count}/{len(entries)} need download')

            if count == 0:
                prog(f'{label}: up to date', 0, 0)
                # still show a quick "up to date" line in terminal
                sys.stderr.write(
                    f'\r[{label:<13}]  {"█" * BAR_WIDTH}  {len(entries)}/{len(entries)}'
                    f'  up to date\n')
                sys.stderr.flush()
                continue

            # ── terminal progress bar for this group ───────────────────────
            tprog = _TermProgress(label, count)
            tprog.start()
            prog(f'{label}', 0, count)

            job_q    = queue.Queue()
            for item in to_update:
                job_q.put(item)

            counter  = [0]
            n_failed = [0]
            lock     = Lock()

            def _worker(thread_num: int, _label=label, _count=count, _type=type_tag):
                while True:
                    try:
                        entry, local_path = job_q.get_nowait()
                    except queue.Empty:
                        return

                    fname = entry['path'].split('/', 1)[1]
                    ok, source, attempts = self._download_with_result(
                        entry, local_path, _type, session)

                    with lock:
                        counter[0] += 1
                        if not ok:
                            n_failed[0] += 1
                        c = counter[0]

                    # ── log file: final result only ────────────────────────
                    attempt_word = f'{attempts} attempt{"s" if attempts > 1 else ""}'
                    if ok:
                        log.info(
                            f'SyncManager [{_label}] T-{thread_num}: '
                            f'OK {source} ({attempt_word}) — {fname}')
                    else:
                        log.warning(
                            f'SyncManager [{_label}] T-{thread_num}: '
                            f'FAILED ({attempt_word}) — {fname}')

                    # ── terminal: update progress line ─────────────────────
                    status = f'T-{thread_num}: {fname}'
                    tprog.update(c, status)
                    prog(f'{_label}', c, _count)
                    job_q.task_done()

            n_threads = min(MAX_THREADS, count)
            threads   = [
                Thread(target=_worker, args=(i + 1,), name=f'sync-T{i+1}')
                for i in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            updated = counter[0] - n_failed[0]
            total_updated += updated
            total_failed  += n_failed[0]
            if type_tag == 'cargo' and updated > 0:
                cargo_was_updated = True

            summary = f'{updated} updated'
            if n_failed[0]:
                summary += f', {n_failed[0]} FAILED'
            log.info(f'SyncManager [{label}]: done — {summary}')
            tprog.finish(summary)
            prog(f'{label}: {summary}', count, count)

        report = {
            'checked':       len(tree),
            'updated':       total_updated,
            'failed':        total_failed,
            'cargo_updated': cargo_was_updated,
        }
        log.info(f'SyncManager: complete — {report}')

        if total_updated == 0 and total_failed == 0:
            prog('All assets up to date', 0, 0)
        else:
            prog(f'Sync done: {total_updated} updated, {total_failed} failed', 0, 0)

        return report

    # -----------------------------------------------------------------------
    # Public: wiki-only groups (Boff Abilities, Skill Icons)
    # -----------------------------------------------------------------------

    def download_wiki_group(
            self,
            label: str,
            names: list[str],
            suffix: str,
            on_splash: 'Callable[[str,int,int],None] | None' = None,
    ) -> dict[str, int]:
        """
        Download a list of items from stowiki only (no GitHub mirror).
        Uses _TermProgress for terminal output.
        Returns {name: timestamp} failed dict.
        """
        from urllib.parse import quote_plus as _qp
        if not names:
            return {}

        prog = on_splash or (lambda t, c, n: None)
        tprog = _TermProgress(label, len(names))
        tprog.start()
        prog(label, 0, len(names))

        counter  = [0]
        n_failed = [0]
        lock     = Lock()
        failed   = {}

        session = self._downloader._session
        images_dir = self._images_dir

        def _worker(name: str):
            filename = _qp(name) + '.png'
            local_path = images_dir / filename
            url = WIKI_IMAGE_URL + name.replace(' ', '_') + suffix
            data, _ = self._fetch(session, url, min_size=10)
            ok = False
            if data is not None:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(data)
                ok = True
            with lock:
                counter[0] += 1
                if not ok:
                    n_failed[0] += 1
                    failed[name] = int(__import__('time').time())
                c = counter[0]
            tprog.update(c)
            prog(label, c, len(names))

        import queue as _queue
        q = _queue.Queue()
        for n in names:
            q.put(n)

        def _thread_worker():
            while True:
                try:
                    name = q.get_nowait()
                except _queue.Empty:
                    return
                _worker(name)
                q.task_done()

        n_threads = min(MAX_THREADS, len(names))
        threads = [Thread(target=_thread_worker, name=f'wiki-T{i+1}') for i in range(n_threads)]
        for t in threads: t.start()
        for t in threads: t.join()

        updated = len(names) - n_failed[0]
        summary = f'{updated} updated' + (f', {n_failed[0]} FAILED' if n_failed[0] else '')
        log.info(f'SyncManager [{label}]: done — {summary}')
        tprog.finish(summary)
        prog(f'{label}: {summary}', len(names), len(names))
        return failed

    def download_one(self, name: str, type_tag: str) -> bool:
        """
        Download a single asset on-demand (e.g. ship image clicked in UI).
        type_tag: 'ship' | 'icon'
        Returns True on success.
        """
        session = self._downloader._session
        if type_tag == 'ship':
            # Construct fake entry for _download_with_result
            filename = quote_plus(quote_plus(name))
            entry = {
                'path': f'ship_images/{filename}',
                'sha':  '',
                'size': -1,
            }
            local_path = self._ship_images_dir / quote_plus(name)
        else:
            filename = quote_plus(name) + '.png'
            entry = {
                'path': f'images/{filename}',
                'sha':  '',
                'size': -1,
            }
            local_path = self._images_dir / filename

        ok, source, _ = self._download_with_result(entry, local_path, type_tag, session)
        if ok:
            log.debug(f'SyncManager.download_one: OK ({source}) — {name!r}')
        else:
            log.warning(f'SyncManager.download_one: FAILED — {name!r}')
        return ok

    # -----------------------------------------------------------------------
    # Diff
    # -----------------------------------------------------------------------

    def _diff_group(self, entries, type_tag) -> list[tuple]:
        result = []
        for entry in entries:
            lp = self._local_path(entry['path'], type_tag)
            if lp and self._needs_update(lp, entry):
                result.append((entry, lp))
        return result

    def _local_path(self, github_path: str, type_tag: str) -> Path | None:
        filename = github_path.split('/', 1)[1]
        if type_tag == 'icon':  return self._images_dir      / filename
        if type_tag == 'ship':  return self._ship_images_dir / filename
        if type_tag == 'cargo': return self._cargo_dir       / filename
        return None

    def _needs_update(self, local_path: Path, entry: dict) -> bool:
        if not local_path.exists():
            return True
        local_size  = local_path.stat().st_size
        remote_size = entry.get('size', -1)
        if remote_size >= 0 and local_size != remote_size:
            return True
        return _git_sha1(local_path) != entry['sha']

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------

    def _download_with_result(
            self, entry: dict, local_path: Path,
            type_tag: str, session) -> tuple[bool, str, int]:
        """Returns (success, source_label, total_attempts)."""
        url  = f'{GITHUB_CACHE_URL}/{entry["path"]}'
        data, attempts = self._fetch(session, url, min_size=10)
        if data is not None:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            return True, 'github', attempts

        if type_tag == 'ship':
            filename  = entry['path'].split('/', 1)[1]
            name      = unquote_plus(unquote_plus(filename))
            wiki_url  = WIKI_IMAGE_URL + quote_plus(name.replace(' ', '_'), safe='._-')
            wdata, wa = self._fetch(session, wiki_url, min_size=100)
            total     = attempts + wa
            if wdata is not None:
                local_path.write_bytes(wdata)
                return True, 'wiki', total
            return False, 'all', total

        return False, 'github', attempts

    def _fetch(self, session, url: str, min_size: int = 10) -> tuple[bytes | None, int]:
        """
        Fetch with retry. 404 = instant permanent failure (no retry).
        Returns (data_or_None, attempts_made).
        Transient errors logged to file only (stderr is suppressed during progress bar).
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = session.get(url, timeout=(10, STALL_TIMEOUT_S), stream=False)
                if resp.ok and len(resp.content) >= min_size:
                    return resp.content, attempt
                if resp.status_code == 404:
                    return None, attempt   # permanent — don't retry
                error = f'HTTP {resp.status_code}'
            except Exception as e:
                error = str(e)
            log.warning(f'SyncManager: attempt {attempt}/{MAX_RETRIES} — {url} → {error}')
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
        return None, MAX_RETRIES
