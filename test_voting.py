#!/usr/bin/env python3
"""
test_voting.py — stress test: send ALL confirmed crops to the backend as N fake users.

Each crop is sent once per fake user (different install_id), so the voting system
receives N votes for each phash. Run admin_merge.py afterwards to see results.

Usage:
    python test_voting.py [--url URL] [--users 5] [--workers 20] [--delay 0.0]
"""

import argparse
import base64
import json
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import urllib.request
import urllib.error

BACKEND_URL     = 'https://sets-warp-backend.onrender.com'
TRAINING_DATA   = Path('warp/training_data')
ANNOTATIONS     = TRAINING_DATA / 'annotations.json'
CONTRIBUTE_TIMEOUT = 60


def _compute_phash(icon64_bgr: np.ndarray) -> str:
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


def send_contribution(url: str, install_id: str, crop_bgr: np.ndarray,
                      item_name: str) -> dict:
    ok, buf = cv2.imencode('.png', crop_bgr)
    if not ok:
        raise ValueError('imencode failed')
    crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    icon64   = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    phash    = _compute_phash(icon64)

    payload = json.dumps({
        'install_id':   install_id,
        'phash':        phash,
        'crop_png_b64': crop_b64,
        'item_name':    item_name,
        'wrong_name':   '',
        'confirmed':    True,
        'warp_version': 'test-stress',
        'timestamp':    time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }, ensure_ascii=False).encode('utf-8')

    req = urllib.request.Request(
        f'{url}/contribute',
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'User-Agent':   'WARP/test-voting',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=CONTRIBUTE_TIMEOUT) as resp:
        return json.loads(resp.read().decode('utf-8'))


def load_all_crops() -> list[tuple[str, np.ndarray]]:
    """Load all confirmed crops from training_data (deduped by crop file)."""
    data   = json.loads(ANNOTATIONS.read_text(encoding='utf-8'))
    seen   = set()
    result = []
    for anns in data.values():
        for ann in anns:
            if ann.get('state') != 'confirmed':
                continue
            name     = ann.get('name', '').strip()
            crop_key = ann.get('crop_name') or ann.get('crop_file', '')
            if not name or not crop_key or crop_key in seen:
                continue
            crop_path = TRAINING_DATA / crop_key
            if not crop_path.exists():
                continue
            img = cv2.imread(str(crop_path))
            if img is None:
                continue
            seen.add(crop_key)
            result.append((name, img))
    return result


def worker(url: str, task_q: queue.Queue, stats: dict,
           stats_lock: threading.Lock, delay: float) -> None:
    while True:
        try:
            install_id, item_name, crop_bgr = task_q.get(timeout=2)
        except queue.Empty:
            break
        try:
            result = send_contribution(url, install_id, crop_bgr, item_name)
            ok = result.get('ok', False)
            with stats_lock:
                stats['ok' if ok else 'rejected'] += 1
        except urllib.error.HTTPError as e:
            with stats_lock:
                stats['errors'] += 1
                stats['last_error'] = f'HTTP {e.code}'
        except Exception as e:
            with stats_lock:
                stats['errors'] += 1
                stats['last_error'] = str(e)[:80]
        finally:
            task_q.task_done()
            if delay > 0:
                time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',     default=BACKEND_URL)
    parser.add_argument('--users',   type=int, default=5,
                        help='number of fake users (default 5)')
    parser.add_argument('--workers', type=int, default=10,
                        help='concurrent HTTP threads (default 10)')
    parser.add_argument('--delay',   type=float, default=0.0,
                        help='per-worker delay between requests in seconds (default 0)')
    args = parser.parse_args()

    url = args.url.rstrip('/')

    print('Loading crops from training_data…')
    crops = load_all_crops()
    if not crops:
        print('No confirmed crops found — aborting.')
        sys.exit(1)

    fake_users = [str(uuid.uuid4()) for _ in range(args.users)]
    total      = len(crops) * args.users

    print(f'Backend  : {url}')
    print(f'Crops    : {len(crops)}')
    print(f'Users    : {args.users}  (fake install_ids)')
    print(f'Total req: {total}')
    print(f'Workers  : {args.workers}')
    print()

    task_q     = queue.Queue()
    stats      = {'ok': 0, 'rejected': 0, 'errors': 0, 'last_error': ''}
    stats_lock = threading.Lock()

    # Enqueue: for each crop, one task per fake user
    for item_name, crop_bgr in crops:
        for install_id in fake_users:
            task_q.put((install_id, item_name, crop_bgr))

    # Start workers
    threads = []
    for _ in range(args.workers):
        t = threading.Thread(
            target=worker,
            args=(url, task_q, stats, stats_lock, args.delay),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Progress reporter
    t_start = time.time()
    while True:
        remaining = task_q.qsize()
        done      = total - remaining
        elapsed   = time.time() - t_start
        rate      = done / max(elapsed, 0.1)
        eta       = remaining / rate if rate > 0 else 0
        with stats_lock:
            ok, rej, err = stats['ok'], stats['rejected'], stats['errors']
            last_err     = stats['last_error']
        pct = done / total * 100 if total > 0 else 0
        print(
            f'\r{done}/{total} ({pct:.1f}%)  '
            f'ok={ok} rej={rej} err={err}  '
            f'{rate:.1f} req/s  ETA {eta:.0f}s'
            + (f'  last_err={last_err!r}' if last_err else ''),
            end='', flush=True,
        )
        if done >= total:
            break
        time.sleep(1)

    task_q.join()
    elapsed = time.time() - t_start
    print(f'\n\nDone in {elapsed:.1f}s  |  ok={stats["ok"]}  rejected={stats["rejected"]}  errors={stats["errors"]}')
    print('Run admin_merge.py to merge contributions into knowledge.json.')


if __name__ == '__main__':
    main()
