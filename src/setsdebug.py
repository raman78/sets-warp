"""
Crash-safe debug logger.
- Writes to ~/sets_warp_debug.log (fsynced after every line)
- Also prints to stderr (visible in terminal even on crash)
"""
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import os as _os
_log_path = Path(_os.environ.get('SETS_DIR') or Path(__file__).parent.parent) / 'sets_warp_debug.log'
_lock = threading.Lock()

try:
    _fh = open(_log_path, 'w', buffering=1)
    _file_ok = True
except Exception as e:
    _fh = None
    _file_ok = False
    print(f'[SETS-LOG] WARNING: cannot open log file {_log_path}: {e}', flush=True)


def _write(level, msg):
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    tid = threading.current_thread().name
    line = f'{ts}  [{level}]  [{tid}]  {msg}'
    # Print INFO/WARN/ERROR to stderr — suppress DEBUG from terminal
    if level != 'DEBUG':
        print(f'[SETS] {line}', file=sys.stderr, flush=True)
    # Always write everything (including DEBUG) to log file
    if _file_ok and _fh:
        with _lock:
            try:
                _fh.write(line + '\n')
                _fh.flush()
                os.fsync(_fh.fileno())
            except Exception:
                pass


class _Log:
    def info(self, msg):    _write('INFO ', msg)
    def debug(self, msg):   _write('DEBUG', msg)
    def warning(self, msg): _write('WARN ', msg)
    def error(self, msg):   _write('ERROR', msg)


log = _Log()
log.info(f'=== setsdebug initialized  pid={os.getpid()}  log={_log_path} ===')
