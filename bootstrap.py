"""
bootstrap.py — launcher for SETS

Strategy (in order):
  1. If already running inside our .venv → launch main.py directly
  2. If .venv exists but we're not in it → relaunch with venv Python
  3. .venv does not exist → show installer GUI, then:
     a. Download portable Python into .python/  (~65 MB, no root, no compile)
     b. Create .venv with that Python
     c. pip install all dependencies
     d. Relaunch with venv Python

Portable Python: python-build-standalone (astral-sh/python-build-standalone)
Requires only: Python 3.11+ stdlib (tomllib, tkinter, urllib, subprocess)
"""

import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import tomllib
import urllib.request
import zipfile
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
# SETS_DIR is set by SETS.sh — always use it when available so ROOT is correct
# even if Python caches a .pyc from a different location (e.g. Trash)
ROOT        = Path(os.environ['SETS_DIR']).resolve() if 'SETS_DIR' in os.environ else Path(__file__).resolve().parent
VENV_DIR    = ROOT / ".venv"
PYTHON_DIR  = ROOT / ".python"
PYPROJECT   = ROOT / "pyproject.toml"
SETUP_LOG   = ROOT / "sets_setup.log"

IS_WINDOWS  = sys.platform == "win32"
IS_MAC      = sys.platform == "darwin"
ARCH        = platform.machine().lower()

# ── portable Python config ─────────────────────────────────────────────────────
PBS_BASE    = "https://github.com/astral-sh/python-build-standalone/releases/download"
PBS_TAG     = "20250317"
PBS_VERSION = "3.13.2"

PBS_TARGETS = {
    ("linux",   "x86_64"):  f"cpython-{PBS_VERSION}+{PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz",
    ("linux",   "aarch64"): f"cpython-{PBS_VERSION}+{PBS_TAG}-aarch64-unknown-linux-gnu-install_only.tar.gz",
    ("darwin",  "x86_64"):  f"cpython-{PBS_VERSION}+{PBS_TAG}-x86_64-apple-darwin-install_only.tar.gz",
    ("darwin",  "arm64"):   f"cpython-{PBS_VERSION}+{PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz",
    ("windows", "x86_64"):  f"cpython-{PBS_VERSION}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz",
}

# ── version guard ─────────────────────────────────────────────────────────────
# If user forces reinstall with --reinstall flag, wipe venv and .python
if '--reinstall' in sys.argv:
    import shutil
    print("[bootstrap] --reinstall: removing .venv and .python ...", flush=True)
    for d in (VENV_DIR, PYTHON_DIR):
        if d.exists():
            shutil.rmtree(d)
            print(f"[bootstrap]   removed {d}", flush=True)
    sys.argv.remove('--reinstall')

# --repair flag: skip to venv health-check / repair flow, then exit (don't relaunch)
_REPAIR_ONLY = '--repair' in sys.argv
if _REPAIR_ONLY:
    sys.argv.remove('--repair')


def _platform_key():
    if IS_WINDOWS:
        return ("windows", "x86_64")
    sys_name = "darwin" if IS_MAC else "linux"
    arch = ("arm64" if ARCH in ("arm64", "aarch64") and IS_MAC
            else "aarch64" if ARCH == "aarch64"
            else "x86_64")
    return (sys_name, arch)


def portable_python_exe() -> Path:
    """Expected path to portable Python binary after extraction."""
    if IS_WINDOWS:
        return PYTHON_DIR / "python" / "python.exe"
    return PYTHON_DIR / "python" / "bin" / f"python{PBS_VERSION[:4]}"


def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def running_in_our_venv() -> bool:
    """True only when sys.executable IS our venv Python (handles symlinks)."""
    try:
        exe     = Path(sys.executable).resolve()
        venv_py = venv_python().resolve()
        if exe == venv_py:
            return True
        # sys.executable may be python3.13 while venv_python() points to 'python'
        # — both resolve to the same directory, accept any python* in venv bin/
        return (exe.parent == venv_py.parent and
                exe.name.startswith('python'))
    except Exception:
        return False


# ── parse pyproject.toml ───────────────────────────────────────────────────────

def parse_pyproject():
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("dependencies", [])


# ── venv verification ──────────────────────────────────────────────────────────

def _parse_specifier(dep: str) -> tuple[str, list[tuple[str, tuple]]]:
    """
    Parse a PEP-508 dependency string into (package_name, [(op, version_tuple), ...]).
    e.g. "PySide6>=6.7,<6.10" → ("pyside6", [(">=", (6,7)), ("<", (6,10))])
    """
    # split name from specifiers
    m = re.match(r'^([A-Za-z0-9_\-\.]+)\s*(.*)', dep.strip())
    if not m:
        return dep.strip().lower(), []
    name = m.group(1).lower().replace('-', '_').replace('.', '_')
    spec_str = m.group(2).strip()
    specs = []
    for part in re.findall(r'([><=!]+)\s*([\d\.]+)', spec_str):
        op, ver = part
        specs.append((op, tuple(int(x) for x in ver.split('.'))))
    return name, specs


def _version_tuple(ver_str: str) -> tuple:
    try:
        return tuple(int(x) for x in re.split(r'[.\-]', ver_str)[:3])
    except Exception:
        return (0,)


def _satisfies(installed_ver: str, specs: list) -> bool:
    """Check if installed version satisfies all specifiers."""
    if not specs:
        return True  # no version constraint → any version is fine
    iv = _version_tuple(installed_ver)
    ops = {
        '>=': lambda a, b: a >= b,
        '<=': lambda a, b: a <= b,
        '>':  lambda a, b: a > b,
        '<':  lambda a, b: a < b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
    }
    for op, req_ver in specs:
        fn = ops.get(op)
        if fn is None:
            continue
        # pad tuples to same length
        maxlen = max(len(iv), len(req_ver))
        a = iv + (0,) * (maxlen - len(iv))
        b = req_ver + (0,) * (maxlen - len(req_ver))
        if not fn(a, b):
            return False
    return True


def check_venv_health(on_line) -> list[str]:
    """
    Checks venv Python works and all required packages are installed with
    correct versions. Returns list of dep strings that need (re)installing.
    Empty list means everything is healthy.
    """
    py = str(venv_python())

    # 1 — does venv Python run at all?
    on_line("  Checking venv Python...")
    try:
        r = subprocess.run([py, '--version'], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            on_line(f"  WARN venv Python failed ({r.stderr.strip()}) — will reinstall all")
            return parse_pyproject()
        on_line(f"  OK  {r.stdout.strip()}")
    except Exception as exc:
        on_line(f"  WARN venv Python not usable ({exc}) — will reinstall all")
        return parse_pyproject()

    # 2 — get installed packages via pip list
    on_line("  Checking installed packages...")
    try:
        r = subprocess.run(
            [py, '-m', 'pip', 'list', '--format=json'],
            capture_output=True, text=True, timeout=30)
        import json
        installed = {
            pkg['name'].lower().replace('-', '_').replace('.', '_'): pkg['version']
            for pkg in json.loads(r.stdout)
        }
    except Exception as exc:
        on_line(f"  WARN could not query pip ({exc}) — will reinstall all")
        return parse_pyproject()

    # 3 — compare against pyproject.toml
    deps = parse_pyproject()
    broken = []
    for dep in deps:
        name, specs = _parse_specifier(dep)
        ver = installed.get(name)
        if ver is None:
            on_line(f"  MISSING  {dep}")
            broken.append(dep)
        elif not _satisfies(ver, specs):
            on_line(f"  WRONG VERSION  {name}=={ver}  (required: {dep})")
            broken.append(dep)
        else:
            on_line(f"  OK  {name}=={ver}")

    return broken


# ── portable Python download ───────────────────────────────────────────────────

def _find_portable_python_exe() -> Path | None:
    """Scan .python/ for a usable Python executable after extraction."""
    expected = portable_python_exe()
    if expected.exists():
        return expected
    # Fallback scan
    for candidate in sorted(PYTHON_DIR.rglob("python3*")):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    if IS_WINDOWS:
        for candidate in sorted(PYTHON_DIR.rglob("python.exe")):
            if candidate.is_file():
                return candidate
    return None


def download_portable_python(on_line) -> str:
    """Download + extract portable Python. Returns path to executable."""
    key = _platform_key()
    filename = PBS_TARGETS.get(key)
    if not filename:
        raise RuntimeError(
            f"No portable Python binary available for platform {key}.\n"
            "Please install Python 3.13+ manually: https://www.python.org/downloads/")

    url = f"{PBS_BASE}/{PBS_TAG}/{filename}"
    on_line(f"  Downloading portable Python {PBS_VERSION}...")

    PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(tmp_fd)

    last_pct = [-1]
    def _progress(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            if pct != last_pct[0]:
                last_pct[0] = pct
                on_line(f"  Downloading... {pct}%", replace_last=True)

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        on_line(f"  Extracting...")

        if filename.endswith(".tar.gz"):
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(PYTHON_DIR)
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(PYTHON_DIR)

        exe = _find_portable_python_exe()
        if exe is None:
            raise RuntimeError(
                f"Could not locate Python executable after extraction in {PYTHON_DIR}")

        if not IS_WINDOWS:
            exe.chmod(0o755)

        on_line(f"  OK portable Python ready: {exe}")
        return str(exe)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── venv + pip ─────────────────────────────────────────────────────────────────

def create_venv(python_exe: str, on_line):
    on_line("  Creating virtual environment...")
    r = subprocess.run(
        [python_exe, "-m", "venv", str(VENV_DIR)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"venv creation failed:\n{r.stderr.strip()}")
    on_line("  OK virtual environment created")


def _run_pip(args: list, on_line, label: str):
    """Run pip command, streaming output line by line so UI stays responsive."""
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    # Read output so pip never blocks on a full pipe
    for line in proc.stdout:
        line = line.strip()
        if line:
            on_line(f"    {line}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"pip failed for {label} (exit {proc.returncode})")


def install_dependencies(on_line, deps: list | None = None, force: bool = False):
    import time
    py = str(venv_python())
    on_line("  Upgrading pip...")
    _run_pip([py, "-m", "pip", "install", "--upgrade", "pip"], on_line, "pip")

    if deps is None:
        deps = parse_pyproject()
    if not deps:
        on_line("  All packages already up to date.")
        return

    # ── Split: torch/torchvision need CPU-only index, rest use PyPI ───────────
    TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
    TORCH_PKGS      = {'torch', 'torchvision', 'torchaudio'}

    torch_deps  = [d for d in deps if re.split(r'[><=!\[]', d)[0].strip().lower() in TORCH_PKGS]
    other_deps  = [d for d in deps if re.split(r'[><=!\[]', d)[0].strip().lower() not in TORCH_PKGS]

    install_start = time.monotonic()

    def _install_batch(batch: list[str], extra_args: list[str], label: str):
        if not batch:
            return
        on_line(f"  Installing {label}...")
        pip_args = [py, "-m", "pip", "install"]
        if force:
            pip_args.append("--force-reinstall")
        pip_args += extra_args + batch
        _run_pip(pip_args, on_line, label)

    # Install torch CPU-only first (separate index)
    _install_batch(
        torch_deps,
        ["--index-url", TORCH_CPU_INDEX],
        "torch (CPU-only)"
    )

    # Install everything else in one shot (pip resolves transitive deps)
    _install_batch(
        other_deps,
        [],
        f"{len(other_deps)} package(s)"
    )

    total_elapsed = time.monotonic() - install_start
    elapsed_str = (f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s"
                   if total_elapsed >= 60 else f"{int(total_elapsed)}s")
    on_line(f"  All packages installed  (total: {elapsed_str})")

    on_line("  Purging pip cache...")
    _cleanup_after_install(on_line)


# ── relaunch ───────────────────────────────────────────────────────────────────

def relaunch_in_venv():
    py = str(venv_python())
    if not Path(py).exists():
        print(f"[bootstrap] ERROR: venv Python not found at {py}", file=sys.stderr)
        sys.exit(1)
    args = [py, str(ROOT / "main.py")] + sys.argv[1:]
    if IS_WINDOWS:
        sys.exit(subprocess.run(args).returncode)
    else:
        os.execv(py, args)


# ── install orchestration ──────────────────────────────────────────────────────

def run_install(on_line, on_done, on_error, repair_only: bool = False):
    """Full install flow, runs in background thread.
    If repair_only=True, skips portable Python download and venv creation,
    only fixes broken dependencies."""
    try:
        if repair_only:
            on_line("=== SETS Dependency Repair ===")
            on_line("")
            on_line("Checking installed packages...")
            broken = check_venv_health(on_line)
            on_line("")
            if not broken:
                on_line("All packages OK — no action needed.")
            else:
                on_line(f"Fixing {len(broken)} package(s)...")
                install_dependencies(on_line, deps=broken, force=True)
                on_line("")
                on_line("Repair complete!")
            on_done()
            return

        on_line(f"=== SETS-WARP First-Time Setup ===")
        on_line(f"ROOT: {ROOT}")
        on_line("")

        # Step 1 — portable Python
        exe = _find_portable_python_exe()
        if exe is not None:
            on_line(f"OK Portable Python already present: {exe}\n")
            python_exe = str(exe)
        else:
            on_line("Step 1/3  Downloading portable Python (~65 MB, one-time)...")
            python_exe = download_portable_python(on_line)
            on_line("")

        # Step 2 — venv
        on_line("Step 2/3  Creating virtual environment...")
        create_venv(python_exe, on_line)
        on_line("")

        # Step 3 — dependencies
        on_line("Step 3/4  Installing dependencies...")
        on_line("  Removing obsolete packages...")
        _uninstall_obsolete_packages(on_line)
        on_line("  Cleaning stale .venv cache...")
        _cleanup_venv_pycache(on_line)
        install_dependencies(on_line)
        on_line("")

        # Step 4 — WARP data
        on_line("Step 4/4  Preparing WARP data...")
        _setup_warp_dirs(on_line)
        _run_warp_scraper(on_line)
        on_line("")

        on_line("Setup complete!  Starting SETS-WARP...\n")
        on_done()

    except Exception as exc:
        on_error(str(exc))



# ── tkinter GUI ────────────────────────────────────────────────────────────────

def _setup_log_writer():
    """Returns a callable that appends a timestamped line to sets_setup.log."""
    import datetime
    try:
        log_file = open(SETUP_LOG, 'a', encoding='utf-8', buffering=1)
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"SETS Setup  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*60}\n")
    except OSError:
        return lambda msg: None  # silently ignore if can't write

    def write_log(msg: str):
        try:
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            log_file.write(f"[{ts}] {msg}\n")
        except OSError:
            pass

    return write_log


def run_with_tkinter_gui():
    import tkinter as tk

    BG, FG, ACCENT = "#1a1a1a", "#eeeeee", "#c59129"

    root = tk.Tk()
    root.title("SETS-WARP — Setup")
    root.configure(bg=BG)
    root.resizable(True, True)

    W, H = 700, 520
    root.geometry(f"{W}x{H}")
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

    # ── top: logo (fixed height) ──────────────────────────────────────────────
    top = tk.Frame(root, bg=BG)
    top.pack(side=tk.TOP, fill=tk.X)

    try:
        img = tk.PhotoImage(file=str(ROOT / "local" / "sets_loading.png"))
        # Scale image down if it's too tall (PhotoImage supports subsample)
        # Limit to 160px height
        orig_h = img.height()
        if orig_h > 160:
            factor = max(1, orig_h // 160)
            img = img.subsample(factor, factor)
        tk.Label(top, image=img, bg=BG).pack(pady=(12, 4))
        top._img = img  # prevent GC
    except Exception:
        tk.Label(top, text="SETS", bg=BG, fg=ACCENT,
                 font=("Helvetica", 22, "bold")).pack(pady=12)

    # ── middle: log text (expands) ────────────────────────────────────────────
    mid = tk.Frame(root, bg=BG)
    mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 4))

    sb = tk.Scrollbar(mid)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    log = tk.Text(mid, bg="#111", fg=FG, font=("Courier", 10),
                  wrap=tk.WORD, yscrollcommand=sb.set,
                  state=tk.DISABLED, relief=tk.FLAT, height=12)
    log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.config(command=log.yview)

    # ── bottom: status label ──────────────────────────────────────────────────
    status_var = tk.StringVar(value="Starting setup...")
    tk.Label(root, textvariable=status_var, bg=BG, fg=ACCENT,
             font=("Helvetica", 10)).pack(side=tk.BOTTOM, pady=(0, 10))

    _prev_was_replace = [False]
    _write_log = _setup_log_writer()

    def _append(msg: str, replace_last: bool = False):
        log.config(state=tk.NORMAL)
        if replace_last and _prev_was_replace[0]:
            log.delete("end-2l", "end-1l")
        log.insert(tk.END, msg + "\n")
        log.see(tk.END)
        log.config(state=tk.DISABLED)
        _prev_was_replace[0] = replace_last
        # Also show last meaningful line in status bar
        if msg.strip():
            status_var.set(msg.strip()[:80])

    def on_line(msg: str, replace_last: bool = False):
        _write_log(msg)
        root.after(0, lambda m=msg, r=replace_last: _append(m, r))

    def on_done():
        root.after(0, root.destroy)

    def on_error(msg: str):
        def _show():
            _append(f"\nERROR: {msg}\n")
            _append("Please install Python 3.13+ manually: https://www.python.org/downloads/")
            status_var.set("Setup failed — see error above")
            tk.Button(root, text="Close", bg=ACCENT, fg=BG,
                      font=("Helvetica", 11, "bold"), relief=tk.FLAT,
                      command=root.destroy).pack(pady=8)
        root.after(0, _show)

    # Start install thread AFTER mainloop is about to run, via after()
    root.after(100, lambda: threading.Thread(
        target=run_install, args=(on_line, on_done, on_error), daemon=True).start())

    root.mainloop()


# ── plain-text fallback ────────────────────────────────────────────────────────

def run_plain_text():
    _last = [None]
    _write_log = _setup_log_writer()

    def on_line(msg: str, replace_last: bool = False):
        _write_log(msg)
        if replace_last and _last[0]:
            print(f"\r{msg}          ", end="", flush=True)
        else:
            if _last[0] and not _last[0].endswith("\n"):
                print()
            print(msg, flush=True)
        _last[0] = msg

    done = threading.Event()
    errors = []

    def on_done():   done.set()
    def on_error(m): errors.append(m); done.set()

    threading.Thread(
        target=run_install, args=(on_line, on_done, on_error), daemon=True).start()
    done.wait()

    if errors:
        print(f"\nSetup failed: {errors[0]}", file=sys.stderr)
        sys.exit(1)


# ── entry point ────────────────────────────────────────────────────────────────

def _check_portable_python_version() -> bool:
    """
    Returns True if .python/ contains the expected PBS_VERSION.
    If the version file is missing or wrong, returns False → caller should wipe .python/.
    """
    version_file = PYTHON_DIR / "python" / "VERSION"
    if not version_file.exists():
        # Try the pyvenv.cfg inside the python dir as fallback
        cfg = PYTHON_DIR / "python" / "pyvenv.cfg"
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                if line.lower().startswith("version"):
                    ver = line.split("=", 1)[1].strip()
                    return ver.startswith(PBS_VERSION)
        # No version info found — check the executable name
        exe = _find_portable_python_exe()
        if exe is None:
            return False
        try:
            r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=10)
            return PBS_VERSION in r.stdout or PBS_VERSION in r.stderr
        except Exception:
            return False
    try:
        return version_file.read_text().strip().startswith(PBS_VERSION)
    except Exception:
        return False


def _uninstall_obsolete_packages(on_line):
    """
    Removes packages installed in venv that are no longer needed —
    neither listed in pyproject.toml nor a transitive dependency of any
    listed package.  Skips pip/setuptools/wheel and friends.
    """
    import json
    py = str(venv_python())
    keep_always = {'pip', 'setuptools', 'wheel', 'pkg_resources', 'distlib',
                   'platformdirs', 'filelock', 'virtualenv', 'packaging'}

    # ── Collect all installed packages ────────────────────────────────────────
    try:
        r = subprocess.run(
            [py, '-m', 'pip', 'list', '--format=json'],
            capture_output=True, text=True, timeout=30)
        installed = json.loads(r.stdout)
    except Exception as exc:
        on_line(f"  WARN cannot list packages for cleanup: {exc}")
        return

    if not installed:
        return

    # ── Build full required set: top-level + all transitive deps ──────────────
    def _norm(n): return n.lower().replace('-', '_').replace('.', '_')

    top_level = set()
    for dep in parse_pyproject():
        name, _ = _parse_specifier(dep)
        top_level.add(name)

    # Use pip show to get Requires for every installed package
    all_names = [pkg['name'] for pkg in installed]
    try:
        r = subprocess.run(
            [py, '-m', 'pip', 'show'] + all_names,
            capture_output=True, text=True, timeout=60)
        # Parse multi-record output separated by '---'
        pkg_deps: dict[str, set[str]] = {}
        current = None
        for line in r.stdout.splitlines():
            if line.startswith('Name:'):
                current = _norm(line.split(':', 1)[1].strip())
                pkg_deps[current] = set()
            elif line.startswith('Requires:') and current:
                reqs = line.split(':', 1)[1].strip()
                if reqs:
                    pkg_deps[current] = {_norm(x.strip()) for x in reqs.split(',')}
    except Exception as exc:
        on_line(f"  WARN cannot resolve transitive deps: {exc} — skipping cleanup")
        return

    # Walk dependency graph from top-level roots
    required: set[str] = set(keep_always) | top_level
    queue = list(top_level)
    while queue:
        pkg = queue.pop()
        for dep in pkg_deps.get(pkg, set()):
            if dep and dep not in required:
                required.add(dep)
                queue.append(dep)

    # ── Remove anything not reachable from required set ───────────────────────
    to_remove = [
        pkg['name'] for pkg in installed
        if _norm(pkg['name']) not in required
    ]

    if not to_remove:
        on_line("  No obsolete packages to remove.")
        return

    on_line(f"  Removing {len(to_remove)} obsolete package(s): {', '.join(to_remove)}")
    try:
        r = subprocess.run(
            [py, '-m', 'pip', 'uninstall', '-y'] + to_remove,
            capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            on_line(f"  OK removed: {', '.join(to_remove)}")
        else:
            on_line(f"  WARN uninstall failed: {r.stderr.strip()[:200]}")
    except Exception as exc:
        on_line(f"  WARN uninstall error: {exc}")


def _cleanup_venv_pycache(on_line):
    """Remove stale __pycache__ dirs from .venv before installing new packages."""
    import shutil
    removed = 0
    for pycache in VENV_DIR.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            removed += 1
        except Exception:
            pass
    if removed:
        on_line(f"  Removed {removed} __pycache__ dirs from .venv")
    else:
        on_line("  No stale __pycache__ found")


def _cleanup_after_install(on_line):
    """Post-install: purge pip cache to free disk space."""
    py = str(venv_python())
    try:
        r = subprocess.run(
            [py, '-m', 'pip', 'cache', 'purge'],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            on_line("  pip cache purged")
        else:
            on_line(f"  WARN pip cache purge: {r.stderr.strip()[:100]}")
    except Exception as exc:
        on_line(f"  WARN pip cache purge failed: {exc}")
    _setup_warp_dirs(on_line)


def _setup_warp_dirs(on_line):
    """Create WARP data directories inside the SETS-WARP installation if missing."""
    for subdir in ('warp/models', 'warp/training_data/crops', 'warp/data/icons'):
        p = ROOT / subdir
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            on_line(f"  WARN WARP dir {subdir}: {exc}")


def _run_warp_scraper(on_line):
    """
    Run warp.tools.scraper automatically after first install / repair.
    Builds warp/data/item_db.json from SETS cargo cache.
    Called once: if item_db.json already exists and is non-empty, skips.
    """
    db_path = ROOT / 'warp' / 'data' / 'item_db.json'
    if db_path.exists() and db_path.stat().st_size > 1000:
        on_line("  WARP data already present — skipping scraper")
        return

    on_line("  Building WARP item database from SETS cargo cache...")
    py = str(venv_python())
    try:
        env = {**os.environ, 'SETS_DIR': str(ROOT)}
        proc = subprocess.Popen(
            [py, '-m', 'warp.tools.scraper',
             '--cargo',       str(ROOT / '.config' / 'cargo'),
             '--sets-images', str(ROOT / '.config' / 'images'),
             '--output',      str(ROOT / 'warp' / 'data'),
             '--skip-icons',   # icons downloaded on demand later
             '--skip-vger',    # vger is CSR, skip
             '--skip-github',  # only use local cargo
             ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
            env=env,
        )
        last_info = ['']
        for line in proc.stdout:
            line = line.strip()
            if line:
                on_line(f"  {line}", replace_last=(line.startswith('INFO') and
                        last_info[0].startswith('INFO')))
                last_info[0] = line
        proc.wait()
        if proc.returncode == 0 and db_path.exists():
            import json
            db = json.loads(db_path.read_text())
            on_line(f"  WARP database ready: {len(db)} items")
        else:
            on_line("  WARN scraper exited non-zero — WARP will retry on next start")
    except Exception as exc:
        on_line(f"  WARN WARP scraper failed: {exc}")
        on_line("  WARP will retry on next application start")


def _warp_scraper_needed() -> bool:
    """True if WARP item_db needs to be (re)built."""
    db_path = ROOT / 'warp' / 'data' / 'item_db.json'
    if not db_path.exists() or db_path.stat().st_size < 1000:
        return True
    # Re-run if cargo is newer than item_db
    cargo_eq = ROOT / '.config' / 'cargo' / 'equipment.json'
    if cargo_eq.exists():
        try:
            return cargo_eq.stat().st_mtime > db_path.stat().st_mtime
        except Exception:
            pass
    return False


def _quick_check_venv() -> list[str]:
    """
    Fast silent venv health check. Returns list of broken dep strings.
    Checks both pip metadata AND actual importability of critical packages.
    Runs imports in a subprocess so any XCB/Qt crash state cannot affect us.
    """
    import json
    py = str(venv_python())

    # Is venv Python executable?
    try:
        r = subprocess.run([py, '--version'], capture_output=True, timeout=10)
        if r.returncode != 0:
            return parse_pyproject()
    except Exception:
        return parse_pyproject()

    # Get installed packages from pip metadata
    try:
        r = subprocess.run(
            [py, '-m', 'pip', 'list', '--format=json'],
            capture_output=True, text=True, timeout=30)
        installed = {
            pkg['name'].lower().replace('-', '_').replace('.', '_'): pkg['version']
            for pkg in json.loads(r.stdout)
        }
    except Exception:
        return parse_pyproject()

    broken = []
    for dep in parse_pyproject():
        name, specs = _parse_specifier(dep)
        ver = installed.get(name)
        if ver is None or not _satisfies(ver, specs):
            broken.append(dep)

    if broken:
        return broken  # metadata already shows problems, skip import checks

    # Metadata looks fine — verify critical packages actually import.
    # Use QT_QPA_PLATFORM=offscreen so PySide6 does not try to open a display.
    env_safe = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
    critical = [
        ('pyside6',  'import PySide6'),
        ('requests', 'import requests'),
        ('numpy',    'import numpy'),
        ('lxml',     'import lxml'),
    ]
    for pkg_key, import_stmt in critical:
        try:
            r = subprocess.run(
                [py, '-c', import_stmt],
                capture_output=True, timeout=15, env=env_safe)
            if r.returncode != 0:
                # map back to the full dep specifier from pyproject
                for dep in parse_pyproject():
                    if _parse_specifier(dep)[0].startswith(pkg_key):
                        broken.append(dep)
                        break
                else:
                    broken.append(pkg_key)
        except Exception:
            broken.append(pkg_key)

    return broken


def _run_repair(broken: list[str], allow_gui: bool = False):
    """Install/update broken packages.

    allow_gui=True only on genuine first-run (no venv yet) where it is safe
    to open a tkinter window.  In all other cases (venv repair, --repair flag)
    we use plain terminal output to avoid XCB/Qt assertion crashes that happen
    when tkinter tries to open a display that Qt has already claimed.
    When PySide6 is available we show a simple Qt progress window instead.
    """
    if allow_gui:
        try:
            import tkinter  # noqa: F401
            _run_repair_gui(broken)
            return
        except Exception:
            pass
    # Try Qt progress window (safe — Qt is already in the venv)
    try:
        _run_repair_qt(broken)
        return
    except Exception:
        pass
    _run_repair_plain(broken)


def _run_repair_qt(broken: list[str]):
    """
    Show a minimal Qt window while pip installs missing packages.
    Uses PySide6 which is already installed in the venv.
    Safe to call even when Qt environment is active (no XCB conflict).
    """
    from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                                   QLabel, QTextEdit, QProgressBar)
    from PySide6.QtCore import Qt, QThread, Signal, QMetaObject, Q_ARG
    from PySide6.QtGui import QFont, QColor, QPalette
    import sys as _sys

    app = QApplication.instance() or QApplication(_sys.argv[:1])

    # ── Window ────────────────────────────────────────────────────────────────
    win = QWidget()
    win.setWindowTitle('SETS-WARP — Installing packages')
    win.setFixedSize(560, 340)
    win.setWindowFlags(Qt.WindowType.Window |
                       Qt.WindowType.WindowTitleHint |
                       Qt.WindowType.CustomizeWindowHint)

    pal = win.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor('#1a1a1a'))
    win.setAutoFillBackground(True)
    win.setPalette(pal)

    lay = QVBoxLayout(win)
    lay.setContentsMargins(18, 18, 18, 18)
    lay.setSpacing(10)

    title = QLabel('Installing missing packages...')
    title.setStyleSheet('color:#c59129; font-size:13px; font-weight:bold;')
    lay.addWidget(title)

    pkgs_lbl = QLabel('Packages: ' + ', '.join(
        p.split('>=')[0].split('==')[0] for p in broken))
    pkgs_lbl.setStyleSheet('color:#aaaaaa; font-size:11px;')
    pkgs_lbl.setWordWrap(True)
    lay.addWidget(pkgs_lbl)

    bar = QProgressBar()
    bar.setRange(0, 0)          # indeterminate
    bar.setStyleSheet(
        'QProgressBar{background:#2a2a2a;border-radius:4px;height:8px;text-align:center;}'
        'QProgressBar::chunk{background:#c59129;border-radius:4px;}')
    bar.setFixedHeight(12)
    lay.addWidget(bar)

    log = QTextEdit()
    log.setReadOnly(True)
    log.setStyleSheet(
        'background:#111111;color:#cccccc;font-size:10px;border:1px solid #333;')
    log.setFont(QFont('Monospace', 9))
    lay.addWidget(log)

    status = QLabel('Starting...')
    status.setStyleSheet('color:#888888; font-size:10px;')
    lay.addWidget(status)

    win.show()
    app.processEvents()

    # ── Worker thread ─────────────────────────────────────────────────────────
    done_flag   = [False]
    error_flag  = [None]

    def on_line(msg, replace_last=False):
        # Append to log widget (called from worker thread → must be thread-safe)
        import sys as _sys
        print(msg, flush=True)
        _write_log = _setup_log_writer()
        _write_log(msg)
        # Schedule UI update on main thread
        QMetaObject.invokeMethod(
            log, 'append',
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, msg))
        QMetaObject.invokeMethod(
            status, 'setText',
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, msg[-80:] if len(msg) > 80 else msg))

    def on_done():
        done_flag[0] = True

    def on_error(m):
        error_flag[0] = m
        done_flag[0]  = True

    import threading as _th
    t = _th.Thread(
        target=_repair_worker, args=(on_line, on_done, on_error, broken),
        daemon=True)
    t.start()

    # ── Event loop — spin until worker done ───────────────────────────────────
    while not done_flag[0]:
        app.processEvents()
        import time; time.sleep(0.05)

    app.processEvents()

    # Final state
    if error_flag[0]:
        bar.setRange(0, 1); bar.setValue(0)
        bar.setStyleSheet(
            'QProgressBar{background:#2a2a2a;border-radius:4px;}'
            'QProgressBar::chunk{background:#ff6b6b;border-radius:4px;}')
        status.setStyleSheet('color:#ff6b6b; font-size:10px;')
        status.setText(f'Repair failed: {error_flag[0]}')
        title.setText('Installation failed — see log above')
        # Keep window open for 5 s so user can read error
        import time
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.05)
    else:
        bar.setRange(0, 1); bar.setValue(1)
        bar.setStyleSheet(
            'QProgressBar{background:#2a2a2a;border-radius:4px;}'
            'QProgressBar::chunk{background:#7effc8;border-radius:4px;}')
        status.setStyleSheet('color:#7effc8; font-size:10px;')
        status.setText('Done — starting SETS-WARP...')
        title.setText('Packages installed successfully')
        import time; time.sleep(1.2)
        app.processEvents()

    win.close()


def _run_repair_gui(broken: list[str]):
    import tkinter as tk

    BG, FG, ACCENT, RED = "#1a1a1a", "#eeeeee", "#c59129", "#ff6b6b"
    exit_code = [0]  # 0=repaired, 2=user exit

    root = tk.Tk()
    root.title("SETS-WARP — Dependency Repair")
    root.configure(bg=BG)
    W, H = 620, 460
    root.geometry(f"{W}x{H}")
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

    # ── confirmation screen ───────────────────────────────────────────────────
    confirm_frame = tk.Frame(root, bg=BG)
    confirm_frame.pack(fill="both", expand=True)

    tk.Label(confirm_frame, text="⚠  SETS-WARP — Missing or Broken Dependencies",
             bg=BG, fg=ACCENT, font=("Helvetica", 13, "bold")).pack(pady=(22, 8))
    tk.Label(confirm_frame,
             text="The following packages need to be repaired:",
             bg=BG, fg=FG, font=("Helvetica", 10)).pack(pady=(0, 6))

    pkg_box = tk.Text(confirm_frame, bg="#111", fg=RED, font=("Courier", 10),
                      relief="flat", height=len(broken) + 1, state="normal")
    for dep in broken:
        pkg_box.insert("end", f"  - {dep}\n")
    pkg_box.config(state="disabled")
    pkg_box.pack(fill="x", padx=30, pady=(0, 16))

    btn_frame = tk.Frame(confirm_frame, bg=BG)
    btn_frame.pack(pady=8)

    # ── repair progress screen (hidden until Repair clicked) ─────────────────
    repair_frame = tk.Frame(root, bg=BG)

    tk.Label(repair_frame, text="SETS — Repairing Dependencies",
             bg=BG, fg=ACCENT, font=("Helvetica", 13, "bold")).pack(pady=(18, 4))

    mid = tk.Frame(repair_frame, bg=BG)
    mid.pack(fill="both", expand=True, padx=16, pady=4)
    sb = tk.Scrollbar(mid)
    sb.pack(side="right", fill="y")
    log_widget = tk.Text(mid, bg="#111", fg=FG, font=("Courier", 10),
                         wrap="word", yscrollcommand=sb.set,
                         state="disabled", relief="flat", height=14)
    log_widget.pack(side="left", fill="both", expand=True)
    sb.config(command=log_widget.yview)

    status_var = tk.StringVar(value="Starting repair...")
    tk.Label(repair_frame, textvariable=status_var, bg=BG, fg=ACCENT,
             font=("Helvetica", 10)).pack(side="bottom", pady=(0, 10))

    _write_log = _setup_log_writer()
    _prev = [False]

    def _append(msg, replace_last=False):
        log_widget.config(state="normal")
        if replace_last and _prev[0]:
            log_widget.delete("end-2l", "end-1l")
        log_widget.insert("end", msg + "\n")
        log_widget.see("end")
        log_widget.config(state="disabled")
        _prev[0] = replace_last
        if msg.strip():
            status_var.set(msg.strip()[:80])

    def on_line(msg, replace_last=False):
        _write_log(msg)
        root.after(0, lambda m=msg, r=replace_last: _append(m, r))

    def on_done():
        root.after(0, root.destroy)

    def on_error(msg):
        def _show():
            _append(f"\nERROR: {msg}\n")
            tk.Button(repair_frame, text="Close", bg=ACCENT, fg=BG,
                      font=("Helvetica", 11, "bold"), relief="flat",
                      command=root.destroy).pack(pady=8)
        root.after(0, _show)

    def do_repair():
        confirm_frame.pack_forget()
        repair_frame.pack(fill="both", expand=True)
        threading.Thread(
            target=_repair_worker, args=(on_line, on_done, on_error, broken),
            daemon=True).start()

    def do_exit():
        exit_code[0] = 2
        root.destroy()

    tk.Button(btn_frame, text="Repair Automatically", bg=ACCENT, fg=BG,
              font=("Helvetica", 11, "bold"), relief="flat", padx=16, pady=6,
              command=do_repair).pack(side="left", padx=12)
    tk.Button(btn_frame, text="Exit", bg="#333", fg=FG,
              font=("Helvetica", 11), relief="flat", padx=16, pady=6,
              command=do_exit).pack(side="left", padx=12)

    root.mainloop()

    if exit_code[0] != 0:
        sys.exit(exit_code[0])


def _run_repair_plain(broken: list[str]):
    _write_log = _setup_log_writer()

    def on_line(msg, replace_last=False):
        _write_log(msg)
        print(msg, flush=True)

    done_evt = threading.Event()
    errors = []

    def on_done():
        done_evt.set()

    def on_error(m):
        errors.append(m)
        done_evt.set()

    threading.Thread(
        target=_repair_worker, args=(on_line, on_done, on_error, broken),
        daemon=True).start()
    done_evt.wait()
    if errors:
        print(f"\nRepair failed: {errors[0]}", file=sys.stderr)


def _repair_worker(on_line, on_done, on_error, broken: list[str]):
    try:
        on_line("=== SETS Dependency Repair ===")
        on_line(f"Found {len(broken)} package(s) to fix:")
        for dep in broken:
            on_line(f"  • {dep}")
        on_line("")
        on_line("Removing obsolete packages...")
        _uninstall_obsolete_packages(on_line)
        on_line("Cleaning stale .venv cache...")
        _cleanup_venv_pycache(on_line)
        install_dependencies(on_line, deps=broken, force=True)
        on_line("")
        on_line("Repair complete!")
        # Refresh WARP data if needed after repair
        if _warp_scraper_needed():
            on_line("")
            on_line("Refreshing WARP item database...")
            _run_warp_scraper(on_line)
        on_line("Starting SETS-WARP...")
        on_done()
    except Exception as exc:
        on_error(str(exc))


def main():
    print(f"[bootstrap] ROOT={ROOT}", flush=True)
    print(f"[bootstrap] sys.executable={sys.executable}", flush=True)
    print(f"[bootstrap] venv_python={venv_python()}", flush=True)
    print(f"[bootstrap] venv exists={venv_python().exists()}", flush=True)
    print(f"[bootstrap] running_in_our_venv={running_in_our_venv()}", flush=True)
    print(f"[bootstrap] portable_python exists={portable_python_exe().exists()}", flush=True)

    # cx_Freeze / PyInstaller frozen binary
    if getattr(sys, "frozen", False):
        import importlib.util
        spec = importlib.util.spec_from_file_location("__main__", ROOT / "main.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return

    # --repair flag: called by main.py on ImportError — fix then exit (caller relaunches)
    if _REPAIR_ONLY:
        if venv_python().exists():
            broken = _quick_check_venv()
            if not broken:
                broken = parse_pyproject()  # force full reinstall if check can't isolate issue
            _run_repair(broken)
        else:
            try:
                import tkinter  # noqa: F401
                run_with_tkinter_gui()
            except ImportError:
                run_plain_text()
        return  # don't relaunch — main.py handles re-exec via os.execv

    # Already running inside our venv → health-check then run
    if running_in_our_venv():
        # Check for missing/outdated packages (e.g. after pyproject.toml update)
        broken = _quick_check_venv()
        if broken:
            print(f"[bootstrap] packages missing/outdated: {broken} — repairing", flush=True)
            _run_repair(broken)
        # Silently refresh WARP data in background if cargo is newer than item_db
        if _warp_scraper_needed():
            print("[bootstrap] WARP data stale — refreshing in background...", flush=True)
            threading.Thread(
                target=_run_warp_scraper,
                args=(lambda msg, **_: print(f"[warp] {msg}", flush=True),),
                daemon=True,
            ).start()
        import importlib.util
        spec = importlib.util.spec_from_file_location("__main__", ROOT / "main.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return

    # Check if portable Python version matches config — wipe if stale
    if PYTHON_DIR.exists() and not _check_portable_python_version():
        import shutil
        print(f"[bootstrap] portable Python version mismatch (expected {PBS_VERSION}) — removing .python/ and .venv/", flush=True)
        shutil.rmtree(PYTHON_DIR, ignore_errors=True)
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    # Venv exists — fast health-check before relaunching
    if venv_python().exists():
        broken = _quick_check_venv()
        if broken:
            print(f"[bootstrap] venv check found issues: {broken} — repairing", flush=True)
            _run_repair(broken)
        relaunch_in_venv()
        return  # not reached on Linux/macOS (execv)

    # First run: download portable Python, create venv, install packages
    try:
        import tkinter  # noqa: F401
        run_with_tkinter_gui()
    except ImportError:
        run_plain_text()

    relaunch_in_venv()


if __name__ == "__main__":
    main()
