#!/bin/sh
# SETS.sh — entry point for SETS (Linux / macOS)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || { echo "[Error] Cannot change to script directory."; exit 1; }

if [ ! -r "." ] || [ ! -w "." ]; then
    echo "[Error] The current folder must be readable and writable."
    exit 1
fi

# Export SETS_DIR so bootstrap.py and main.py always know where they live,
# regardless of Python's __file__ or .pyc cache pointing elsewhere
export SETS_DIR="$SCRIPT_DIR"

# Prevent Python from using stale .pyc cache from a previous location
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$SCRIPT_DIR/.pycache"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Always go through bootstrap — it does a fast venv health-check (~0.5s)
# and auto-repairs missing/wrong-version packages before launching main.py.
# On a healthy venv bootstrap just relaunches immediately, so startup is fast.
if [ -x "$VENV_PYTHON" ]; then
    exec "$VENV_PYTHON" "$SCRIPT_DIR/bootstrap.py" "$@"
fi

# ── First run — find Python 3.11+ to run bootstrap ────────────────────────────
PYTHON=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" > /dev/null 2>&1; then
        ok=$("$candidate" -c "import sys; print(1 if sys.version_info >= (3,11) else 0)" 2>/dev/null)
        if [ "$ok" = "1" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "  [SETS] Error: No Python 3.11+ found."
    echo ""
    echo "  Please install Python 3.11 or newer, then re-run SETS."
    echo "  https://www.python.org/downloads/"
    echo ""
    exit 1
fi

# ── Check tkinter availability ─────────────────────────────────────────────────
# tkinter is needed to show the installer GUI on first run.
# The portable Python we download has tkinter built-in, so this is only
# needed once. If missing we show distro-specific install instructions.
TKINTER_OK=$("$PYTHON" -c "import tkinter" 2>/dev/null && echo "1" || echo "0")

if [ "$TKINTER_OK" = "0" ]; then
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║          SETS — Missing dependency: tkinter                  ║"
    echo "  ║                                                              ║"
    echo "  ║  tkinter is required to display the installer window.        ║"
    echo "  ║  It is NOT part of SETS — it must be installed system-wide.  ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Detect distro from /etc/os-release (systemd standard, available on
    # virtually all modern Linux distros: Debian, Ubuntu, Fedora, Arch,
    # openSUSE, Alpine, Void, Gentoo, NixOS, Slackware derivatives, etc.)
    DISTRO_ID=""
    DISTRO_LIKE=""
    if [ -f /etc/os-release ]; then
        DISTRO_ID=$(. /etc/os-release && echo "${ID:-}" | tr '[:upper:]' '[:lower:]')
        DISTRO_LIKE=$(. /etc/os-release && echo "${ID_LIKE:-}" | tr '[:upper:]' '[:lower:]')
    fi

    # Helper: check if a string contains a word
    contains() { echo "$1" | grep -qw "$2"; }

    # Match distro → install command
    # We check ID first, then fall back to ID_LIKE for derivatives
    # (e.g. Linux Mint has ID=linuxmint, ID_LIKE=ubuntu debian)
    INSTALL_CMD=""

    if contains "$DISTRO_ID $DISTRO_LIKE" "ubuntu" || \
       contains "$DISTRO_ID $DISTRO_LIKE" "debian" || \
       contains "$DISTRO_ID $DISTRO_LIKE" "linuxmint" || \
       contains "$DISTRO_ID $DISTRO_LIKE" "pop" || \
       contains "$DISTRO_ID $DISTRO_LIKE" "elementary" || \
       contains "$DISTRO_ID $DISTRO_LIKE" "zorin" || \
       contains "$DISTRO_ID $DISTRO_LIKE" "kali" || \
       contains "$DISTRO_ID $DISTRO_LIKE" "raspbian"; then
        PKG="python3-tk"
        INSTALL_CMD="sudo apt install $PKG"

    elif contains "$DISTRO_ID $DISTRO_LIKE" "fedora" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "rhel" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "centos" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "almalinux" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "rocky" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "ol"; then
        PKG="python3-tkinter"
        INSTALL_CMD="sudo dnf install $PKG"

    elif contains "$DISTRO_ID $DISTRO_LIKE" "arch" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "manjaro" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "endeavouros" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "garuda"; then
        PKG="tk"
        INSTALL_CMD="sudo pacman -S $PKG"

    elif contains "$DISTRO_ID $DISTRO_LIKE" "opensuse" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "suse"; then
        PKG="python3-tk"
        INSTALL_CMD="sudo zypper install $PKG"

    elif contains "$DISTRO_ID $DISTRO_LIKE" "alpine"; then
        PKG="py3-tkinter"
        INSTALL_CMD="sudo apk add $PKG"

    elif contains "$DISTRO_ID $DISTRO_LIKE" "void"; then
        PKG="python3-tkinter"
        INSTALL_CMD="sudo xbps-install $PKG"

    elif contains "$DISTRO_ID $DISTRO_LIKE" "gentoo"; then
        INSTALL_CMD="sudo emerge -av dev-lang/python[tk]  (rebuild Python with USE=tk)"

    elif contains "$DISTRO_ID $DISTRO_LIKE" "nixos" || \
         contains "$DISTRO_ID $DISTRO_LIKE" "nix"; then
        INSTALL_CMD="nix-env -iA nixpkgs.python3Packages.tkinter"

    elif contains "$DISTRO_ID $DISTRO_LIKE" "slackware"; then
        INSTALL_CMD="installpkg python3-tkinter (from Slackware extras or SlackBuilds)"

    elif [ "$(uname)" = "Darwin" ]; then
        INSTALL_CMD="brew install python-tk  (or reinstall Python from python.org)"
    fi

    if [ -n "$INSTALL_CMD" ]; then
        echo "  Detected: ${DISTRO_ID:-unknown}${DISTRO_LIKE:+ (like: $DISTRO_LIKE)}"
        echo ""
        echo "  Run this command to install tkinter:"
        echo ""
        echo "      $INSTALL_CMD"
        echo ""
        echo "  Then re-run:  ./SETS.sh"
    else
        echo "  Could not detect your Linux distribution automatically."
        echo ""
        echo "  Install the tkinter package for your Python version, for example:"
        echo "    - Debian/Ubuntu:  sudo apt install python3-tk"
        echo "    - Fedora/RHEL:    sudo dnf install python3-tkinter"
        echo "    - Arch Linux:     sudo pacman -S tk"
        echo "    - openSUSE:       sudo zypper install python3-tk"
        echo "    - Alpine:         sudo apk add py3-tkinter"
        echo "    - Void Linux:     sudo xbps-install python3-tkinter"
        echo ""
        echo "  Then re-run:  ./SETS.sh"
    fi
    echo ""
    exit 1
fi

echo "[SETS] First run — starting setup (this will take a few minutes)..."
echo "[SETS] SCRIPT_DIR=$SCRIPT_DIR"
exec "$PYTHON" "$SCRIPT_DIR/bootstrap.py" "$@"
