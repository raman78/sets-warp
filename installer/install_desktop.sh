#!/bin/bash
# installer/install_desktop.sh
# Installs SETS-WARP icon and .desktop entry for Linux desktop environments.
#
# Run once after cloning or extracting the archive:
#   bash installer/install_desktop.sh
#
# Installs to ~/.local/ — no root required.
# To uninstall:
#   bash installer/install_desktop.sh --uninstall

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ICON_SRC="$SCRIPT_DIR/local/SETS_icon_small.png"
EXEC="$SCRIPT_DIR/sets_warp.sh"
ICON_NAME="sets-warp"
# Each install gets a unique desktop filename (passed via SETS_DESKTOP_NAME from sets_warp.sh)
DESKTOP_BASE="${SETS_DESKTOP_NAME:-sets-warp}"
ICON_DIR_128="$HOME/.local/share/icons/hicolor/128x128/apps"
ICON_DIR_FALLBACK="$HOME/.local/share/icons"
DESKTOP_DIR="$HOME/.local/share/applications"
DESKTOP_FILE="$DESKTOP_DIR/${DESKTOP_BASE}.desktop"

# ── Uninstall ─────────────────────────────────────────────────────────────────
if [ "${1}" = "--uninstall" ]; then
    echo "Uninstalling SETS-WARP desktop entry (${DESKTOP_BASE})..."
    rm -f "$DESKTOP_FILE"
    # Also clean up legacy non-hashed entry if it points to this install
    _LEGACY_FILE="$DESKTOP_DIR/sets-warp.desktop"
    if [ -f "$_LEGACY_FILE" ] && grep -qF "Exec=$SCRIPT_DIR/" "$_LEGACY_FILE" 2>/dev/null; then
        rm -f "$_LEGACY_FILE"
    fi
    rm -f "$ICON_DIR_128/$ICON_NAME.png"
    rm -f "$ICON_DIR_FALLBACK/$ICON_NAME.png"
    if command -v gtk-update-icon-cache > /dev/null 2>&1; then
        gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
    fi
    if command -v update-desktop-database > /dev/null 2>&1; then
        update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
    fi
    echo "Done. SETS-WARP removed from application menu."
    exit 0
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$ICON_SRC" ]; then
    echo "Error: icon not found: $ICON_SRC"
    echo "Run this script from the sets-warp directory or its installer/ subfolder."
    exit 1
fi

if [ ! -f "$EXEC" ]; then
    echo "Error: launch script not found: $EXEC"
    exit 1
fi

chmod +x "$EXEC"

# ── Install icon ──────────────────────────────────────────────────────────────
mkdir -p "$ICON_DIR_128"
cp "$ICON_SRC" "$ICON_DIR_128/$ICON_NAME.png"
# Also copy to flat fallback location (some DEs prefer this)
mkdir -p "$ICON_DIR_FALLBACK"
cp "$ICON_SRC" "$ICON_DIR_FALLBACK/$ICON_NAME.png"

# ── Write .desktop file ───────────────────────────────────────────────────────
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=SETS-WARP
GenericName=STO Build Planner
Comment=Star Trek Online build planner with WARP screenshot recognition ($SCRIPT_DIR)
Exec=$EXEC
Icon=$ICON_NAME
Terminal=false
Type=Application
Categories=Game;Utility;
Keywords=STO;Star Trek Online;build;planner;WARP;
StartupWMClass=sets-warp
StartupNotify=true
Path=$(dirname "$EXEC")
EOF

chmod +x "$DESKTOP_FILE"

# ── Update caches ─────────────────────────────────────────────────────────────
if command -v gtk-update-icon-cache > /dev/null 2>&1; then
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
fi
if command -v update-desktop-database > /dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
fi

echo ""
echo "SETS-WARP installed to your application menu."
echo "  Icon:    $ICON_DIR_128/$ICON_NAME.png"
echo "  Desktop: $DESKTOP_FILE"
echo "  Launch:  $EXEC"
echo ""
echo "The entry will appear in your DE's application launcher."
echo "To uninstall: use Settings → Uninstall inside the app, or"
echo "  bash installer/install_desktop.sh --uninstall"
