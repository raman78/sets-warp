# warp/__init__.py
# WARP — Screenshot-to-Build importer module for SETS
# Integrated with SETS v2026.03b060+
#
# Uses PySide6 (same as SETS — NOT PyQt6).
# All cache/config access goes through the SETS `self` object passed at construction.
#
# Entry points:
#   inject_warp_button(sets_instance)  — called from app.py setup_main_layout
#   WarpCoreWindow(sets_instance)      — standalone trainer window

__version__ = "0.1.0"
