#!/usr/bin/env python3
"""
warp/tools/test_pipeline.py — End-to-end test WARP pipeline
============================================================
Runs the full pipeline on a folder of screenshots WITHOUT the GUI.

Usage (from the SETS-WARP directory):
    python warp/tools/test_pipeline.py --folder /path/to/screenshots
    python warp/tools/test_pipeline.py --folder ~/screenshots --type GROUND
    python warp/tools/test_pipeline.py --folder ~/screenshots --verbose
    python warp/tools/test_pipeline.py --image screenshot.png   # single file

Requirements:
    - SETS launched at least once (so .config/cargo/ and images/ exist)
    - Or: python warp/tools/test_pipeline.py --skip-matcher  (layout only)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── Auto-restart w .venv SETS-WARP ────────────────────────────────────────────
# Uruchamiamy się zawsze w lokalnym .venv żeby mieć cv2, numpy itp.

def _ensure_venv():
    root    = Path(__file__).resolve().parent.parent.parent
    is_win  = sys.platform == 'win32'
    venv_py = root / ('.venv/Scripts/python.exe' if is_win else '.venv/bin/python')

    if venv_py.exists() and Path(sys.executable).resolve() == venv_py.resolve():
        return  # już jesteśmy w .venv

    if venv_py.exists():
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)

    # No .venv — run bootstrap.py
    bootstrap = root / 'bootstrap.py'
    if bootstrap.exists():
        print('  → No .venv found — running bootstrap.py ...')
        subprocess.check_call([sys.executable, str(bootstrap)])
        if venv_py.exists():
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)

    print('ERROR: brak .venv — uruchom najpierw sets_warp.sh / sets_warp.bat', file=sys.stderr)
    sys.exit(1)

_ensure_venv()

import argparse
import json
import logging
import time

# ── Dodaj root projektu do sys.path ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(name)s: %(message)s',
)
log = logging.getLogger('test_pipeline')

G = '\033[92m'; Y = '\033[93m'; R = '\033[91m'; B = '\033[94m'; W = '\033[97m'; RS = '\033[0m'


# ── Stub sets_app (bez GUI) ────────────────────────────────────────────────────

class _StubCache:
    """Minimalne cache żeby SETSIconMatcher nie crashował."""
    def __init__(self, config_dir: Path):
        self.equipment: dict = {}
        self._load_equipment(config_dir)

    def _load_equipment(self, config_dir: Path):
        """Load JSON files from .config/cargo/ as equipment cache."""
        cargo = config_dir / 'cargo'
        if not cargo.exists():
            log.warning(f'Missing cargo directory: {cargo}')
            return
        count = 0
        for f in cargo.glob('*.json'):
            if f.name == 'ship_list.json':
                continue
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    self.equipment[f.stem] = data
                    count += len(data)
            except Exception as e:
                log.debug(f'skip {f.name}: {e}')
        log.info(f'Cache: {count} items z {len(self.equipment)} kategorii')


class _StubApp:
    """Stub sets_app przekazywany do WarpImporter."""
    def __init__(self, root: Path):
        config_dir = root / '.config'
        self.cache  = _StubCache(config_dir)
        self.images_dir = root / 'images'
        self._root  = root

    def get_image_path(self, name: str) -> Path | None:
        """Symuluje dostęp do obrazków itemów."""
        candidates = [
            self.images_dir / f'{name}.png',
            self.images_dir / f'{name}.jpg',
        ]
        for c in candidates:
            if c.exists():
                return c
        return None


# ── Test ───────────────────────────────────────────────────────────────────────

def run_test(
    paths:        list[Path],
    build_type:   str,
    skip_matcher: bool,
    verbose:      bool,
) -> None:
    from warp.warp_importer import WarpImporter

    app = _StubApp(ROOT)
    importer = WarpImporter(sets_app=app, build_type=build_type)

    if skip_matcher:
        log.info('--skip-matcher: skipping SETSIconMatcher')
        importer._get_matcher = lambda: None  # type: ignore

    # If a folder was given — collect files
    all_files: list[Path] = []
    for p in paths:
        if p.is_dir():
            exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
            all_files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in exts))
        elif p.is_file():
            all_files.append(p)
        else:
            log.warning(f'Nie istnieje: {p}')

    if not all_files:
        print(f'{R}No images to process.{RS}')
        sys.exit(1)

    print(f'\n{W}WARP Pipeline Test{RS}')
    print(f'  Pliki:      {len(all_files)}')
    print(f'  Build type: {build_type}')
    print(f'  Root:       {ROOT}')
    print()

    t0 = time.time()

    # Przetwarzaj każdy plik osobno (żeby zobaczyć co zwraca każdy)
    total_items = 0
    total_errors = 0

    for i, fpath in enumerate(all_files):
        print(f'{B}[{i+1}/{len(all_files)}]{RS} {fpath.name}')
        ft = time.time()

        try:
            import cv2
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if img is None:
                print(f'  {R}✗ Nie można wczytać obrazu{RS}')
                total_errors += 1
                continue

            h, w = img.shape[:2]
            print(f'  Rozmiar: {w}×{h}px')

            # Step 1: TextExtractor
            text_ex = importer._get_text()
            info = text_ex.extract_ship_info(img)
            ship  = info.get('ship_name', '?')
            stype = info.get('ship_type', '?')
            tier  = info.get('ship_tier', '?')
            btype = info.get('build_type', build_type)
            print(f'  {G}Statek:{RS} {ship!r}  typ: {stype!r}  tier: {tier}  build: {btype}')

            # Step 2: ShipDB
            db      = importer._get_shipdb()
            profile = db.get_profile(ship, stype)
            slots_found = {k: v for k, v in profile.items() if v > 0}
            print(f'  {G}Profil:{RS} {slots_found}')

            # Step 3: LayoutDetector
            layout_det = importer._get_layout()
            layout = layout_det.detect(img, btype, profile)
            bbox_count = sum(len(v) for v in layout.values())
            print(f'  {G}Layout:{RS} {len(layout)} slot groups, {bbox_count} bboxes total')

            if verbose:
                for slot, bboxes in layout.items():
                    if bboxes:
                        print(f'    {slot}: {len(bboxes)} × {bboxes[0]}')

            # Step 4: IconMatcher (opcjonalny)
            if not skip_matcher:
                matcher = importer._get_matcher()
                matched = 0
                from warp.warp_importer import SLOT_ORDER
                for slot_def in SLOT_ORDER.get(btype, []):
                    slot_name = slot_def['name']
                    max_count = profile.get(slot_name, 0)
                    if max_count == 0:
                        continue
                    bboxes = layout.get(slot_name, [])[:max_count]
                    for idx, bbox in enumerate(bboxes):
                        crop = importer._crop(img, bbox)
                        if crop is None or crop.size == 0:
                            continue
                        name, conf, _ = matcher.match(crop)
                        if name:
                            matched += 1
                            total_items += 1
                            status = G if conf >= 0.72 else Y
                            if verbose or conf >= 0.72:
                                print(f'    {status}{slot_name}[{idx}]:{RS} {name!r} ({conf:.2f})')
                print(f'  {G}Rozpoznano:{RS} {matched} itemów')
            else:
                print(f'  (matcher pominięty)')

        except Exception as e:
            log.exception(f'Błąd przy {fpath.name}')
            print(f'  {R}✗ BŁĄD: {e}{RS}')
            total_errors += 1

        elapsed = time.time() - ft
        print(f'  {B}Czas: {elapsed:.1f}s{RS}\n')

    total_time = time.time() - t0
    print('─' * 50)
    print(f'{W}Podsumowanie:{RS}')
    print(f'  Pliki:    {len(all_files)}')
    print(f'  Itemy:    {total_items}')
    print(f'  Błędy:    {total_errors}')
    print(f'  Czas:     {total_time:.1f}s ({total_time/len(all_files):.1f}s/img)')
    if total_errors == 0:
        print(f'  {G}✓ Wszystko OK{RS}')
    else:
        print(f'  {R}✗ {total_errors} błędów{RS}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='WARP end-to-end pipeline test (bez GUI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python warp/tools/test_pipeline.py --folder ~/screeny
  python warp/tools/test_pipeline.py --image screenshot.png
  python warp/tools/test_pipeline.py --folder ~/screeny --type GROUND --verbose
  python warp/tools/test_pipeline.py --folder ~/screeny --skip-matcher
        """
    )
    parser.add_argument('--folder', '-f', metavar='DIR',
                        help='Folder ze screenshotami')
    parser.add_argument('--image', '-i', metavar='FILE', action='append',
                        help='Pojedynczy plik (można podać wielokrotnie)')
    parser.add_argument('--type', '-t', choices=['SPACE', 'GROUND'], default='SPACE',
                        dest='build_type', help='Typ buildu (domyślnie: SPACE)')
    parser.add_argument('--skip-matcher', action='store_true',
                        help='Pomiń SETSIconMatcher (tylko layout + OCR)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Pokaż szczegóły każdego slotu')
    parser.add_argument('--debug', action='store_true',
                        help='Włącz debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    paths: list[Path] = []
    if args.folder:
        paths.append(Path(args.folder))
    if args.image:
        paths.extend(Path(p) for p in args.image)

    if not paths:
        parser.print_help()
        print(f'\n{R}Provide --folder or --image{RS}')
        sys.exit(1)

    run_test(
        paths        = paths,
        build_type   = args.build_type,
        skip_matcher = args.skip_matcher,
        verbose      = args.verbose,
    )


if __name__ == '__main__':
    main()
