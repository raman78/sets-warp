"""OCR diagnostic — measure slot label coverage across a directory of screenshots.

Usage: python -m warp.trainer.ocr_diag <dir>

Reports per-image: image size, screen type (heuristic), count of matched slot
labels, list of matched labels, sample of unmatched OCR tokens.

Aggregates: mean/median/min/max labels per image, percentile counts.
"""
import os, sys, json, cv2, re
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, '/home/raman/PycharmProjects/sets-warp')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from warp.recognition.layout_detector import SLOT_LABEL_ALIASES

LABEL_KEYS = list(SLOT_LABEL_ALIASES.keys())

# Screen-type hints — for bucketing results
MIXED_HINTS = {'devices', 'starship traits', 'reputation'}
EQ_HINTS    = {'fore weapons', 'aft weapons', 'deflector', 'engines'}
BOFF_HINTS  = {'stations', 'space stations', 'standard away team'}
TRAIT_HINTS = {'personal space traits', 'personal ground traits', 'starship traits'}

def fuzzy_match(text_lower: str) -> str | None:
    """Return canonical slot name if text matches any alias (exact or substring)."""
    t = text_lower.strip()
    if t in SLOT_LABEL_ALIASES:
        return SLOT_LABEL_ALIASES[t]
    # Loose substring — STO often renders "FORE WEAPONS (4)"
    for k, v in SLOT_LABEL_ALIASES.items():
        if len(k) >= 5 and k in t:
            return v
    return None

def diagnose_image(reader, img_path: Path) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        return {'error': 'cannot_read', 'path': str(img_path)}
    h, w = img.shape[:2]
    try:
        ocr = reader.readtext(img)
    except Exception as e:
        return {'error': str(e), 'path': str(img_path)}

    matched = {}           # slot_name → list of (text, conf, cx, cy)
    all_tokens = []
    for (bbox_pts, text, conf) in ocr:
        if conf < 0.40:
            continue
        t_lower = text.strip().lower()
        all_tokens.append((t_lower, conf))
        slot = fuzzy_match(t_lower)
        if slot:
            cx = sum(pt[0] for pt in bbox_pts) / 4
            cy = sum(pt[1] for pt in bbox_pts) / 4
            matched.setdefault(slot, []).append((text.strip(), conf, int(cx), int(cy)))

    hints = {t for (t, _) in all_tokens}
    screen_type = 'UNKNOWN'
    if hints & MIXED_HINTS and hints & EQ_HINTS:
        screen_type = 'MIXED'
    elif hints & EQ_HINTS:
        screen_type = 'EQ'
    elif hints & BOFF_HINTS:
        screen_type = 'BOFFS'
    elif hints & TRAIT_HINTS:
        screen_type = 'TRAITS'

    return {
        'path': str(img_path),
        'w': w, 'h': h,
        'screen_type': screen_type,
        'n_labels': len(matched),
        'labels': list(matched.keys()),
        'match_positions': {k: [(x, y) for (_, _, x, y) in v] for k, v in matched.items()},
        'n_ocr_tokens': len(all_tokens),
    }

def main(root: str, limit: int | None = None):
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    root_p = Path(root)
    if root_p.is_file():
        paths = [root_p]
    else:
        paths = sorted([p for p in root_p.rglob('*')
                        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if limit:
        paths = paths[:limit]
    print(f'Scanning {len(paths)} images', flush=True)

    results = []
    for i, p in enumerate(paths):
        r = diagnose_image(reader, p)
        results.append(r)
        if 'error' in r:
            print(f'[{i+1}/{len(paths)}] ERR {p.name}: {r["error"]}', flush=True)
        else:
            print(f'[{i+1}/{len(paths)}] {r["screen_type"]:<7} {r["n_labels"]:2d} labels  '
                  f'{r["w"]}x{r["h"]}  {p.name}', flush=True)

    # Aggregate
    by_type = defaultdict(list)
    for r in results:
        if 'error' in r:
            continue
        by_type[r['screen_type']].append(r['n_labels'])

    print()
    print('=== Aggregate (labels per image) ===')
    for st in ['MIXED', 'EQ', 'BOFFS', 'TRAITS', 'UNKNOWN']:
        vals = by_type.get(st, [])
        if not vals:
            continue
        vals.sort()
        print(f'  {st:<7} n={len(vals):3d}  '
              f'min={vals[0]} p25={vals[len(vals)//4]} median={vals[len(vals)//2]} '
              f'p75={vals[3*len(vals)//4]} max={vals[-1]}  '
              f'mean={sum(vals)/len(vals):.1f}')

    # Feasibility verdict for MIXED
    mixed_vals = sorted(by_type.get('MIXED', []))
    if mixed_vals:
        ge10 = sum(1 for v in mixed_vals if v >= 10)
        ge5  = sum(1 for v in mixed_vals if v >= 5)
        print()
        print(f'=== MIXED feasibility ===')
        print(f'  images with ≥10 labels: {ge10}/{len(mixed_vals)} ({100*ge10/len(mixed_vals):.0f}%)')
        print(f'  images with ≥5  labels: {ge5}/{len(mixed_vals)} ({100*ge5/len(mixed_vals):.0f}%)')

    # Dump full JSON for inspection
    out = Path('/tmp/ocr_diag.json')
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'\nFull results → {out}')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/home/raman/STO_screens/',
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
