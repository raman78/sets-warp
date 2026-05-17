"""
Diagnostic — ship_name / ship_type / ship_tier OCR extraction.

Iterates all screenshots with confirmed Ship Name / Type / Tier annotations,
runs the production OCR path (TextExtractor.extract_ship_info +
refine_ship_info), and reports:
  - Per-screen extracted vs ground-truth values
  - Per-field accuracy summary
  - PNG visualization with overlays (top-band crop + extracted/GT side-by-side)

Read-only — does not modify any source files. Output goes to
tests/_diag_out/ship_info_ocr/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from warp.recognition.text_extractor import TextExtractor, SHIP_TIER_VALUES
from warp.warp_importer import ShipDB

OUT_DIR = ROOT / 'tests' / '_diag_out' / 'ship_info_ocr'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN_PATH = ROOT / 'warp' / 'training_data' / 'annotations.json'
SEARCH_DIRS = [
    ROOT / 'warp' / 'training_data' / 'migrated_images',
    Path('/home/raman/STO_screens/screeny'),
    Path('/home/raman/STO_screens/screeny2'),
    Path('/home/raman/STO_screens/screeny_test'),
    Path('/home/raman/STO_screens/screeny_test2'),
    Path('/home/raman/STO_screens/screeny_group_test3'),
]


def _locate(fname: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


def _norm(s: str) -> str:
    """Case- and whitespace-insensitive comparison."""
    return ' '.join((s or '').lower().split())


def _draw_overlay(img: np.ndarray, gt: dict, pred: dict,
                   ocr_tokens: list, gt_texts: dict, pred_texts: dict) -> np.ndarray:
    """
    Render a comprehensive overlay:
      - Original top 25% of screenshot
      - All OCR tokens marked with bbox + height label
      - GT bboxes: thick colored (green=name, cyan=type, yellow=tier)
      - Predicted bboxes: dashed red
      - Side text panel listing GT vs PRED with hit markers
    """
    h, w = img.shape[:2]
    top_h = min(h, int(h * 0.25))
    img_view = img[0:top_h].copy()

    # Draw every OCR token in pale orange with its height label
    for (bx, by, bw, bh, text, conf) in ocr_tokens:
        if by + bh > top_h:
            continue
        cv2.rectangle(img_view, (bx, by), (bx + bw, by + bh), (50, 130, 200), 1)
        cv2.putText(img_view, f'h{bh}', (bx, max(8, by - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 130, 200), 1)

    # GT bboxes: thick coloured
    gt_colors = {'Ship Name': (0, 220, 0),
                 'Ship Type': (255, 255, 0),
                 'Ship Tier': (0, 200, 255)}
    for slot, color in gt_colors.items():
        bb = gt.get(slot, {}).get('bbox')
        if not bb:
            continue
        x, y, bw, bh = bb
        if y + bh > top_h:
            continue
        cv2.rectangle(img_view, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(img_view, f'GT {slot[5:]}', (x, max(10, y - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Predicted bboxes: thick red
    for slot, key in (('Ship Name', 'ship_name_bbox'),
                       ('Ship Type', 'ship_type_bbox'),
                       ('Ship Tier', 'ship_tier_bbox')):
        bb = pred.get(key)
        if not bb:
            continue
        x, y, bw, bh = bb
        if y + bh > top_h:
            continue
        cv2.rectangle(img_view, (x - 2, y - 2),
                      (x + bw + 2, y + bh + 2), (0, 50, 255), 2)

    # Right-side text panel
    panel_w = 700
    panel = np.full((top_h, panel_w, 3), 25, dtype=np.uint8)
    lines = [
        ('PRED vs GT', (255, 255, 255), 0.55),
        ('', None, None),
    ]
    for label, key_gt, key_pr in (
        ('NAME', 'name', 'name'),
        ('TYPE', 'type', 'type'),
        ('TIER', 'tier', 'tier'),
    ):
        gt_v = gt_texts.get(key_gt, '') or '—'
        pr_v = pred_texts.get(key_pr, '') or '—'
        hit = (gt_v != '—') and (_norm(gt_v) == _norm(pr_v))
        color = (60, 220, 60) if hit else ((200, 80, 80) if gt_v != '—' else (160, 160, 160))
        marker = 'OK' if hit else ('FAIL' if gt_v != '—' else '--')
        lines.append((f'[{marker}] {label}', color, 0.50))
        lines.append((f'  GT  : {gt_v[:75]}', (200, 200, 200), 0.40))
        lines.append((f'  PRED: {pr_v[:75]}', (160, 200, 250), 0.40))
        lines.append(('', None, None))

    y_cur = 20
    for text, color, scale in lines:
        if not text:
            y_cur += 6
            continue
        cv2.putText(panel, text, (10, y_cur),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        y_cur += int(scale * 38)

    # Stack image + panel side-by-side
    canvas = np.full((top_h, w + panel_w, 3), 0, dtype=np.uint8)
    canvas[:, :w] = img_view
    canvas[:, w:] = panel
    return canvas


def _ship_display_words(ship: dict | None) -> set:
    """Replicate ShipDB._load display-word index for a single ship entry."""
    if not ship:
        return set()
    parts: list[str] = []
    for k in ('displayprefix', 'displayclass', 'displaytype'):
        v = ship.get(k)
        if v:
            parts.append(str(v))
    raw_name = ship.get('name') or ''
    n = ' '.join(raw_name) if isinstance(raw_name, list) else str(raw_name)
    if n:
        parts.append(n)
    return set(' '.join(parts).lower().split())


def _shipdb_match_correct(matched: dict | None, gt_type: str) -> bool:
    """A matched ship is 'correct' when GT type words ⊆ matched ship's display words."""
    if not matched or not gt_type:
        return False
    gt_words = set(gt_type.lower().split())
    return gt_words.issubset(_ship_display_words(matched))


def _write_report(rows: list[dict], summary: dict) -> None:
    md = OUT_DIR / 'REPORT.md'
    lines = [
        '# Ship Info OCR Diagnostic',
        '',
        f'Total screens: **{summary["total"]}**',
        '',
        '## OCR extractor accuracy',
        '',
        '| field | hits | accuracy |',
        '|---|---|---|',
        f'| ship_name | {summary["name_hits"]} / {summary["name_total"]} | {summary["name_acc"]:.0%} |',
        f'| ship_type | {summary["type_hits"]} / {summary["type_total"]} | {summary["type_acc"]:.0%} |',
        f'| ship_tier | {summary["tier_hits"]} / {summary["tier_total"]} | {summary["tier_acc"]:.0%} |',
        '',
        '## ShipDB resolution (full pipeline — OCR + ShipDB lookup)',
        '',
        '| outcome | count |',
        '|---|---|',
        f'| Resolved correctly (GT ⊆ matched ship words) | {summary["sdb_correct"]} / {summary["sdb_total"]} ({summary["sdb_acc"]:.0%}) |',
        f'| Resolved wrong ship | {summary["sdb_wrong"]} |',
        f'| Keyword fallback (ShipDB found nothing) | {summary["sdb_fallback"]} |',
        '',
        '## Per-screen results',
        '',
        '| file | field | GT | predicted | hit |',
        '|---|---|---|---|---|',
    ]
    for r in rows:
        for field in ('name', 'type', 'tier'):
            gt_v = r['gt'].get(field, '')
            pr_v = r['pred'].get(field, '')
            if not gt_v:
                continue
            hit = '✅' if r['hits'].get(field) else '❌'
            lines.append(f'| {r["file"]} | {field} | `{gt_v}` | `{pr_v}` | {hit} |')

    lines += [
        '',
        '## Per-screen ShipDB lookup',
        '',
        '| file | OCR type | GT type | strategy | matched ship | correct |',
        '|---|---|---|---|---|---|',
    ]
    for r in rows:
        sdb = r.get('shipdb', {})
        matched = sdb.get('matched') or {}
        matched_name = matched.get('name') if isinstance(matched, dict) else ''
        if isinstance(matched_name, list):
            matched_name = ' '.join(matched_name)
        mark = '✅' if sdb.get('correct') else '❌'
        lines.append(
            f'| {r["file"]} | `{r["pred"].get("type","")}` | '
            f'`{r["gt"].get("type","")}` | {sdb.get("strategy","")} | '
            f'`{matched_name or "—"}` | {mark} |'
        )

    md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  report: {md}')


def main() -> int:
    print(f'Loading annotations from {ANN_PATH}')
    data = json.loads(ANN_PATH.read_text(encoding='utf-8'))

    targets = []
    for fname, lst in data.items():
        slots = {a['slot']: a for a in lst if isinstance(a, dict) and 'slot' in a}
        if 'Ship Type' not in slots or 'Ship Tier' not in slots:
            continue
        p = _locate(fname)
        if p is None:
            continue
        targets.append((p, slots))

    print(f'Found {len(targets)} screens with Ship Type+Tier annotations on disk')

    extractor = TextExtractor()
    # Use the same corrections file the importer loads
    corr = ROOT / 'warp' / 'models' / 'ship_type_corrections.json'
    if corr.exists():
        TextExtractor.load_corrections(corr)
        print(f'Loaded corrections: {corr}')

    # ShipDB — same cargo dir resolution as WarpImporter._get_shipdb
    cargo_dir = ROOT / '.config' / 'cargo'
    shipdb = ShipDB(cargo_dir)

    rows: list[dict] = []
    sums = {'name_hits': 0, 'type_hits': 0, 'tier_hits': 0,
            'name_total': 0, 'type_total': 0, 'tier_total': 0,
            'sdb_total': 0, 'sdb_correct': 0, 'sdb_wrong': 0, 'sdb_fallback': 0}

    for i, (path, slots) in enumerate(targets, 1):
        img = cv2.imread(str(path))
        if img is None:
            print(f'[{i}/{len(targets)}] cannot read {path}')
            continue
        info = extractor.extract_ship_info(img)
        info = extractor.refine_ship_info(img, info, SHIP_TIER_VALUES, None)

        # Re-run OCR on top band for visualization (cheap — single call)
        h, w = img.shape[:2]
        top_band = img[0:int(h * 0.20), :]
        try:
            ocr_raw = extractor._get_ocr().readtext(top_band)
        except Exception:
            ocr_raw = []
        ocr_tokens = []
        for poly, t, c in ocr_raw:
            if c < 0.25 or not t.strip():
                continue
            xs = [int(p[0]) for p in poly]; ys = [int(p[1]) for p in poly]
            x0, y0 = min(xs), min(ys)
            ocr_tokens.append((x0, y0, max(xs) - x0, max(ys) - y0, t, c))

        gt = {
            'name': slots.get('Ship Name', {}).get('name', '') or '',
            'type': slots.get('Ship Type', {}).get('name', '') or '',
            'tier': slots.get('Ship Tier', {}).get('name', '') or '',
        }
        pred = {
            'name': info.get('ship_name', ''),
            'type': info.get('ship_type', ''),
            'tier': info.get('ship_tier', ''),
        }
        hits = {
            'name': bool(gt['name']) and _norm(gt['name']) == _norm(pred['name']),
            'type': bool(gt['type']) and _norm(gt['type']) == _norm(pred['type']),
            'tier': bool(gt['tier']) and _norm(gt['tier']) == _norm(pred['tier']),
        }
        for f in ('name', 'type', 'tier'):
            if gt[f]:
                sums[f'{f}_total'] += 1
                if hits[f]:
                    sums[f'{f}_hits'] += 1

        # ShipDB resolution — feed OCR result into the same lookup the importer uses
        shipdb.get_profile(pred['name'], pred['type'], pred['tier'])
        sdb_strategy = shipdb.last_match_strategy or '—'
        sdb_matched = shipdb.last_match
        sdb_correct = _shipdb_match_correct(sdb_matched, gt['type'])
        if gt['type']:
            sums['sdb_total'] += 1
            if sdb_strategy == 'keyword-fallback' or sdb_matched is None:
                sums['sdb_fallback'] += 1
            elif sdb_correct:
                sums['sdb_correct'] += 1
            else:
                sums['sdb_wrong'] += 1

        rows.append({
            'file': path.name, 'gt': gt, 'pred': pred, 'hits': hits,
            'shipdb': {
                'strategy': sdb_strategy,
                'matched': sdb_matched,
                'correct': sdb_correct,
            },
        })

        # Visualization
        gt_bboxes = {s: slots.get(s, {}) for s in ('Ship Name', 'Ship Type', 'Ship Tier')}
        overlay = _draw_overlay(img, gt_bboxes, info, ocr_tokens, gt, pred)
        out_png = OUT_DIR / f'{path.stem}__overlay.png'
        cv2.imwrite(str(out_png), overlay)

        n_marker = '✅' if hits['name'] else ('—' if not gt['name'] else '❌')
        t_marker = '✅' if hits['type'] else '❌'
        r_marker = '✅' if hits['tier'] else '❌'
        print(f'[{i:2}/{len(targets)}] {path.name[:55]:55}  '
              f'N{n_marker} T{t_marker} Tier{r_marker}')
        if not hits['type']:
            print(f'         type GT={gt["type"]!r}')
            print(f'         type PR={pred["type"]!r}')
        if not hits['tier'] and gt['tier']:
            print(f'         tier GT={gt["tier"]!r} PR={pred["tier"]!r}')
        if not hits['name'] and gt['name']:
            print(f'         name GT={gt["name"]!r} PR={pred["name"]!r}')

    total = len(rows)
    summary = {
        'total': total,
        **sums,
        'name_acc': sums['name_hits'] / sums['name_total'] if sums['name_total'] else 0.0,
        'type_acc': sums['type_hits'] / sums['type_total'] if sums['type_total'] else 0.0,
        'tier_acc': sums['tier_hits'] / sums['tier_total'] if sums['tier_total'] else 0.0,
        'sdb_acc':  sums['sdb_correct'] / sums['sdb_total'] if sums['sdb_total'] else 0.0,
    }

    print()
    print('═' * 60)
    print(f'SUMMARY — {total} screens')
    print(f'  ship_name: {sums["name_hits"]:3}/{sums["name_total"]:3}  '
          f'({summary["name_acc"]:.0%})')
    print(f'  ship_type: {sums["type_hits"]:3}/{sums["type_total"]:3}  '
          f'({summary["type_acc"]:.0%})')
    print(f'  ship_tier: {sums["tier_hits"]:3}/{sums["tier_total"]:3}  '
          f'({summary["tier_acc"]:.0%})')
    print(f'  ShipDB OK: {sums["sdb_correct"]:3}/{sums["sdb_total"]:3}  '
          f'({summary["sdb_acc"]:.0%})  '
          f'(wrong={sums["sdb_wrong"]}, fallback={sums["sdb_fallback"]})')
    print('═' * 60)

    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
    _write_report(rows, summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
