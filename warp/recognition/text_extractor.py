# warp/recognition/text_extractor.py
#
# Extracts ship name, type, tier and build/screen type from STO screenshots.
#
# Detected screen types:
#   SPACE         — Status tab: ship name + equipment slots
#   GROUND        — Ground equipment tab
#   SPACE_TRAITS  — Traits tab: Personal Space Traits / Starship Traits / Reputation
#   GROUND_TRAITS — Traits tab: Personal Ground Traits / Ground Reputation
#   BOFFS         — Bridge Officer abilities panel
#   SPEC          — Specialization trees (Primary / Secondary)
#   (empty)       — Unknown / not detected
#
# Specializations (Primary — 30 abilities):
#   Command Officer, Intelligence Officer, Miracle Worker,
#   Pilot, Temporal Operative
# Specializations (Secondary only — 15 abilities):
#   Constable, Commando, Strategist
# (all released between Delta Rising and Season 14)

from __future__ import annotations

import re
import logging
import numpy as np

log = logging.getLogger(__name__)
try:
    from src.setsdebug import log as _slog
except Exception:
    _slog = log

RE_TIER      = re.compile(r'\[?(T[1-6](?:-(?:U|X|X2))?)(?:\]|$)', re.IGNORECASE)
RE_TIER_LOOSE = re.compile(r'\b(T[1-6](?:-(?:U|X|X2))?)\b', re.IGNORECASE)

# Ship name prefix — Federation/Klingon/Romulan/etc. registries.
# Used as a fallback anchor when tier token is missing (typical for T5 ships).
# Tolerates OCR variants: missing dots, digits-for-letters, underscores, colons.
RE_NAME_PREFIX = re.compile(
    r'^\s*(?:U|I|R|V|K|N|D|L|S|Z)\s*\.?\s*[A-Z0-9]\s*\.?\s*[A-Z0-9]\s*[\W_]\s*\S',
    re.IGNORECASE,
)


def _is_name_prefix_alone(text: str) -> bool:
    """
    Match a standalone prefix-only OCR token like 'U.S.S.', 'I.K.S.',
    'U.s.5_', 'L.KS:', 'Z.l.KS:'. Short, no spaces, starts with a letter,
    contains 2+ non-alphanumeric chars (dots/colons/underscores).
    """
    s = text.strip()
    if not s or len(s) > 10 or ' ' in s:
        return False
    if not s[0].isalpha():
        return False
    non_alnum = sum(1 for c in s if not c.isalnum())
    return non_alnum >= 2


def _is_name_prefix_token(text: str) -> bool:
    """
    Loose detector for ship name prefix (U.S.S., I.K.S., R.R.W. and OCR-noisy
    variants like 'U.s.5_', 'Z.l.KS:', 'L.KS:'). Pattern:
      - starts with a single capital letter,
      - followed by 1-7 chars containing 2+ non-alphanumeric chars
        (dots, colons, underscores, spaces),
      - then whitespace, then a proper-name word.
    """
    if not text:
        return False
    s = text.strip()
    if RE_NAME_PREFIX.match(s):
        return True
    m = re.match(r'^([A-Z][\w\W]{1,7}?)\s+([A-Z]\S{2,})', s, re.IGNORECASE)
    if not m:
        return False
    prefix = m.group(1)
    non_alnum = sum(1 for c in prefix if not c.isalnum())
    return non_alnum >= 2

# HUD / tab / section words that frequently leak into the top-band OCR
# but are never part of the ship name/type. Matched as whole tokens.
_HUD_BLACKLIST = {
    'collapse', 'collapse all', 'details', 'active', 'active space duty',
    'active ground duty', 'status', 'skills', 'traits', 'ship', 'stations',
    'reputation', 'summer', 'tactical', 'engineering', 'science',
    'personal space traits', 'personal ground traits', 'starship traits',
    'space reputation', 'ground reputation',
}

# Canonical STO ship tier values — used for fuzzy-match snapping after OCR.
# Single source of truth for both extract_ship_info refinement and the trainer's
# per-bbox OCR fallback worker.
SHIP_TIER_VALUES: list[str] = [
    'T1', 'T2', 'T3', 'T4', 'T5', 'T5-U', 'T5-X', 'T5-X2',
    'T6', 'T6-X', 'T6-X2',
]

# ROI for ship name/type block (top-left, fraction of image)
SHIP_INFO_ROI = (0.0, 0.0, 0.34, 0.28)

# ── Keyword sets per screen type ──────────────────────────────────────────────
# Checked against lowercase OCR text of scan regions.
# More specific keywords are listed first (longer matches win).

# ── Screen type keyword sets ─────────────────────────────────────────────────
# Each dict maps lowercase OCR substring → screen type.
# More specific (longer) strings should be listed first.

# Traits screen — labels that only appear on traits/reputation tabs
_TRAIT_KEYWORDS: dict[str, str] = {
    'personal space traits':      'SPACE_TRAITS',
    'starship traits':             'SPACE_TRAITS',
    'space reputation':            'SPACE_TRAITS',
    'active space rep':            'SPACE_TRAITS',
    'active reputation':           'SPACE_TRAITS',
    'personal ground traits':      'GROUND_TRAITS',
    'ground reputation':           'GROUND_TRAITS',
    'active ground rep':           'GROUND_TRAITS',
    'active ground reputation':    'GROUND_TRAITS',
}

# Bridge Officer screen — space vs ground distinguished by screen header
_BOFF_KEYWORDS: dict[str, str] = {
    'space stations':           'SPACE_BOFFS',   # STO header for space boff abilities
    'standard away team':       'GROUND_BOFFS',  # STO header for ground boff abilities
    'bridge officer abilities': 'BOFFS',
    'bridge officer':           'BOFFS',
    'boff abilities':           'BOFFS',
    'tactical ability':         'BOFFS',
    'engineering ability':      'BOFFS',
    'science ability':          'BOFFS',
}

# Space equipment slot labels — presence of 2+ confirms SPACE_EQ
_SPACE_EQ_LABELS: list[str] = [
    'fore weapons', 'aft weapons', 'experimental weapon',
    'deflector', 'secondary deflector',
    'impulse', 'warp core', 'singularity core',
    'shields', 'shield array',
    'engineering consoles', 'science consoles', 'tactical consoles',
    'universal consoles', 'hangar',
    'devices',
]

# Ground equipment slot labels — presence of 2+ confirms GROUND_EQ
_GROUND_EQ_LABELS: list[str] = [
    'kit modules', 'kit module',
    'body armor', 'combat armor',
    'ev suit', 'environmental suit',
    'personal shield',
    'ground weapon', 'secondary weapon',
    'ground device',
]

# Minimum label hits to confirm equipment screen type
_EQ_MIN_HITS = 2

# All STO specialization names (Primary and Secondary).
# Each appears as a section header on the Specializations screen.
_SPEC_NAMES: list[str] = [
    # Primary specializations
    'command officer',
    'intelligence officer',
    'miracle worker',
    'pilot',
    'temporal operative',
    # Secondary specializations
    'constable',
    'commando',
    'strategist',
]

# UI headers that confirm we are on the Specializations screen
_SPEC_HEADER_KEYWORDS: list[str] = [
    'primary specialization',
    'secondary specialization',
    'specialization points',
]


def _name_text_from_row_tokens(tokens: list) -> tuple[str, list]:
    """
    Build ship-name text from a row's tokens, defending against two cases
    where a far-away label (e.g. 'Fore Weapons') leaks into the name:

      (a) EasyOCR fused name + distant label into ONE wide token with many
          internal spaces ('U.S.S. ILLINOIS                Fore').
      (b) Adjacent tokens in the same row are visually far apart in x.

    Returns (text, kept_tokens). kept_tokens is the filtered list (for bbox
    union) — empty when no tokens survive.
    """
    if not tokens:
        return '', []
    toks = sorted(tokens, key=lambda t: t['x'])
    kept = [toks[0]]
    for t in toks[1:]:
        prev = kept[-1]
        gap = t['x'] - (prev['x'] + prev['w'])
        if gap > max(prev['h'], t['h']) * 2.0:
            break
        kept.append(t)
    text = ' '.join(t['text'] for t in kept).strip()
    text = re.split(r'\s{5,}', text, maxsplit=1)[0].strip()
    return text, kept


def _detect_type_from_text(lines_lower: list[str]) -> str:
    """
    Return detected screen type from OCR lines, or empty string if unknown.

    Priority order (most specific first):
      traits → boffs → spec → ground_eq → space_eq
    Equipment screens need 2+ matching labels to avoid false positives
    (single words like 'shields' or 'devices' can appear anywhere).
    """
    joined = ' '.join(lines_lower)

    # 1. Trait screen — very specific multi-word phrases
    for kw, btype in _TRAIT_KEYWORDS.items():
        if kw in joined:
            return btype

    # 2. Boff screen
    for kw, btype in _BOFF_KEYWORDS.items():
        if kw in joined:
            return btype

    # 3. Specialization screen — header keywords
    for kw in _SPEC_HEADER_KEYWORDS:
        if kw in joined:
            return 'SPEC'

    # 4. Specialization screen — at least one spec name present
    spec_hits = sum(1 for name in _SPEC_NAMES if name in joined)
    if spec_hits >= 1:
        return 'SPEC'

    # 5. Ground equipment — 2+ ground slot labels
    ground_hits = sum(1 for lbl in _GROUND_EQ_LABELS if lbl in joined)
    if ground_hits >= _EQ_MIN_HITS:
        return 'GROUND'

    # 6. Space equipment — 2+ space slot labels
    space_hits = sum(1 for lbl in _SPACE_EQ_LABELS if lbl in joined)
    if space_hits >= _EQ_MIN_HITS:
        return 'SPACE'

    # 7. Single strong space indicator
    strong_space = ['fore weapons', 'aft weapons', 'warp core',
                    'singularity core', 'experimental weapon']
    if any(kw in joined for kw in strong_space):
        return 'SPACE'

    # 8. Single strong ground indicator
    strong_ground = ['kit modules', 'body armor', 'ev suit', 'personal shield']
    if any(kw in joined for kw in strong_ground):
        return 'GROUND'

    return ''


class TextExtractor:
    """
    Extracts structured info from an STO screenshot.

    Returns dict:
        ship_name  : str   — e.g. "U.S.S. Genius"
        ship_type  : str   — e.g. "Typhoon Temporal Battlecruiser"
        ship_tier  : str   — e.g. "T6-X2"
        build_type : str   — SPACE | GROUND | SPACE_TRAITS | GROUND_TRAITS
                             | BOFFS | SPEC | '' (unknown)
    """

    # Community OCR correction map: {raw_ocr_text: corrected_text}.
    # Loaded from ship_type_corrections.json downloaded by ModelUpdater.
    # Applied to ship_type and ship_tier results after OCR extraction.
    _corrections: dict[str, str] = {}

    @classmethod
    def load_corrections(cls, path) -> None:
        """Load ship_type_corrections.json from the given path."""
        import json
        from pathlib import Path
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            if isinstance(data, dict):
                cls._corrections = data
                log.debug(f'TextExtractor: loaded {len(data)} OCR corrections from {path}')
        except Exception as e:
            log.warning(f'TextExtractor: could not load corrections from {path}: {e}')

    def __init__(self):
        self._ocr = None

    @staticmethod
    def _poly_to_xywh(poly) -> tuple[int, int, int, int]:
        """EasyOCR polygon [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] → axis-aligned (x,y,w,h)."""
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        x0, y0 = min(xs), min(ys)
        return x0, y0, max(xs) - x0, max(ys) - y0

    @staticmethod
    def _is_dark_bg(img: np.ndarray, bbox: tuple[int, int, int, int],
                    pad: int = 2) -> bool:
        """
        Sample mean brightness of pixels around (not inside) a text bbox.
        Ship_* labels in STO sit on near-black panel background; HUD tab
        labels and tooltips sit on lighter overlays. Threshold ~70 catches
        the panel background while letting in slight gradients.
        """
        try:
            x, y, w, h = bbox
            ih, iw = img.shape[:2]
            x0 = max(0, x - pad); x1 = min(iw, x + w + pad)
            y0 = max(0, y - pad); y1 = min(ih, y + h + pad)
            if x1 <= x0 or y1 <= y0:
                return False
            patch = img[y0:y1, x0:x1]
            if patch.size == 0:
                return False
            gray = patch.mean(axis=2) if patch.ndim == 3 else patch
            # Use the darker half of pixels — text foreground biases mean upward.
            flat = gray.flatten()
            if flat.size == 0:
                return False
            flat.sort()
            dark_mean = float(flat[: max(1, len(flat) // 2)].mean())
            return dark_mean < 100.0
        except Exception:
            return False

    @staticmethod
    def _union_xywh(*boxes) -> tuple[int, int, int, int] | None:
        """Return the smallest (x,y,w,h) covering all non-None inputs."""
        valid = [b for b in boxes if b]
        if not valid:
            return None
        x0 = min(b[0] for b in valid)
        y0 = min(b[1] for b in valid)
        x1 = max(b[0] + b[2] for b in valid)
        y1 = max(b[1] + b[3] for b in valid)
        return x0, y0, x1 - x0, y1 - y0

    def extract_ship_info(self, img: np.ndarray) -> dict:
        result = {
            'ship_name':  '',
            'ship_type':  '',
            'ship_tier':  '',
            'ship_name_bbox': None,   # (x,y,w,h) | None
            'ship_type_bbox': None,
            'ship_tier_bbox': None,
            'build_type': '',
            'scan_scope': 'partial',  # 'partial' or 'full' — set below
        }
        try:
            h, w = img.shape[:2]

            # ── Pass 1: screen type detection ────────────────────────────────
            # Two-stage: fast partial scan first, full scan if needed.
            # Partial covers most cases; full scan handles MIXED layouts
            # where labels can appear anywhere on screen.
            def _ocr_region(region):
                if region.size == 0: return []
                try:
                    out = self._get_ocr().readtext(region)
                    return [t.lower() for (_, t, c) in out if c > 0.25]
                except Exception:
                    return []

            # Stage 1a: fast partial scan
            partial_lines = (
                _ocr_region(img[0:int(h * 0.55), int(w * 0.45):]) +  # top-right
                _ocr_region(img[int(h * 0.65):, :])                   # bottom
            )
            detected = _detect_type_from_text(partial_lines)
            _slog.info(f'TextExtractor: partial scan → {detected!r} '
                       f'({len(partial_lines)} tokens)')

            if not detected:
                # Stage 1b: full image scan — needed for MIXED screens
                _slog.info('TextExtractor: partial scan inconclusive — scanning full image')
                full_lines = _ocr_region(img)
                detected = _detect_type_from_text(full_lines)
                all_lines = full_lines
                result['scan_scope'] = 'full'
                _slog.info(f'TextExtractor: full scan → {detected!r} '
                           f'({len(full_lines)} tokens)')
            else:
                all_lines = partial_lines
                result['scan_scope'] = 'partial'

            if detected:
                result['build_type'] = detected

            # ── Pass 2: wide-scan for ship info ───────────────────────────────
            # Scan entire top 20% of image — works regardless of where ship info
            # appears (left, centre, right, cropped screenshots)
            top_band = img[0:int(h * 0.20), :]
            try:
                ocr_out = self._get_ocr().readtext(top_band)
            except Exception as e:
                _slog.debug(f'TextExtractor OCR failed: {e}')
                return result

            if not ocr_out:
                return result

            # Sort all detections left-to-right, top-to-bottom
            ocr_out.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
            items = [(bbox, t.strip(), c) for (bbox, t, c) in ocr_out
                     if c > 0.20 and t.strip()]
            _slog.info(f'TextExtractor: {len(items)} OCR tokens in top band')
            for bbox, t, c in items:
                _slog.debug(f'  OCR: {t!r} conf={c:.2f} y={bbox[0][1]:.0f}')

            # ── Pre-compute per-token features ────────────────────────────────
            # xywh + dark-bg flag — ship_* labels sit on near-black panel BG.
            tokens = []
            for bbox, t, c in items:
                x, y, w_, h_ = self._poly_to_xywh(bbox)
                dark = self._is_dark_bg(img, (x, y, w_, h_))
                tokens.append({
                    'x': x, 'y': y, 'w': w_, 'h': h_,
                    'text': t, 'conf': c, 'dark': dark,
                    'cy': y + h_ // 2,
                })

            # Group tokens into visual rows by y-clustering.
            # Median token height defines tolerance; STO ship_* rows are ~14-22px.
            if tokens:
                med_h = float(np.median([t['h'] for t in tokens]))
                tol = max(6.0, med_h * 0.45)
                rows = []  # list[dict(cy, tokens)]
                for tok in sorted(tokens, key=lambda t: t['cy']):
                    placed = False
                    for r in rows:
                        if abs(tok['cy'] - r['cy']) <= tol:
                            r['tokens'].append(tok)
                            r['cy'] = float(np.mean([t['cy'] for t in r['tokens']]))
                            placed = True
                            break
                    if not placed:
                        rows.append({'cy': float(tok['cy']), 'tokens': [tok]})
                for r in rows:
                    r['tokens'].sort(key=lambda t: t['x'])
            else:
                rows = []

            def _is_blacklisted(text: str) -> bool:
                """True if token is a known HUD/tab/section word (never part of ship_*)."""
                low = text.lower().strip().strip(':')
                if low in _HUD_BLACKLIST:
                    return True
                # Substring check for compound HUD labels
                for bad in ('collapse', 'active space duty', 'active ground duty',
                            'personal space traits', 'personal ground traits',
                            'starship traits'):
                    if bad in low:
                        return True
                return False

            _SECTION_HEADER_RE = re.compile(
                r'\b(traits|reputation|abilities|bridge.?officer|boff|'
                r'equipment|consoles?|weapons?|devices?|kit\b|armor|'
                r'fore|aft|stations?|skills|status|details|summer)\b',
                re.IGNORECASE,
            )

            # Helpers operating on tokens.
            def _registry_token(text: str) -> bool:
                """Match registry like (NCC-1234), [NX-A-5], NCC-1517-A."""
                low = text.strip().strip('()[]')
                return bool(re.match(r'^[\[\(]?(NCC|NX|NCD|RRW)[-\s]', text, re.IGNORECASE)) \
                    or bool(re.fullmatch(r'[\(\[]?[A-Z]{2,3}[-\s].+', low))

            def _looks_like_name_token(text: str) -> bool:
                """
                Detect a token that belongs to the ship NAME (not type).
                Catches both: (a) explicit prefix `U.S.S. SHIP`, and
                (b) bare proper-name continuation `SIMONZ`, `LAZURITE`,
                where OCR split the name into multiple tokens.
                Bare names are: short (1-2 words), all-caps or capitalized,
                no spaces or with a single dot/apostrophe.
                """
                s = text.strip()
                if not s:
                    return False
                if _is_name_prefix_token(s):
                    return True
                # Standalone proper-name token: ≤14 chars, no spaces, mostly upper.
                if len(s) <= 14 and ' ' not in s:
                    letters = [ch for ch in s if ch.isalpha()]
                    if letters and sum(1 for ch in letters if ch.isupper()) / len(letters) >= 0.7:
                        return True
                return False

            # ── Anchor 1: tier token ──────────────────────────────────────────
            tier_row = None
            tier_tok = None
            for r in rows:
                for tok in r['tokens']:
                    m = RE_TIER_LOOSE.search(tok['text'])
                    if not m:
                        continue
                    tier_tok = tok
                    tier_row = r
                    result['ship_tier'] = m.group(1).upper().replace(' ', '')
                    result['ship_tier_bbox'] = (tok['x'], tok['y'], tok['w'], tok['h'])
                    _slog.info(f'TextExtractor: tier={result["ship_tier"]!r} from {tok["text"]!r}')
                    # Same-token prefix.
                    prefix = tok['text'][:m.start()].strip().rstrip(' [')
                    if len(prefix) > 4 and not _is_blacklisted(prefix):
                        result['ship_type'] = prefix
                        result['ship_type_bbox'] = (tok['x'], tok['y'], tok['w'], tok['h'])
                    break
                if tier_tok:
                    break

            # ── Anchor 1b: bracketed tier inside a single fused token ────────
            # Low-res screens often produce one wide token like
            # 'Aetherian Salvation [TB-X2]' — name+tier fused, with the digit
            # misread (T6 → TB, T6 → T8, etc.). RE_TIER_LOOSE can't catch the
            # malformed inner tier. Pull bracket content out and fuzzy-snap
            # against SHIP_TIER_VALUES; treat the prefix as ship_type.
            if tier_tok is None:
                import difflib as _df
                for r in rows:
                    for tok in r['tokens']:
                        m_br = re.search(r'\[([A-Za-z0-9\- ]{2,8})\]', tok['text'])
                        if not m_br:
                            continue
                        cand = m_br.group(1).upper().replace(' ', '')
                        matches = _df.get_close_matches(
                            cand, SHIP_TIER_VALUES, n=1, cutoff=0.5)
                        if not matches:
                            continue
                        tier_tok = tok
                        tier_row = r
                        result['ship_tier'] = matches[0]
                        result['ship_tier_bbox'] = (
                            tok['x'], tok['y'], tok['w'], tok['h'])
                        prefix = tok['text'][:m_br.start()].strip().rstrip(' [')
                        if len(prefix) > 4 and not _is_blacklisted(prefix):
                            result['ship_type'] = prefix
                            result['ship_type_bbox'] = (
                                tok['x'], tok['y'], tok['w'], tok['h'])
                        _slog.info(
                            f'TextExtractor: bracket-tier fuzzy '
                            f'{m_br.group(1)!r} → {matches[0]!r} '
                            f'(from {tok["text"]!r})')
                        break
                    if tier_tok:
                        break

            anchor_kind = ''
            anchor_x = anchor_y = anchor_w = anchor_h = None

            if tier_tok is not None:
                anchor_x = tier_tok['x']; anchor_y = tier_tok['y']
                anchor_w = tier_tok['w']; anchor_h = tier_tok['h']
                anchor_kind = 'tier'
            else:
                # ── Anchor 2: ship-name prefix row ────────────────────────────
                # A row is a name row if any token is a full prefix+name combo,
                # OR if any token is a prefix-only ('U.S.S.', 'U.s.5_') AND the
                # row has at least one more token (the bare name to its right).
                for r in rows:
                    hit = None
                    for tok in r['tokens']:
                        if _is_name_prefix_token(tok['text']):
                            hit = tok
                            break
                        if _is_name_prefix_alone(tok['text']) and len(r['tokens']) > 1:
                            hit = tok
                            break
                    if hit:
                        anchor_x = hit['x']; anchor_y = hit['y']
                        anchor_w = hit['w']; anchor_h = hit['h']
                        anchor_kind = 'name'
                        name_text, kept_toks = _name_text_from_row_tokens(r['tokens'])
                        result['ship_name'] = name_text
                        result['ship_name_bbox'] = self._union_xywh(
                            *((t['x'], t['y'], t['w'], t['h']) for t in kept_toks))
                        tier_row = r
                        _slog.info(f'TextExtractor: name-prefix anchor → {name_text!r}')
                        break

            if anchor_kind:
                col_pad = max(80, int(anchor_w * 2.0))
                col_lo = anchor_x - col_pad
                col_hi = anchor_x + anchor_w + col_pad

                def _in_column(tok) -> bool:
                    return (tok['x'] + tok['w']) > col_lo and tok['x'] < col_hi

                def _row_in_column(row) -> bool:
                    return any(_in_column(t) for t in row['tokens'])

                def _valid_type_tok(tok) -> bool:
                    # Dark-bg is NOT required here — column window + HUD
                    # blacklist already separate ship_* labels from UI overlays,
                    # and Status-tab panels can have light gradients.
                    if len(tok['text'].strip()) <= 2:
                        return False
                    if _is_blacklisted(tok['text']):
                        return False
                    if _SECTION_HEADER_RE.search(tok['text']):
                        return False
                    if _registry_token(tok['text']):
                        return False
                    if not _in_column(tok):
                        return False
                    return True

                # Determine which row(s) hold the ship_type.
                # Strategy:
                #   - tier anchor: prefer SAME row as tier (left of tier), then
                #     the row immediately above. Reject rows that are mostly
                #     name tokens (U.S.S. SIMONZ).
                #   - name anchor: row immediately below name.
                anchor_row_idx = rows.index(tier_row) if tier_row in rows else -1

                def _row_is_name_row(row) -> bool:
                    """A row that contains a name-prefix OR is dominated by
                    short bare-name tokens (SIMONZ, LAZURITE)."""
                    if any(_is_name_prefix_token(t['text']) for t in row['tokens']):
                        return True
                    if any(_is_name_prefix_alone(t['text']) for t in row['tokens']):
                        return True
                    if not row['tokens']:
                        return False
                    name_like = sum(1 for t in row['tokens']
                                    if _looks_like_name_token(t['text']))
                    return name_like == len(row['tokens'])

                def _row_to_type(row, exclude=None):
                    """Build type string from a row, skipping invalid tokens."""
                    excl = set(id(t) for t in (exclude or []))
                    kept = [t for t in row['tokens']
                            if id(t) not in excl and _valid_type_tok(t)
                            and not _looks_like_name_token(t['text'])]
                    if not kept:
                        return '', None
                    text = ' '.join(t['text'] for t in kept).strip()
                    bb = self._union_xywh(*((t['x'], t['y'], t['w'], t['h']) for t in kept))
                    return text, bb

                if anchor_kind == 'tier' and anchor_row_idx >= 0:
                    # Same row: tokens left-adjacent to the tier token, then
                    # extend upward (row above) when not the name row.
                    # Limit to direct neighbours of tier — tooltip text in the
                    # same y-band but far in x must NOT leak in.
                    def _adjacent_left_of(row, target_tok, max_gap_ratio=4.0):
                        """Pick tokens to the LEFT of target_tok, contiguous in x.
                        Stop at a horizontal gap > max_gap_ratio × target height."""
                        left = [t for t in row['tokens']
                                if t['x'] + t['w'] <= target_tok['x'] + 2]
                        left.sort(key=lambda t: t['x'], reverse=True)
                        gap_thr = max(40.0, target_tok['h'] * max_gap_ratio)
                        kept = []
                        prev_left = target_tok['x']
                        for t in left:
                            gap = prev_left - (t['x'] + t['w'])
                            if gap > gap_thr:
                                break
                            if not _valid_type_tok(t):
                                # Allow it to break adjacency only if it is a
                                # name-like token (likely the ship name to the
                                # left). For invalid HUD/blacklist tokens, stop.
                                break
                            kept.append(t)
                            prev_left = t['x']
                        kept.sort(key=lambda t: t['x'])
                        if not kept:
                            return '', None
                        text = ' '.join(t['text'] for t in kept).strip()
                        bb = self._union_xywh(
                            *((t['x'], t['y'], t['w'], t['h']) for t in kept))
                        return text, bb

                    same_text, same_bb = _adjacent_left_of(tier_row, tier_tok)
                    if same_text:
                        prefix_type = result.get('ship_type', '')
                        if prefix_type and prefix_type in same_text:
                            result['ship_type'] = same_text
                        elif prefix_type:
                            result['ship_type'] = (same_text + ' ' + prefix_type).strip()
                        else:
                            result['ship_type'] = same_text
                        result['ship_type_bbox'] = self._union_xywh(
                            same_bb, result.get('ship_type_bbox'))

                    # Extend up by 1 row when it contains the ship class name
                    # (Constitution, Chronos, Yamaguchi, Augur, Negh'Var, …).
                    # Guard: if any token is ALL-CAPS proper noun (SIMONZ,
                    # MORTIS, LAZURITE, FERASIA), the row is the ship NAME,
                    # not the type — skip.
                    def _has_allcaps_proper(row) -> bool:
                        # Split each OCR token by whitespace — OCR sometimes
                        # lumps a name and a HUD word into one token
                        # ('susurrus MORTIS'). We want MORTIS to flag the row.
                        for t in row['tokens']:
                            for word in t['text'].split():
                                s = word.strip()
                                if len(s) < 3 or len(s) > 18:
                                    continue
                                letters = [c for c in s if c.isalpha()]
                                if not letters:
                                    continue
                                up_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                                if up_ratio >= 0.80:
                                    return True
                        return False

                    if anchor_row_idx >= 1:
                        r_above = rows[anchor_row_idx - 1]
                        if (_row_in_column(r_above)
                                and not _row_is_name_row(r_above)
                                and not _has_allcaps_proper(r_above)):
                            text, bb = _row_to_type(r_above)
                            if text:
                                prefix_type = result.get('ship_type', '')
                                result['ship_type'] = (text + ' ' + prefix_type).strip() \
                                    if prefix_type else text
                                result['ship_type_bbox'] = self._union_xywh(
                                    bb, result.get('ship_type_bbox'))

                    # Name: topmost name-row in column above tier.
                    if not result.get('ship_name'):
                        for ri in range(anchor_row_idx - 1, -1, -1):
                            r_above = rows[ri]
                            if not _row_in_column(r_above):
                                continue
                            if _row_is_name_row(r_above):
                                name_text, kept_toks = _name_text_from_row_tokens(
                                    r_above['tokens'])
                                if not name_text:
                                    continue
                                result['ship_name'] = name_text
                                result['ship_name_bbox'] = self._union_xywh(
                                    *((t['x'], t['y'], t['w'], t['h'])
                                      for t in kept_toks))
                                break

                elif anchor_kind == 'name' and anchor_row_idx >= 0:
                    # Type: row immediately below name in same column.
                    for ri in range(anchor_row_idx + 1, len(rows)):
                        r_below = rows[ri]
                        if not _row_in_column(r_below):
                            continue
                        # Skip registry rows.
                        if all(_registry_token(t['text']) for t in r_below['tokens']):
                            continue
                        text, bb = _row_to_type(r_below)
                        if text:
                            result['ship_type'] = text
                            result['ship_type_bbox'] = bb
                            break

                _slog.info(f'TextExtractor: [{anchor_kind} anchor] '
                           f'name={result["ship_name"]!r} '
                           f'type={result["ship_type"]!r} '
                           f'tier={result["ship_tier"]!r}')
            else:
                # No anchor — no SPACE_EQ-like signal. Do NOT write any
                # token to ship_name: on SPEC / BOFFS / TRAITS screens the
                # topmost dark token is a tab label ('Commando', 'Space
                # Stations', …) and polluting ship_name breaks multi-screen
                # aggregation in WarpImporter.process_folder.
                _slog.info(f'TextExtractor: no anchor, ship info unset')

            # ── Infer build type if not already detected ──────────────────────
            if not result['build_type']:
                ship_type_lower = result['ship_type'].lower()
                if any(kw in ship_type_lower
                       for kw in ('ground', 'combat armor', 'kit')):
                    result['build_type'] = 'GROUND'
                else:
                    result['build_type'] = 'SPACE'

            # ── Strip trailing tier-bracket from type, recover tier ──────────
            # OCR sometimes lumps tier into the type token: "Foo Bar [T6-X2]"
            # or noisy variants "[TB-X2]" / "(T5-U)". ShipDB never wants the
            # bracket in the type string — strip it and fuzzy-snap its content
            # back into ship_tier when ship_tier is still empty.
            if result.get('ship_type'):
                m_br = re.search(
                    r'\s*[\[\(]([A-Za-z0-9\-\s]{2,8})[\]\)]\s*$',
                    result['ship_type'])
                if m_br:
                    inner = m_br.group(1).strip()
                    result['ship_type'] = result['ship_type'][:m_br.start()].strip()
                    if not result.get('ship_tier'):
                        import difflib as _df
                        # Build a candidate token from the bracket content,
                        # tolerating OCR noise like 'TB-X2' or 'T8-X2'.
                        cand = inner.upper().replace(' ', '')
                        matches = _df.get_close_matches(
                            cand, SHIP_TIER_VALUES, n=1, cutoff=0.5)
                        if matches:
                            result['ship_tier'] = matches[0]
                            _slog.info(f'TextExtractor: tier recovered from '
                                       f'bracket {inner!r} → {matches[0]!r}')

            # ── Apply community OCR corrections ───────────────────────────────
            if self._corrections:
                for key in ('ship_type', 'ship_tier'):
                    raw = result[key]
                    if raw and raw in self._corrections:
                        corrected = self._corrections[raw]
                        _slog.debug(f'TextExtractor: OCR correction {raw!r} → {corrected!r}')
                        result[key] = corrected

        except Exception as e:
            _slog.debug(f'TextExtractor: unexpected error: {e}')

        return result

    def _try_ship_name(self, img: np.ndarray, w: int, h: int, result: dict):
        """
        Attempt to extract ship name from top-left ROI
        and store in result['ship_name'] if not already set.
        """
        if result.get('ship_name'):
            return
        try:
            x2 = int(SHIP_INFO_ROI[2] * w)
            y2 = int(SHIP_INFO_ROI[3] * h)
            roi     = img[0:y2, 0:x2]
            ocr_out = self._get_ocr().readtext(roi)
            ocr_out.sort(key=lambda r: r[0][0][1])
            lines = [t.strip() for (_, t, c) in ocr_out
                     if c > 0.3 and t.strip()]
            if lines:
                result['ship_name'] = lines[0]
        except Exception:
            pass

    def refine_single_crop(self, crop_bgr: np.ndarray, slot: str,
                            valid_tiers: list[str] | None = None,
                            valid_types: list[str] | None = None
                            ) -> tuple[str, float, str]:
        """
        2× upscale → OCR → community correction → slot-specific fuzzy snap.
        Centralised so WARP (extract_ship_info refinement) and WARP CORE
        (per-bbox OCR worker triggered by user-drawn bbox) share one mechanism.

        Returns (text, conf, ocr_raw). Empty strings on failure.
        """
        import cv2
        import difflib

        if crop_bgr is None or crop_bgr.size == 0:
            return '', 0.0, ''
        try:
            crop_proc = cv2.resize(crop_bgr, None, fx=2.0, fy=2.0,
                                   interpolation=cv2.INTER_CUBIC)
            result = self._get_ocr().readtext(crop_proc)
            if not result:
                return '', 0.0, ''
            full_text = ' '.join(res[1] for res in result).strip()
            best_conf = max(res[2] for res in result)
            ocr_raw = full_text

            if full_text in self._corrections:
                full_text = self._corrections[full_text]
                best_conf = 1.0

            text, conf = full_text, best_conf
            if slot == 'Ship Tier':
                pool = valid_tiers or SHIP_TIER_VALUES
                m = re.search(RE_TIER, full_text)
                if m:
                    extracted = m.group(1).upper()
                    matches = difflib.get_close_matches(extracted, pool, n=1, cutoff=0.7)
                    if matches:
                        text, conf = matches[0], 1.0
                    else:
                        text, conf = '', 0.0
                else:
                    # RE_TIER failed (e.g. 'IT6-X21', 'T8-X2', 'Tb-X2' — OCR
                    # substitution errors). Fuzzy-snap the whole token directly
                    # against the tier pool — replaces hand-maintained typo
                    # entries in ship_type_corrections.json.
                    matches = difflib.get_close_matches(
                        full_text.upper(), pool, n=1, cutoff=0.5)
                    if matches:
                        text, conf = matches[0], 1.0
                    else:
                        text, conf = '', 0.0
            elif slot == 'Ship Type' and valid_types:
                matches = difflib.get_close_matches(full_text, valid_types, n=1, cutoff=0.6)
                if matches:
                    text, conf = matches[0], 1.0
                else:
                    text, conf = '', 0.0
            # Ship Name: full_text returned as-is.
            return text, conf, ocr_raw
        except Exception as e:
            _slog.debug(f'TextExtractor.refine_single_crop: {e}')
            return '', 0.0, ''

    def refine_ship_info(self, img: np.ndarray, info: dict,
                          valid_tiers: list[str] | None = None,
                          valid_types: list[str] | None = None) -> dict:
        """
        Fallback for extract_ship_info: when ship_tier / ship_type are empty
        but the corresponding bbox is set, run a focused 2× crop OCR with
        fuzzy snap against valid values. Mutates and returns `info` dict.

        Pure autodetection — same code that the trainer's per-bbox worker
        used to run inline. Shared by WARP and WARP CORE.
        """
        if img is None or not isinstance(info, dict):
            return info

        def _crop(bbox):
            x, y, w, h = bbox
            return img[max(0, y):y + h, max(0, x):x + w].copy()

        # Ship Tier fallback
        if not info.get('ship_tier') and info.get('ship_tier_bbox'):
            crop = _crop(info['ship_tier_bbox'])
            text, conf, raw = self.refine_single_crop(
                crop, 'Ship Tier', valid_tiers, valid_types)
            if text:
                info['ship_tier'] = text
                _slog.info(f'TextExtractor: tier refined from crop → {text!r} (raw={raw!r}, conf={conf:.2f})')

        # Ship Type fallback
        if not info.get('ship_type') and info.get('ship_type_bbox'):
            crop = _crop(info['ship_type_bbox'])
            text, conf, raw = self.refine_single_crop(
                crop, 'Ship Type', valid_tiers, valid_types)
            if text:
                info['ship_type'] = text
                _slog.info(f'TextExtractor: type refined from crop → {text!r} (raw={raw!r}, conf={conf:.2f})')

        return info

    def _get_ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._ocr
