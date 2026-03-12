# warp/recognition/text_extractor.py
# Extracts ship name, type, tier and build_type from STO Status / Traits screenshots.

from __future__ import annotations

import re
import logging
import numpy as np

log = logging.getLogger(__name__)

RE_TIER = re.compile(r'\[?(T[1-6](?:-(?:U|X|X2))?)\]?', re.IGNORECASE)

# ROI for ship info block (fraction of image)
SHIP_INFO_ROI = (0.0, 0.0, 0.34, 0.28)

# Header labels that reveal the screenshot type
# Checked against OCR text of the full image (first pass, fast)
_TRAIT_HEADERS = {
    'personal space traits': 'SPACE_TRAITS',
    'personal ground traits': 'GROUND_TRAITS',
    'starship traits': 'SPACE_TRAITS',
    'space reputation': 'SPACE_TRAITS',
    'ground reputation': 'GROUND_TRAITS',
    'active reputation': 'SPACE_TRAITS',
}
_BOFF_HEADERS = {
    'bridge officer': 'BOFFS',
    'boff abilities': 'BOFFS',
    'tactical':       'BOFFS',   # right panel header in full STOCD view
}
_SPEC_HEADERS = {
    'primary specialization':   'SPEC',
    'secondary specialization': 'SPEC',
    'temporal operative':       'SPEC',  # typical spec name patterns
    'strategist':               'SPEC',
    'intelligence':             'SPEC',
    'commando':                 'SPEC',
}


def _detect_type_from_text(lines_lower: list[str]) -> str | None:
    """Return detected build_type from OCR lines, or None if unknown."""
    joined = ' '.join(lines_lower)
    for kw, btype in _TRAIT_HEADERS.items():
        if kw in joined:
            return btype
    for kw, btype in _BOFF_HEADERS.items():
        if kw in joined:
            return btype
    # Spec detection: two consecutive spec-like lines at bottom
    for kw, btype in _SPEC_HEADERS.items():
        if kw in joined:
            return btype
    return None


class TextExtractor:

    def __init__(self):
        self._ocr = None

    def extract_ship_info(self, img: np.ndarray) -> dict:
        result = {
            'ship_name': '',
            'ship_type': '',
            'ship_tier': '',
            'build_type': '',   # SPACE / GROUND / SPACE_TRAITS / GROUND_TRAITS / BOFFS / SPEC
        }
        try:
            h, w = img.shape[:2]

            # ── Quick full-image scan for section headers ─────────────────────
            # Use bottom 40% + right 60% where trait/boff headers appear
            # This is fast: small area, only text detection
            scan_regions = [
                img[int(h*0.0):int(h*0.5), int(w*0.5):],  # top-right (traits panel)
                img[int(h*0.7):,           :],              # bottom (spec names)
            ]
            all_ocr_lines = []
            for region in scan_regions:
                if region.size == 0:
                    continue
                try:
                    out = self._get_ocr().readtext(region)
                    all_ocr_lines += [t.lower() for (_, t, c) in out if c > 0.3]
                except Exception:
                    pass

            detected_type = _detect_type_from_text(all_ocr_lines)
            if detected_type:
                result['build_type'] = detected_type
                # For trait/boff/spec screenshots we still try to get ship name
                # from top-left if present
                try:
                    x1, y1 = 0, 0
                    x2, y2 = int(SHIP_INFO_ROI[2]*w), int(SHIP_INFO_ROI[3]*h)
                    roi = img[y1:y2, x1:x2]
                    ocr_out = self._get_ocr().readtext(roi)
                    ocr_out.sort(key=lambda r: r[0][0][1])
                    lines = [t.strip() for (_, t, c) in ocr_out if c > 0.3 and t.strip()]
                    if lines:
                        result['ship_name'] = lines[0]
                except Exception:
                    pass
                return result

            # ── Standard ship status tab ──────────────────────────────────────
            x1, y1 = int(SHIP_INFO_ROI[0]*w), int(SHIP_INFO_ROI[1]*h)
            x2, y2 = int(SHIP_INFO_ROI[2]*w), int(SHIP_INFO_ROI[3]*h)
            roi = img[y1:y2, x1:x2]
            ocr_out = self._get_ocr().readtext(roi)

        except Exception as e:
            log.debug(f'TextExtractor OCR failed: {e}')
            return result

        ocr_out.sort(key=lambda r: r[0][0][1])
        lines = [text.strip() for (_, text, conf) in ocr_out if conf > 0.3 and text.strip()]
        if not lines:
            return result

        result['ship_name'] = lines[0]
        for line in lines[1:]:
            m = RE_TIER.search(line)
            if m:
                result['ship_tier'] = m.group(1).upper().replace(' ', '')
                type_text = line[:m.start()].strip().rstrip(' [')
                if type_text:
                    result['ship_type'] = type_text
            elif not result['ship_type'] and len(line) > 4:
                result['ship_type'] = (result['ship_type'] + ' ' + line).strip()

        return result

    def _get_ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._ocr

