# warp/recognition/text_extractor.py
# Extracts ship name, type and tier from the top-left block of STO Status tab.

from __future__ import annotations

import re
import logging
import numpy as np

log = logging.getLogger(__name__)

RE_TIER = re.compile(r'\[?(T[1-6](?:-(?:U|X|X2))?)\]?', re.IGNORECASE)

# ROI for ship info block (fraction of image)
SHIP_INFO_ROI = (0.0, 0.0, 0.34, 0.28)


class TextExtractor:

    def __init__(self):
        self._ocr = None

    def extract_ship_info(self, img: np.ndarray) -> dict:
        result = {'ship_name': '', 'ship_type': '', 'ship_tier': ''}
        try:
            h, w = img.shape[:2]
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
