import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def classify_boff_profession(crop_bgr):
    ih, iw = crop_bgr.shape[:2]
    b = max(3, int(min(ih, iw) * 0.22))
    top    = crop_bgr[:b, :].reshape(-1, 3)
    bottom = crop_bgr[-b:, :].reshape(-1, 3)
    left   = crop_bgr[b:-b, :b].reshape(-1, 3)
    right  = crop_bgr[b:-b, -b:].reshape(-1, 3)
    border = np.concatenate([top, bottom, left, right])

    hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat_mask = (hsv[:, 1] > 80) & (hsv[:, 2] > 60)
    if sat_mask.sum() < 8:
        return 'science' # default / pure blue

    hues = hsv[sat_mask, 0]
    hist, _ = np.histogram(hues, bins=36, range=(0, 180))

    def _h(lo, hi):
        return int(hist[lo // 5: hi // 5 + 1].sum())

    red_lo    = _h(0,    9)
    red_hi    = _h(160, 175)
    red_total = red_lo + red_hi
    amber     = _h(10,  30)
    mid_blue  = _h(105, 120)
    green     = _h(48,  72)
    purple    = _h(115, 145)
    bg_blue   = _h(85, 120)

    color_scores = {
        'tactical': red_total,
        'engineering': amber,
        'temporal': amber + mid_blue,
        'miracle worker': green,
        'intelligence': purple,
        'science': bg_blue
    }
    
    best = max(color_scores.items(), key=lambda x: x[1])
    
    if best[1] < 10:
        return 'science'
    
    return best[0]

def main():
    training_data_dir = Path('/home/raman/PycharmProjects/sets-warp/warp/training_data')
    ann_file = training_data_dir / 'annotations.json'
    
    if not ann_file.exists():
        print("Annotations file not found.")
        return

    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = defaultdict(lambda: defaultdict(int))
    total = 0
    tested = 0
    
    for filename, annotations in data.items():
        if not isinstance(annotations, list): continue
        for ann in annotations:
            slot = ann.get('slot', '')
            if slot.startswith('Boff '):
                true_profession = slot.replace('Boff ', '').lower()
                crop_name = ann.get('crop_name')
                if not crop_name: continue
                
                crop_path = training_data_dir / crop_name
                if not crop_path.exists(): continue
                
                img = cv2.imread(str(crop_path))
                if img is None: continue
                
                pred = classify_boff_profession(img)
                
                results[true_profession][pred] += 1
                tested += 1

    print(f"Tested {tested} BOFF crops.")
    print(f"{'TRUE PROFESSION':<20} | {'PREDICTIONS':<40}")
    print("-" * 65)
    for true_prof, preds in results.items():
        preds_str = ", ".join(f"{k}:{v}" for k, v in preds.items())
        print(f"{true_prof:<20} | {preds_str}")

if __name__ == '__main__':
    main()
