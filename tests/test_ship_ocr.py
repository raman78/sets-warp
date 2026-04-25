import json
from pathlib import Path
import cv2

from warp.recognition.text_extractor import TextExtractor

def main():
    training_data_dir = Path('/home/raman/PycharmProjects/sets-warp/warp/training_data')
    ann_file = training_data_dir / 'annotations.json'
    
    if not ann_file.exists():
        print("Annotations file not found.")
        return

    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    te = TextExtractor()
    
    total = 0
    found = 0
    
    print(f"{'FILENAME':<30} | {'NAME':<30} | {'TYPE':<40} | {'TIER':<10}")
    print("-" * 115)
    
    for img_path in training_data_dir.glob('screen_types/SPACE_MIXED/*.png'):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        total += 1
        res = te.extract_ship_info(img)
        name = res.get('ship_name', '')
        stype = res.get('ship_type', '')
        tier = res.get('ship_tier', '')
        
        if stype: found += 1
        
        name_str = name if name else ""
        type_str = stype if stype else ""
        tier_str = tier if tier else ""
        print(f"{img_path.name[:30]:<30} | {name_str[:30]:<30} | {type_str[:40]:<40} | {tier_str:<10}")
        
    print("-" * 115)
    print(f"Total Space Mixed images tested: {total}")
    print(f"Ship type extracted successfully: {found} ({found/total*100:.1f}%)" if total else "No images.")

if __name__ == '__main__':
    main()
