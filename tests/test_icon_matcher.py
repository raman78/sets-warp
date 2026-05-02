import json
from pathlib import Path
import cv2
import numpy as np
import warnings

# Suppress annoying warnings
warnings.filterwarnings('ignore', category=UserWarning)

from warp.recognition.icon_matcher import SETSIconMatcher

# Mock SETS app for IconMatcher
class MockSetsApp:
    class MockCache:
        def __init__(self):
            # We need to build a cache that includes ALL BOFF abilities.
            # To simulate, we load them from data/cargo/boff_abilities.json
            self.boff_abilities = {}
            cargo_file = Path('/home/raman/PycharmProjects/sets-warp/data/cargo/boff_abilities.json')
            if cargo_file.exists():
                with open(cargo_file, 'r', encoding='utf-8') as f:
                    cargo_data = json.load(f)
                    for space_ground in ['space', 'ground']:
                        if space_ground not in cargo_data: continue
                        for prof, ranks in cargo_data[space_ground].items():
                            for rank_dict in ranks:
                                if not rank_dict: continue
                                for name in rank_dict.keys():
                                    self.boff_abilities[name] = {'name': name, 'profession': prof}
            print(f"Loaded {len(self.boff_abilities)} BOFF abilities from cargo.")

    def __init__(self):
        self.cache = self.MockCache()

def main():
    training_data_dir = Path('/home/raman/PycharmProjects/sets-warp/warp/training_data')
    ann_file = training_data_dir / 'annotations.json'
    
    if not ann_file.exists():
        print("Annotations file not found.")
        return

    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    app = MockSetsApp()
    matcher = SETSIconMatcher(app)
    
    # Restrict candidate_names to ONLY BOFF abilities
    all_boff_names = set(app.cache.boff_abilities.keys())
    
    confusion = {prof: {p: 0 for p in ['tactical', 'engineering', 'science', 'command', 'intelligence', 'pilot', 'temporal', 'miracle worker', 'unknown']} 
                 for prof in ['tactical', 'engineering', 'science', 'command', 'intelligence', 'pilot', 'temporal', 'miracle worker']}
    
    total = 0
    correct = 0
    
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
                
                total += 1
                
                # Match crop against all BOFF abilities
                pred_name, conf, _, _ = matcher.match(img, candidate_names=all_boff_names)
                
                pred_prof = 'unknown'
                if pred_name and pred_name in app.cache.boff_abilities:
                    pred_prof = app.cache.boff_abilities[pred_name].get('profession', 'unknown').lower()
                    if pred_prof == 'special':
                        # For specialist abilities, profession is in category
                        cat = app.cache.boff_abilities[pred_name].get('category', '').lower()
                        pred_prof = cat.replace(' bridge officer abilities', '').strip()
                
                if pred_prof not in confusion[true_profession]:
                    confusion[true_profession][pred_prof] = 0
                confusion[true_profession][pred_prof] += 1
                
                if pred_prof == true_profession:
                    correct += 1

    print(f"Tested {total} BOFF crops with IconMatcher.")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)" if total else "No crops.")
    print("\nTRUE PROFESSION      | PREDICTIONS                             ")
    print("-" * 65)
    for true_prof, preds in confusion.items():
        if sum(preds.values()) == 0: continue
        preds_str = ", ".join(f"{k}:{v}" for k, v in preds.items() if v > 0)
        print(f"{true_prof:<20} | {preds_str}")

if __name__ == '__main__':
    main()
