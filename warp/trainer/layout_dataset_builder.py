# warp/trainer/layout_dataset_builder.py
#
# Processes annotations into a structured dataset for CNN Layout Regression.

import json
import logging
from pathlib import Path
import cv2
import numpy as np

log = logging.getLogger(__name__)

class LayoutDatasetBuilder:
    def __init__(self, sets_root: Path):
        self.sets_root = sets_root
        self.training_dir = sets_root / 'warp' / 'training_data'
        self.annotations_path = self.training_dir / 'annotations.json'

    def build(self, build_type: str = 'SPACE') -> list[dict]:
        """
        Returns a list of samples: {
            'image_path': Path,
            'labels': dict[slot_name, bbox_normalized]
        }
        """
        if not self.annotations_path.exists():
            log.warning(f"Annotations not found at {self.annotations_path}")
            return []

        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for filename, annotations in data.items():
            img_path = self.training_dir / filename
            if not img_path.exists():
                continue

            # We only want screenshots that have a good number of confirmed slots 
            # to train a full layout
            confirmed = [a for a in annotations if a.get('state') == 'confirmed']
            if len(confirmed) < 5: # Need at least some density
                continue

            # Map slot names to normalized bboxes
            # We use image size to normalize
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            slot_labels = {}
            for ann in confirmed:
                slot = ann.get('slot')
                bbox = ann.get('bbox') # [x, y, w, h]
                if not slot or not bbox:
                    continue
                
                # Normalize to 0.0 - 1.0
                nx = bbox[0] / w
                ny = bbox[1] / h
                nw = bbox[2] / w
                nh = bbox[3] / h
                slot_labels[slot] = [nx, ny, nw, nh]

            samples.append({
                'image_path': img_path,
                'width': w,
                'height': h,
                'labels': slot_labels
            })

        log.info(f"Built layout dataset with {len(samples)} samples for {build_type}")
        return samples

if __name__ == '__main__':
    # Simple test
    logging.basicConfig(level=logging.INFO)
    builder = LayoutDatasetBuilder(Path('.').resolve())
    data = builder.build()
    print(f"Loaded {len(data)} samples")
