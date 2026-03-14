# migrate_sister_data.py
import sys
import json
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
import urllib.parse

# --- Configuration ---
SISTER_ROOT = Path("/home/raman/PycharmProjects/SISTER")
WARP_ROOT = Path("/home/raman/PycharmProjects/sets-warp")
# Path where SETS stores its downloaded images
SETS_IMAGES_DIR = WARP_ROOT / ".config" / "images"
# --- End Configuration ---

def migrate():
    """
    Migrates icon data from SETS local cache using SISTER's item list as a map.
    This avoids Cloudflare blocks by using already available local data.
    """
    print("--- Starting Local SISTER to WARP Data Migration ---")

    # 1. Add WARP to path
    sys.path.insert(0, str(WARP_ROOT))
    try:
        from warp.trainer.training_data import TrainingDataManager, AnnotationState
    except ImportError as e:
        print(f"Error: Could not import WARP modules. Check WARP_ROOT path.\nDetails: {e}")
        return

    # 2. Check if SISTER cache exists
    sister_image_cache_path = SISTER_ROOT / "sister_sto" / "resources" / "cache" / "image_cache.default.json"
    if not sister_image_cache_path.exists():
        print(f"Error: SISTER image cache not found at: {sister_image_cache_path}")
        return

    # 3. Check if SETS images directory exists
    if not SETS_IMAGES_DIR.exists():
        print(f"Error: SETS image directory not found at: {SETS_IMAGES_DIR}")
        print("Please run SETS and ensure some items are loaded/downloaded first.")
        return

    print(f"Local sources found. SETS images: {SETS_IMAGES_DIR}")

    # 4. Prepare WARP destination
    warp_data_dir = WARP_ROOT / "warp" / "training_data"
    migrated_images_dir = warp_data_dir / "migrated_images"
    migrated_images_dir.mkdir(parents=True, exist_ok=True)

    data_mgr = TrainingDataManager(warp_data_dir)
    print("WARP TrainingDataManager initialized.")

    # 5. Load items from SISTER
    with open(sister_image_cache_path, 'r', encoding='utf-8') as f:
        sister_cache = json.load(f)
    print(f"Loaded {len(sister_cache)} item definitions from SISTER's cache.")

    # 6. Migrate found files
    migrated_count = 0
    for item_data in tqdm(sister_cache, desc="Migrating Icons"):
        # SISTER names are often cleaner for the model
        item_name = item_data.get("name")
        if not item_name:
            continue

        # SETS stores images using quote_plus(item_name)
        # Note: We need to handle potential differences in naming between SISTER and SETS
        sets_filename = f"{urllib.parse.quote_plus(item_name)}.png"
        source_image_path = SETS_IMAGES_DIR / sets_filename

        # If not found directly, try cleaned_name from SISTER
        if not source_image_path.exists():
            cleaned_name = item_data.get("cleaned_name")
            if cleaned_name:
                sets_filename = f"{urllib.parse.quote_plus(cleaned_name)}.png"
                source_image_path = SETS_IMAGES_DIR / sets_filename

        if not source_image_path.exists():
            continue

        # Target filename in WARP training data
        safe_name = item_name.replace(' ', '_').replace(':', '')
        dest_filename = f"{safe_name}_{migrated_count}.png"
        dest_image_path = migrated_images_dir / dest_filename

        try:
            shutil.copy2(source_image_path, dest_image_path)
            
            img = cv2.imread(str(dest_image_path))
            if img is None:
                dest_image_path.unlink()
                continue
                
            h, w, _ = img.shape
            bbox = (0, 0, w, h)

            data_mgr.add_annotation(
                image_path=dest_image_path,
                bbox=bbox,
                slot="MIGRATED",
                name=item_name,
                state=AnnotationState.CONFIRMED,
            )
            migrated_count += 1
        except Exception as e:
            if dest_image_path.exists():
                dest_image_path.unlink()
            continue

    print(f"\nMigration complete. Migrated {migrated_count} icons from local cache.")
    if migrated_count > 0:
        data_mgr.save()
        print("Done. You can now train the WARP model.")
    else:
        print("No matches found between SISTER items and SETS images.")
        print("Tip: Ensure SETS has downloaded some icons into .config/images/")

if __name__ == "__main__":
    migrate()
