#!/usr/bin/env python3
"""
Quick sanity check on warp/data/item_db.json
Run: python -m warp.tools.check_db
"""
import json, sys
from pathlib import Path
from urllib.parse import quote_plus

db_path = Path('warp/data/item_db.json')
icon_dir = Path('warp/data/icons')

if not db_path.exists():
    print(f"ERROR: {db_path} not found. Run scraper first.")
    sys.exit(1)

db = json.loads(db_path.read_text())
print(f"Total items: {len(db)}")

# Check icon coverage
with_local_icon = 0
with_url_icon   = 0
no_icon         = 0
missing_local   = 0

for name, data in db.items():
    local = icon_dir / (quote_plus(name) + '.png')
    has_local = local.exists()
    has_url   = bool(data.get('icon_url'))
    
    if has_local:       with_local_icon += 1
    elif has_url:       with_url_icon += 1; missing_local += 1
    else:               no_icon += 1

print(f"Local icon (.png):  {with_local_icon}")
print(f"URL only (no png):  {with_url_icon}")
print(f"No icon at all:     {no_icon}")

# Show sample items with icon_name → icon_url
print("\nSample starship traits with icon_url:")
count = 0
for name, data in db.items():
    if data.get('category') == 'starship_traits' and data.get('icon_url'):
        print(f"  {name!r}")
        print(f"    icon_url: {data['icon_url']}")
        count += 1
        if count >= 5: break

if count == 0:
    print("  (none have icon_url — icon_name was null in cargo)")

print("\nSample traits with icon_url:")
count = 0
for name, data in db.items():
    if data.get('category') == 'traits' and data.get('icon_url'):
        print(f"  {name!r}  →  {data['icon_url']}")
        count += 1
        if count >= 5: break

if count == 0:
    print("  (none have icon_url)")

# Check local icon name match vs db names
print(f"\nIcon dir: {icon_dir}")
print(f"PNGs in icon dir: {len(list(icon_dir.glob('*.png')))}")
print("\nSample icon filenames (decoded):")
from urllib.parse import unquote_plus
for p in sorted(icon_dir.glob('*.png'))[:8]:
    decoded = unquote_plus(p.stem)
    in_db = decoded in db
    print(f"  {p.name}  →  '{decoded}'  in_db={in_db}")

# Check boff abilities
print("\nBoff abilities in db:")
boffs = [n for n, d in db.items() if d.get('category') == 'boff_abilities']
print(f"  count: {len(boffs)}")
print(f"  samples: {boffs[:5]}")
