#!/usr/bin/env python3
"""Run: python -m warp.tools.debug_cargo"""
import json, sys
from pathlib import Path

cargo_dir = Path('.config/cargo')
if not cargo_dir.exists():
    print(f"ERROR: {cargo_dir} not found"); sys.exit(1)

for path in sorted(cargo_dir.glob('*.json')):
    data = json.loads(path.read_text())
    t = type(data).__name__
    print(f'\n{"="*55}')
    print(f'FILE: {path.name}  type={t}')
    if isinstance(data, list):
        print(f'  len={len(data)}')
        if data:
            print(f'  [0] keys: {list(data[0].keys()) if isinstance(data[0],dict) else type(data[0]).__name__}')
            for row in data[:2]:
                print(f'  row: {json.dumps(row, ensure_ascii=False)[:300]}')
    elif isinstance(data, dict):
        keys = list(data.keys())
        print(f'  keys({len(keys)}): {keys[:5]}')
        v0 = data[keys[0]] if keys else None
        print(f'  [{keys[0]!r}] = {type(v0).__name__}: {str(v0)[:200]}')
        if isinstance(v0, dict):
            sk = list(v0.keys())
            print(f'    subkeys: {sk[:6]}')
