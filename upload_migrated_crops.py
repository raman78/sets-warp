#!/usr/bin/env python3
"""
upload_migrated_crops.py
========================
Upload SISTER-migrated crops to the HF staging area for central model training.

Reads all  warp/training_data/crops/migrated__*.png  files, parses the item
name from the filename, and uploads them to:

    staging/migration-sister/crops/{sha256}.png
    staging/migration-sister/annotations.jsonl

This makes them visible to admin_train.py which applies democratic voting
(1 install_id = 1 vote per class) — so this batch counts as exactly 1
authoritative vote per unique item, regardless of how many copies exist.

Run from the sets-warp root:
    .venv/bin/python upload_migrated_crops.py --dry-run   # preview
    .venv/bin/python upload_migrated_crops.py             # upload

HF token is read from:
    1. warp/hub_token.txt  (preferred — same token WARP CORE uses for sync)
    2. ../sets-warp-backend/.env  (HF_TOKEN= line)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, timezone, datetime
from pathlib import Path

CROPS_DIR    = Path('warp/training_data/crops')
INSTALL_ID   = 'migration-sister'
HF_DATASET   = 'sets-sto/sto-icon-dataset'
HF_REPO_TYPE = 'dataset'
STAGING_DIR  = f'staging/{INSTALL_ID}'
BATCH_SIZE   = 150   # files per HF commit (keep under API limits)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_token() -> str:
    """Read HF write token from hub_token.txt or sets-warp-backend/.env."""
    candidates = [
        Path('warp/hub_token.txt'),
        Path('../sets-warp-backend/.env'),
    ]
    for p in candidates:
        if not p.exists():
            continue
        text = p.read_text().strip()
        if p.suffix == '.txt':
            if text:
                return text
        else:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith('HF_TOKEN=') and '=' in line:
                    return line.split('=', 1)[1].strip()
    return ''


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _parse_item_name(filename: str) -> str:
    """
    Extract and normalise item name from migrated__{name}__{hash}.png.

    Example:
        'migrated__furiadon_fangs_mk_xiii_[crtd]__b9a9026d746c.png'
        → 'Furiadon Fangs Mk XIII [CRTD]'
    """
    stem = Path(filename).stem                    # drop .png
    stem = stem.removeprefix('migrated__')
    last_sep = stem.rfind('__')
    if last_sep == -1:
        return ''
    name = stem[:last_sep].replace('_', ' ').strip()
    if not name:
        return ''
    # Title-case words; restore common uppercase tokens like [CRTD], MK, XII
    words = []
    for w in name.split():
        if w.startswith('[') and w.endswith(']'):
            words.append(w.upper())         # [crtd] → [CRTD]
        elif w.lower() in ('mk', 'ii', 'iii', 'iv', 'vi', 'vii', 'viii',
                           'ix', 'xi', 'xii', 'xiii', 'xiv', 'xv'):
            words.append(w.upper())         # mk → MK, xii → XII
        else:
            words.append(w.capitalize())
    return ' '.join(words)


def _fetch_existing_hashes(api, token: str) -> set[str]:
    """Return set of sha256 strings already uploaded to staging crops."""
    staging_crop = f'{STAGING_DIR}/crops'
    try:
        files = list(api.list_repo_files(HF_DATASET, repo_type=HF_REPO_TYPE))
        prefix = staging_crop + '/'
        return {
            f[len(prefix):].removesuffix('.png')
            for f in files
            if f.startswith(prefix) and f.endswith('.png')
        }
    except Exception as e:
        print(f'  Warning: could not fetch existing hashes: {e}')
        return set()


def _fetch_existing_anno_shas(token: str) -> set[str]:
    """Return set of crop_sha256 already recorded in annotations.jsonl."""
    from huggingface_hub import hf_hub_download
    anno_path = f'{STAGING_DIR}/annotations.jsonl'
    try:
        local = hf_hub_download(
            HF_DATASET, anno_path, repo_type=HF_REPO_TYPE, token=token,
        )
        shas = set()
        for line in Path(local).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                try:
                    shas.add(json.loads(line)['crop_sha256'])
                except Exception:
                    pass
        return shas
    except Exception:
        return set()


def _load_existing_anno_lines(token: str) -> list[str]:
    """Return current lines of annotations.jsonl (to preserve on append)."""
    from huggingface_hub import hf_hub_download
    anno_path = f'{STAGING_DIR}/annotations.jsonl'
    try:
        local = hf_hub_download(
            HF_DATASET, anno_path, repo_type=HF_REPO_TYPE, token=token,
        )
        return [l for l in Path(local).read_text(encoding='utf-8').splitlines() if l.strip()]
    except Exception:
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Upload SISTER-migrated crops to HF staging for central training.'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and preview — do not upload anything.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Crops per HF commit (default {BATCH_SIZE}).')
    args = parser.parse_args()

    # ── Discover migrated crops ───────────────────────────────────────────────
    migrated = sorted(CROPS_DIR.glob('migrated__*.png'))
    if not migrated:
        print('No migrated crops found in', CROPS_DIR)
        return

    print(f'Found {len(migrated)} migrated crops')

    items: list[tuple[Path, str]] = []
    skipped_no_name = 0
    for path in migrated:
        name = _parse_item_name(path.name)
        if not name:
            skipped_no_name += 1
            continue
        items.append((path, name))

    if skipped_no_name:
        print(f'  Skipped {skipped_no_name} crops with unparseable names')
    print(f'  {len(items)} crops with valid names')

    # Deduplicate by sha256 — identical PNGs with different labels would cause
    # conflicting votes in admin_train.py; keep first occurrence of each sha.
    seen_shas: dict[str, str] = {}   # sha256 → first item_name seen
    deduped: list[tuple[Path, str]] = []
    skipped_dup = 0
    for path, name in items:
        sha = _sha256(path)
        if sha in seen_shas:
            skipped_dup += 1
            # Log collision only when names differ (same icon, different label)
            if seen_shas[sha] != name:
                print(f'  Dup SHA, different names: "{seen_shas[sha]}" vs "{name}" — keeping first')
        else:
            seen_shas[sha] = name
            deduped.append((path, name))
    if skipped_dup:
        print(f'  Deduplicated {skipped_dup} crops with identical PNG content ({len(deduped)} unique)')
    items = deduped

    # ── Dry-run preview ───────────────────────────────────────────────────────
    if args.dry_run:
        print('\n--- DRY RUN (first 20 crops) ---')
        for path, name in items[:20]:
            print(f'  {path.name}')
            print(f'    → "{name}"')
        if len(items) > 20:
            print(f'  ... and {len(items) - 20} more')

        # Count unique names
        unique_names = {name for _, name in items}
        print(f'\nUnique item classes: {len(unique_names)}')
        print('No files uploaded (--dry-run).')
        return

    # ── HF setup ─────────────────────────────────────────────────────────────
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        print('ERROR: pip install huggingface-hub', file=sys.stderr)
        sys.exit(1)

    token = _read_token()
    if not token:
        print(
            'ERROR: HF token not found.\n'
            '  Put your write token in warp/hub_token.txt\n'
            '  or in ../sets-warp-backend/.env as HF_TOKEN=...',
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)

    print(f'\nTarget: {HF_DATASET}  →  {STAGING_DIR}/')
    print('Fetching already-uploaded hashes...')
    existing_hashes = _fetch_existing_hashes(api, token)
    existing_anno   = _fetch_existing_anno_shas(token)
    already_done    = existing_hashes | existing_anno
    print(f'  {len(already_done)} crops already on HF — will skip these')

    # ── Filter to new items only ──────────────────────────────────────────────
    to_upload: list[tuple[Path, str, str]] = []   # (path, name, sha256)
    for path, name in items:
        sha = _sha256(path)
        if sha in already_done:
            continue
        to_upload.append((path, name, sha))

    print(f'{len(to_upload)} new crops to upload')
    if not to_upload:
        print('Nothing to do — all crops already uploaded.')
        return

    # ── Upload in batches ─────────────────────────────────────────────────────
    staging_crop = f'{STAGING_DIR}/crops'
    anno_path    = f'{STAGING_DIR}/annotations.jsonl'
    today        = date.today().isoformat()

    total      = len(to_upload)
    uploaded   = 0
    batch_num  = 0
    anno_lines = _load_existing_anno_lines(token)   # preserve existing annotations

    for batch_start in range(0, total, args.batch_size):
        batch     = to_upload[batch_start: batch_start + args.batch_size]
        batch_num += 1
        n         = len(batch)
        end_idx   = batch_start + n

        print(f'\nBatch {batch_num}: crops {batch_start + 1}–{end_idx} / {total}')

        operations       = []
        new_anno_entries = []

        for path, name, sha in batch:
            operations.append(CommitOperationAdd(
                path_in_repo=f'{staging_crop}/{sha}.png',
                path_or_fileobj=str(path),
            ))
            new_anno_entries.append({
                'slot':        'migrated',   # no slot info from SISTER
                'name':        name,
                'crop_sha256': sha,
                'date':        today,
                'source':      'migration-sister',
            })

        # Rebuild annotations.jsonl: existing lines + new entries
        anno_lines += [json.dumps(e, ensure_ascii=False) for e in new_anno_entries]
        full_jsonl  = '\n'.join(anno_lines) + '\n'
        operations.append(CommitOperationAdd(
            path_in_repo=anno_path,
            path_or_fileobj=full_jsonl.encode('utf-8'),
        ))

        try:
            api.create_commit(
                repo_id=HF_DATASET,
                repo_type=HF_REPO_TYPE,
                operations=operations,
                commit_message=(
                    f'migration-sister: batch {batch_num} '
                    f'({n} crops, {batch_start + 1}–{end_idx}/{total})'
                ),
            )
            uploaded += n
            print(f'  OK — {uploaded}/{total} uploaded total')
        except Exception as e:
            print(f'  ERROR in batch {batch_num}: {e}', file=sys.stderr)
            print('  Stopping — re-run to resume (already-uploaded crops are skipped).')
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    unique_classes = len({name for _, name, _ in to_upload[:uploaded]})
    print(f'\nDone: {uploaded}/{total} crops uploaded')
    print(f'      ~{unique_classes} unique item classes added to staging/{INSTALL_ID}/')
    print(f'\nNext step: trigger central training in sets-warp-backend/')
    print(f'  cd ../sets-warp-backend')
    print(f'  .venv/bin/python admin_train.py --train --min 1')


if __name__ == '__main__':
    main()
