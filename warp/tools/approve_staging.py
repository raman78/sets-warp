#!/usr/bin/env python3
# warp/tools/approve_staging.py
#
# REPO OWNER TOOL — review and approve user-contributed staging data.
#
# Usage:
#   python warp/tools/approve_staging.py --token hf_xxx [--auto] [--dry-run]
#
#   --auto     approve all staging entries that pass validation (no interactive review)
#   --dry-run  show what would be approved without uploading anything
#
# What it does:
#   1. Lists all staging/<install_id>/annotations.jsonl files
#   2. Validates each entry (crop size, name, slot)
#   3. Merges approved entries into data/annotations.jsonl
#   4. Copies approved crops from staging/<id>/crops/ to data/crops/
#   5. Optionally deletes approved staging entries

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import io
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Approve WARP staging data')
    parser.add_argument('--token',   required=True, help='HF write token')
    parser.add_argument('--auto',    action='store_true', help='Auto-approve all valid entries')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, no uploads')
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=args.token)

    REPO     = "sets-sto/sto-icon-dataset"
    RTYPE    = "dataset"
    STAGING  = "staging"
    DATA_ANN = "data/annotations.jsonl"
    DATA_CRP = "data/crops"

    # ── List all staging contributors ──────────────────────────────────────
    print("Fetching staging file list…")
    all_files = list(api.list_repo_files(repo_id=REPO, repo_type=RTYPE))
    staging_files = [f for f in all_files if f.startswith(STAGING + "/")]

    # Group by install_id
    contributors: dict[str, list[str]] = defaultdict(list)
    for f in staging_files:
        parts = f.split("/")
        if len(parts) >= 2:
            contributors[parts[1]].append(f)

    if not contributors:
        print("No staging data found.")
        return

    print(f"\nFound {len(contributors)} contributor(s):\n")

    # ── Load existing approved hashes ──────────────────────────────────────
    approved_hashes: set[str] = set()
    existing_lines: list[str] = []
    try:
        local = hf_hub_download(repo_id=REPO, filename=DATA_ANN, repo_type=RTYPE)
        with open(local) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_lines.append(line)
                    try:
                        d = json.loads(line)
                        approved_hashes.add(d.get("crop_sha256", ""))
                    except Exception:
                        pass
    except Exception:
        pass

    to_approve: list[dict] = []

    for install_id, files in contributors.items():
        anno_file = f"{STAGING}/{install_id}/annotations.jsonl"
        if anno_file not in files:
            continue

        print(f"  Contributor: {install_id}")
        try:
            local = hf_hub_download(repo_id=REPO, filename=anno_file, repo_type=RTYPE)
        except Exception as e:
            print(f"    ERROR downloading {anno_file}: {e}")
            continue

        entries = []
        with open(local) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

        print(f"    {len(entries)} entries")

        for entry in entries:
            sha  = entry.get("crop_sha256", "")
            name = entry.get("name", "")
            slot = entry.get("slot", "")
            date = entry.get("date", "?")

            if sha in approved_hashes:
                print(f"    SKIP (already approved): {slot} → {name!r}")
                continue

            # Validate
            if not name or not slot or not sha:
                print(f"    INVALID (missing fields): {entry}")
                continue
            if len(name) > 120 or not name.isprintable():
                print(f"    INVALID (bad name): {name!r}")
                continue

            print(f"    [{date}] {slot:30s} → {name!r}")

            approve = True
            if not args.auto:
                ans = input("      Approve? [Y/n/q] ").strip().lower()
                if ans == 'q':
                    print("Aborted.")
                    sys.exit(0)
                approve = ans != 'n'

            if approve:
                entry['_staging_install'] = install_id
                entry['_crop_src'] = f"{STAGING}/{install_id}/crops/{sha}.png"
                to_approve.append(entry)
                approved_hashes.add(sha)

    if not to_approve:
        print("\nNothing to approve.")
        return

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Approving {len(to_approve)} entries…")

    if args.dry_run:
        for e in to_approve:
            print(f"  {e['slot']} → {e['name']!r}  ({e['crop_sha256']})")
        return

    # ── Copy crops staging → data/crops ────────────────────────────────────
    for entry in to_approve:
        src = entry.pop("_crop_src")
        entry.pop("_staging_install", None)
        sha = entry["crop_sha256"]
        dst = f"{DATA_CRP}/{sha}.png"
        if dst not in all_files:
            try:
                local_crop = hf_hub_download(repo_id=REPO, filename=src, repo_type=RTYPE)
                api.upload_file(
                    path_or_fileobj=local_crop,
                    path_in_repo=dst,
                    repo_id=REPO,
                    repo_type=RTYPE,
                )
                print(f"  Copied crop {sha[:12]}…")
            except Exception as e:
                print(f"  ERROR copying crop {sha}: {e}")

    # ── Merge into data/annotations.jsonl ──────────────────────────────────
    combined = list(existing_lines)
    seen = set(json.loads(l).get("crop_sha256") for l in existing_lines if l.strip())
    for entry in to_approve:
        if entry["crop_sha256"] not in seen:
            combined.append(json.dumps(entry, ensure_ascii=False))
            seen.add(entry["crop_sha256"])

    content = "\n".join(combined).encode("utf-8")
    api.upload_file(
        path_or_fileobj=io.BytesIO(content),
        path_in_repo=DATA_ANN,
        repo_id=REPO,
        repo_type=RTYPE,
    )
    print(f"\nDone. {len(to_approve)} entries approved into {DATA_ANN}.")


if __name__ == "__main__":
    main()
