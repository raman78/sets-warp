#!/usr/bin/env python3
"""
scripts/upstream_sync.py — Semi-automated upstream SETS merge helper for SETS-WARP.

Fetches upstream/main, classifies changed files, auto-applies documented
patches from docs/src_patches.md, and reports what still needs manual work.

Usage:
    python scripts/upstream_sync.py              # dry-run: recon + patch preview
    python scripts/upstream_sync.py --apply      # create branch + apply patches + commit
    python scripts/upstream_sync.py --apply --branch my-branch-name

Auto-applied patches (LOW/MEDIUM risk):
    src/constants.py      SEVEN_DAYS_IN_SECONDS, GITHUB_CACHE_URL, TRAIT_QUERY_URL, SPECIES
    src/callbacks.py      log import, _EQUIPMENT_CATS, _TRAIT_CATS, _save/_restore_session_slots
    src/buildupdater.py   Intel Holoship fix, item normalization, boff alias-safe lookup
    src/datafunctions.py  trait key migration

Manual review required (HIGH risk — see docs/src_patches.md):
    src/app.py            Downloader/CargoManager/ImageManager init, hooks, splash
    src/iofunc.py         ReturnValueThread, PNG helpers, lxml imports
    src/widgets.py        ImageLabel, TooltipLabel, Cache.alt_images
    src/splash.py         take our version entirely
    src/datafunctions.py  icon_name → cache.alt_images (after trait migration auto-patch)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent


# ── Git helpers ───────────────────────────────────────────────────────────────

def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ['git', *args], cwd=ROOT,
        capture_output=True, text=True, check=check,
    )
    return result.stdout.strip()


def git_file(ref: str, path: str) -> str:
    """Read file content at a given git ref."""
    return git('show', f'{ref}:{path}')


def changed_files_upstream() -> list[str]:
    """Files changed between HEAD and upstream/main."""
    out = git('diff', '--name-only', 'HEAD', 'upstream/main')
    return [f for f in out.splitlines() if f]


def new_commits_upstream() -> list[str]:
    """Commits in upstream/main not in HEAD."""
    out = git('log', '--oneline', 'upstream/main', '^HEAD')
    return out.splitlines()


# ── Patch infrastructure ──────────────────────────────────────────────────────

@dataclass
class PatchResult:
    success: bool
    message: str
    content: str | None = None


@dataclass
class Patch:
    file: str
    description: str
    risk: str                              # LOW / MEDIUM / HIGH
    apply_fn: Callable[[str], PatchResult]


# ── Our-only files: never in upstream, always preserved ──────────────────────

OUR_ONLY_FILES = {
    'src/cargomanager.py',
    'src/downloader.py',
    'src/imagemanager.py',
    'src/setsdebug.py',
    'src/syncmanager.py',
}

# ── Files that need manual re-apply (see docs/src_patches.md) ────────────────

MANUAL_FILES: dict[str, str] = {
    'src/app.py': (
        'HIGH — Downloader/CargoManager/ImageManager init, Cloudflare cookies, '
        'cache_item_aliases, _set_win32_taskbar_icon, dual_cannons icon, '
        'setup_splash progress bar, menu_layout/settings_scroll_layout/settings_scroll_frame hooks.\n'
        '       See docs/src_patches.md → src/app.py'
    ),
    'src/iofunc.py': (
        'MEDIUM — ReturnValueThread class, _is_valid_png helpers, auto_backup_cargo_file, '
        'lxml imports (replaces requests_html).\n'
        '          See docs/src_patches.md → src/iofunc.py'
    ),
    'src/widgets.py': (
        'MEDIUM — ImageLabel(QLabel) with _update_pixmap, ShipImage(QLabel), '
        'TooltipLabel class, Cache.alt_images in reset_cache, dc TooltipLabel in WidgetStorage.\n'
        '          See docs/src_patches.md → src/widgets.py'
    ),
    'src/splash.py': (
        'LOW — take our version entirely (upstream has simple splash; ours has progress bar + ETA).\n'
        '      Steps: git show HEAD:src/splash.py > /tmp/our_splash.py, '
        'then git checkout upstream/main -- src/splash.py, '
        'then cp /tmp/our_splash.py src/splash.py'
    ),
    'src/datafunctions.py': (
        'HIGH — icon_name → cache.alt_images in trait loading loop, '
        'cache_item_aliases() call in populate_cache.\n'
        '        Trait key migration is auto-applied separately.\n'
        '        See docs/src_patches.md → src/datafunctions.py'
    ),
}


# ── Patch functions ───────────────────────────────────────────────────────────
# Each takes upstream file content (str), returns PatchResult.
# Must be idempotent: if already applied, return success with original content.

# ── src/constants.py ─────────────────────────────────────────────────────────

def patch_constants_new_consts(content: str) -> PatchResult:
    if 'SEVEN_DAYS_IN_SECONDS' in content:
        return PatchResult(True, 'already applied', content)
    anchor = 'STARSHIP_TRAIT_QUERY_URL'
    if anchor not in content:
        return PatchResult(False, f'anchor "{anchor}" not found — upstream may have renamed it')
    insert = (
        'SEVEN_DAYS_IN_SECONDS = 60 * 60 * 24 * 7\n'
        "GITHUB_CACHE_URL = 'https://github.com/STOCD/SETS-Data/raw/refs/heads/main'\n"
    )
    return PatchResult(True, 'inserted SEVEN_DAYS_IN_SECONDS + GITHUB_CACHE_URL',
                       content.replace(anchor, insert + anchor, 1))


def patch_constants_trait_query(content: str) -> PatchResult:
    if 'icon_name=icon_name' in content:
        return PatchResult(True, 'already applied', content)
    # Try to find and replace the upstream TRAIT_QUERY_URL fields string
    old = (
        "    WIKI_URL + 'Special:CargoExport?tables=Traits&fields=_pageName%3DPage,name,chartype,'\n"
        "    'environment,type,isunique,description&limit=2500&format=json'\n"
    )
    new = (
        "    WIKI_URL + 'Special:CargoExport?tables=Traits&fields=_pageName%3DPage,name,type,'\n"
        "    'environment,description,icon_name=icon_name&limit=2500&format=json'\n"
    )
    if old not in content:
        return PatchResult(False,
            'TRAIT_QUERY_URL fields anchor not found — upstream may have changed the query')
    return PatchResult(True, 'updated TRAIT_QUERY_URL: added icon_name, removed chartype/isunique',
                       content.replace(old, new, 1))


def patch_constants_species(content: str) -> PatchResult:
    if "'Caitian'" in content:
        return PatchResult(True, 'already applied', content)
    # Federation
    old_fed = (
        "    'Federation': {\n"
        "        'Human', 'Andorian', 'Bajoran', 'Benzite', 'Betazoid', 'Bolian', 'Ferengi', 'Pakled',\n"
        "        'Rigelian', 'Saurian', 'Tellarite', 'Trill', 'Joined Trill', 'Vulcan', 'Alien',\n"
        "        'Liberated Borg'\n"
        "    },"
    )
    new_fed = (
        "    'Federation': {\n"
        "        'Human', 'Andorian', 'Bajoran', 'Benzite', 'Betazoid', 'Bolian', 'Caitian', 'Ferengi',\n"
        "        'Pakled', 'Rigelian', 'Saurian', 'Talaxian', 'Tellarite', 'Trill', 'Joined Trill',\n"
        "        'Vulcan', 'Alien', 'Liberated Borg', 'Klingon'\n"
        "    },"
    )
    if old_fed not in content:
        return PatchResult(False, 'Federation SPECIES block not found — upstream may have changed it')
    content = content.replace(old_fed, new_fed, 1)
    # Klingon (single-line in upstream)
    old_kdf = "    'Klingon': {'Klingon', 'Gorn', 'Lethean', 'Nausicaan', 'Orion', 'Alien', 'Liberated Borg'},"
    new_kdf = (
        "    'Klingon': {\n"
        "        'Klingon', 'Gorn', 'Lethean', 'Nausicaan', 'Orion', 'Alien', 'Liberated Borg',\n"
        "        'Ferasan', 'Talaxian'\n"
        "    },"
    )
    if old_kdf not in content:
        return PatchResult(False, 'Klingon SPECIES line not found — upstream may have changed it')
    return PatchResult(True, 'expanded Federation + Klingon SPECIES sets',
                       content.replace(old_kdf, new_kdf, 1))


# ── src/callbacks.py ──────────────────────────────────────────────────────────

def patch_callbacks_log_import(content: str) -> PatchResult:
    if 'from .setsdebug import log' in content:
        return PatchResult(True, 'already applied', content)
    lines = content.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith(('from ', 'import ')):
            insert_at = i + 1
    lines.insert(insert_at, 'from .setsdebug import log\n')
    return PatchResult(True, 'added setsdebug log import', ''.join(lines))


_CALLBACKS_ADDITIONS = '''\
_EQUIPMENT_CATS = (
    'fore_weapons', 'aft_weapons', 'experimental', 'devices', 'hangars',
    'sec_def', 'deflector', 'engines', 'core', 'shield',
    'uni_consoles', 'eng_consoles', 'sci_consoles', 'tac_consoles',
)
_TRAIT_CATS = ('traits', 'starship_traits', 'rep_traits', 'active_rep_traits')


def _save_session_slots(self):
    """Snapshot current space build slots before ship switch."""
    import copy
    s = self.build['space']
    equip = {}
    for key in _EQUIPMENT_CATS:
        val = s.get(key)
        equip[key] = copy.deepcopy(val) if isinstance(val, (list, dict)) else val
    traits = {k: copy.deepcopy(v) for k in _TRAIT_CATS
              if isinstance((v := s.get(k)), list)}
    boffs = []
    for boff_id in range(6):
        spec = tuple(s['boff_specs'][boff_id]) if boff_id < len(s['boff_specs']) else None
        abilities = copy.deepcopy(s['boffs'][boff_id]) if boff_id < len(s['boffs']) else []
        boffs.append({'spec': spec, 'abilities': abilities})
    self._session_slots = {'equipment': equip, 'traits': traits, 'boffs': boffs}
    log.info('_save_session_slots: snapshot saved')


def _restore_session_slots(self):
    """Restore snapshot into current build (empty slots only)."""
    if not getattr(self, '_session_slots', None):
        return
    saved = self._session_slots
    s = self.build['space']
    for key, saved_val in saved['equipment'].items():
        current = s.get(key)
        if isinstance(current, list) and isinstance(saved_val, list):
            for i in range(min(len(current), len(saved_val))):
                if current[i] in ('', None) and saved_val[i] not in ('', None):
                    slot_equipment_item(self, saved_val[i], 'space', key, i)
        elif isinstance(current, dict) and isinstance(saved_val, dict):
            for i, item in saved_val.items():
                if current.get(i) in ('', None) and item not in ('', None):
                    slot_equipment_item(self, item, 'space', key, i)
        elif current in ('', None) and saved_val not in ('', None):
            slot_equipment_item(self, saved_val, 'space', key, 0)
    for key, saved_list in saved['traits'].items():
        current = s.get(key)
        if not (isinstance(current, list) and isinstance(saved_list, list)):
            continue
        for i in range(min(len(current), len(saved_list))):
            if current[i] in ('', None) and saved_list[i] not in ('', None):
                slot_trait_item(self, saved_list[i], 'space', key, i)
    for boff_id, saved_boff in enumerate(saved['boffs']):
        if boff_id >= len(s['boffs']) or not saved_boff['spec']:
            continue
        spec = s['boff_specs'][boff_id] if boff_id < len(s['boff_specs']) else []
        if not ((spec[0] if spec else '') in (saved_boff['spec'][0], 'Universal') and
                (spec[1] if len(spec) > 1 else '') == (saved_boff['spec'][1] if len(saved_boff['spec']) > 1 else '')):
            continue
        for idx, saved_ability in enumerate(saved_boff['abilities']):
            if idx >= len(s['boffs'][boff_id]):
                break
            if s['boffs'][boff_id][idx] in ('', None) and saved_ability not in ('', None):
                s['boffs'][boff_id][idx] = saved_ability
                self.widgets.build['space']['boffs'][boff_id][idx].set_item(
                    image(self, saved_ability['item']))
    log.info('_restore_session_slots: done')

'''


def patch_callbacks_session_slots(content: str) -> PatchResult:
    if '_EQUIPMENT_CATS' in content:
        return PatchResult(True, 'already applied', content)
    lines = content.splitlines(keepends=True)
    # Insert before first top-level def
    insert_at = next((i for i, l in enumerate(lines) if l.startswith('def ')), None)
    if insert_at is None:
        return PatchResult(False, 'could not find top-level def as insertion anchor')
    lines.insert(insert_at, _CALLBACKS_ADDITIONS)
    return PatchResult(True,
        'added _EQUIPMENT_CATS, _TRAIT_CATS, _save_session_slots, _restore_session_slots',
        ''.join(lines))


# ── src/buildupdater.py ───────────────────────────────────────────────────────

def patch_buildupdater_holoship(content: str) -> PatchResult:
    if 'Federation Intel Holoship' in content:
        return PatchResult(True, 'already applied', content)
    old = (
        "        if 'Innovation Effects' in ship_data['abilities']:\n"
        "            uni_consoles += 1\n"
        "        if '-X2' in self.build['space']['tier']:"
    )
    new = (
        "        if 'Innovation Effects' in ship_data['abilities']:\n"
        "            uni_consoles += 1\n"
        "        elif ship_data['name'] == 'Federation Intel Holoship':\n"
        "            uni_consoles += 1\n"
        "        if '-X2' in self.build['space']['tier']:"
    )
    if old not in content:
        return PatchResult(False,
            'Innovation Effects anchor not found — upstream may have restructured get_console_counts')
    return PatchResult(True, 'added Intel Holoship extra uni_consoles slot',
                       content.replace(old, new, 1))


def patch_buildupdater_normalization(content: str) -> PatchResult:
    if "item.setdefault('mark', '')" in content:
        return PatchResult(True, 'already applied', content)
    old = (
        "            slot_equipment_item(self, item, environment, build_key, subkey)\n"
        "        else:\n"
        "            self.widgets.build[environment][build_key][subkey].clear()"
    )
    new = (
        "            if isinstance(item, dict):\n"
        "                item.setdefault('mark', '')\n"
        "                item.setdefault('modifiers', [None, None, None, None])\n"
        "            slot_equipment_item(self, item, environment, build_key, subkey)\n"
        "        else:\n"
        "            self.widgets.build[environment][build_key][subkey].clear()"
    )
    if old not in content:
        return PatchResult(False,
            'load_equipment_cat anchor not found — upstream may have restructured it')
    return PatchResult(True, 'added mark/modifiers setdefault in load_equipment_cat',
                       content.replace(old, new, 1))


def patch_buildupdater_boff_aliases(content: str) -> PatchResult:
    if "self.cache.item_aliases['boff_abilities']" in content:
        return PatchResult(True, 'already applied', content)
    old = (
        "                    tooltip = self.cache.boff_abilities['all'][ability['item']]\n"
        "                    slot.set_item_full(image(self, ability['item']), None, tooltip)"
    )
    new = (
        "                    tooltip = self.cache.boff_abilities['all'].get(ability['item'], None)\n"
        "                    if tooltip is None:\n"
        "                        new_name = self.cache.item_aliases['boff_abilities'].get(\n"
        "                            ability['item'], None)\n"
        "                        if new_name is None:\n"
        "                            slot.clear()\n"
        "                        else:\n"
        "                            tooltip = self.cache.boff_abilities['all'].get(new_name, None)\n"
        "                            if tooltip is None:\n"
        "                                slot.clear()\n"
        "                            else:\n"
        "                                slot.set_item_full(image(self, new_name), None, tooltip)\n"
        "                    else:\n"
        "                        slot.set_item_full(image(self, ability['item']), None, tooltip)"
    )
    count = content.count(old)
    if count != 2:
        return PatchResult(False,
            f'expected 2 occurrences of direct boff_abilities lookup, found {count} '
            '— upstream may have changed load_boffs')
    return PatchResult(True,
        'replaced direct boff_abilities dict lookup with alias-safe .get() version (space + ground)',
        content.replace(old, new))


# ── src/datafunctions.py ─────────────────────────────────────────────────────

def patch_datafunctions_trait_migration(content: str) -> PatchResult:
    if '_key_map' in content:
        return PatchResult(True, 'already applied', content)
    anchor = (
        "    self.cache.traits = get_cached_cargo_data(self, 'traits.json')\n"
        "    if len(self.cache.traits) == 0:\n"
        "        return False\n"
    )
    migration = (
        "    # Migrate old key format (traits/rep_traits/active_rep_traits → personal/rep/active_rep)\n"
        "    _key_map = {'traits': 'personal', 'rep_traits': 'rep', 'active_rep_traits': 'active_rep'}\n"
        "    for env in ('space', 'ground'):\n"
        "        if env in self.cache.traits and 'traits' in self.cache.traits[env]:\n"
        "            self.cache.traits[env] = {\n"
        "                _key_map.get(k, k): v for k, v in self.cache.traits[env].items()\n"
        "            }\n"
    )
    if anchor not in content:
        return PatchResult(False,
            'traits.json load anchor not found — upstream may have restructured load_cargo_cache')
    return PatchResult(True, 'added trait key migration for old-format cache files',
                       content.replace(anchor, anchor + migration, 1))


# ── Patch registry ────────────────────────────────────────────────────────────

PATCHES: list[Patch] = [
    Patch('src/constants.py',     'Add SEVEN_DAYS_IN_SECONDS + GITHUB_CACHE_URL',         'LOW',    patch_constants_new_consts),
    Patch('src/constants.py',     'Update TRAIT_QUERY_URL (icon_name, -chartype/-isunique)','LOW',   patch_constants_trait_query),
    Patch('src/constants.py',     'Expand Federation + Klingon SPECIES sets',               'LOW',   patch_constants_species),
    Patch('src/callbacks.py',     'Add setsdebug log import',                               'LOW',   patch_callbacks_log_import),
    Patch('src/callbacks.py',     'Add _EQUIPMENT_CATS/_TRAIT_CATS + session_slots fns',   'LOW',   patch_callbacks_session_slots),
    Patch('src/buildupdater.py',  'Intel Holoship extra Universal Console slot',            'LOW',   patch_buildupdater_holoship),
    Patch('src/buildupdater.py',  'Item normalization (mark/modifiers setdefault)',          'LOW',   patch_buildupdater_normalization),
    Patch('src/buildupdater.py',  'Boff ability alias-safe lookup (space + ground)',         'MEDIUM',patch_buildupdater_boff_aliases),
    Patch('src/datafunctions.py', 'Trait key migration for old cache files',                'LOW',   patch_datafunctions_trait_migration),
]


# ── Main logic ────────────────────────────────────────────────────────────────

def recon() -> tuple[list[str], list[str]]:
    """Return (new_commits, changed_files)."""
    print('Fetching upstream/main...')
    git('fetch', 'upstream')
    commits = new_commits_upstream()
    files   = changed_files_upstream()
    return commits, files


def print_recon(commits: list[str], files: list[str]) -> None:
    print(f'\n{"─" * 60}')
    print(f'New upstream commits: {len(commits)}')
    if commits:
        for c in commits[:10]:
            print(f'  {c}')
        if len(commits) > 10:
            print(f'  ... and {len(commits) - 10} more')

    print(f'\nChanged files: {len(files)}')
    for f in files:
        if f in OUR_ONLY_FILES:
            tag = '[our-only — skip]'
        elif f.startswith('warp/'):
            tag = '[our territory — skip]'
        elif f in MANUAL_FILES:
            risk = MANUAL_FILES[f].split(' — ')[0]
            tag = f'[MANUAL {risk}]'
        elif any(p.file == f for p in PATCHES):
            tag = '[AUTO-PATCH]'
        else:
            tag = '[auto-merge]'
        print(f'  {f:45s} {tag}')
    print()


def run_patches(files: list[str], dry_run: bool) -> dict[str, list[tuple[Patch, PatchResult]]]:
    """Apply all patches for files that changed. Returns results per file."""
    results: dict[str, list[tuple[Patch, PatchResult]]] = {}
    changed_set = set(files)

    for patch in PATCHES:
        if patch.file not in changed_set and not dry_run:
            # In apply mode only patch files that actually changed upstream
            # In dry-run always show what would happen
            pass
        try:
            upstream_content = git_file('upstream/main', patch.file)
        except subprocess.CalledProcessError:
            r = PatchResult(False, f'file not found in upstream/main')
            results.setdefault(patch.file, []).append((patch, r))
            continue

        result = patch.apply_fn(upstream_content)
        results.setdefault(patch.file, []).append((patch, result))

    return results


def print_patch_results(results: dict[str, list[tuple[Patch, PatchResult]]]) -> None:
    print('Patch results:')
    print(f'{"─" * 60}')
    all_ok = True
    for file, patch_results in results.items():
        print(f'\n  {file}')
        for patch, result in patch_results:
            icon = '✅' if result.success else '⚠️ '
            print(f'    {icon} [{patch.risk}] {patch.description}')
            if not result.success:
                print(f'         → {result.message}')
                all_ok = False
            elif result.message != 'already applied':
                print(f'         → {result.message}')
    if all_ok:
        print('\nAll auto-patches applied cleanly.')
    else:
        print('\nSome patches need manual attention (see ⚠️  above).')


def print_manual_checklist(files: list[str]) -> None:
    manual_needed = [f for f in files if f in MANUAL_FILES]
    if not manual_needed:
        return
    print(f'\n{"─" * 60}')
    print('Manual review required for the following files:')
    for f in manual_needed:
        print(f'\n  📋 {f}')
        for line in MANUAL_FILES[f].splitlines():
            print(f'     {line}')
    print()


def apply_mode(files: list[str], branch: str) -> None:
    """Create branch, apply patches to upstream versions, commit."""
    print(f'\nCreating branch: {branch}')
    git('checkout', '-b', branch, 'upstream/main')

    patch_results = run_patches(files, dry_run=False)

    # Write patched files
    patched_files: list[str] = []
    for file, file_results in patch_results.items():
        # Get the final content: chain all successful patches
        try:
            content = git_file('upstream/main', file)
        except subprocess.CalledProcessError:
            continue
        changed = False
        for patch, result in file_results:
            if result.success and result.content and result.message != 'already applied':
                content = result.content
                changed = True
        if changed:
            (ROOT / file).write_text(content, encoding='utf-8')
            patched_files.append(file)

    if patched_files:
        git('add', *patched_files)
        git('commit', '-m',
            'fix: re-apply SETS-WARP patches after upstream merge\n\n'
            'Auto-applied by scripts/upstream_sync.py.\n'
            'See docs/src_patches.md for patch details.\n\n'
            'Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>'
        )
        print(f'\nCommitted patches for: {", ".join(patched_files)}')
    else:
        print('\nNo files changed (all patches already applied or no changes needed).')

    print_patch_results(patch_results)
    print_manual_checklist(files)

    print(f'\n{"─" * 60}')
    print(f'Branch "{branch}" ready. Next steps:')
    print('  1. Review ⚠️  patches and manual files listed above')
    print('  2. git add + git commit any manual fixes')
    print(f'  3. git checkout main && git merge {branch}')
    print()


def dry_run_mode(files: list[str]) -> None:
    patch_results = run_patches(files, dry_run=True)
    print_patch_results(patch_results)
    print_manual_checklist(files)
    print('Dry run complete. Run with --apply to create branch and apply patches.')


def main() -> None:
    parser = argparse.ArgumentParser(description='SETS-WARP upstream sync helper')
    parser.add_argument('--apply',  action='store_true', help='Create branch and apply patches')
    parser.add_argument('--branch', default=f'upstream-merge-{date.today().strftime("%Y-%m-%d")}',
                        help='Branch name (default: upstream-merge-YYYY-MM-DD)')
    args = parser.parse_args()

    commits, files = recon()

    if not commits:
        print('Already up to date with upstream/main. Nothing to do.')
        return

    print_recon(commits, files)

    if args.apply:
        apply_mode(files, args.branch)
    else:
        dry_run_mode(files)


if __name__ == '__main__':
    main()
