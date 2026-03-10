# warp/tools/scraper.py
# Scraper for external STO data sources.
# Run this ONCE locally to build the WARP icon+item database.
#
# Sources:
#   1. vger.stobuilds.com  — structured JSON with item names + icon URLs
#      /starship-traits, /space-equipment, /ground-equipment, /personal-traits
#   2. stowiki.net         — wiki pages for item details + additional icons
#   3. SETS icon cache     — already-downloaded icons (no re-download needed)
#
# Output:
#   warp/data/item_db.json          — {item_name: {slot, type, wiki_url, ...}}
#   warp/data/icons/<name>.png      — downloaded icon images (for ML training)
#
# Usage (run from SETS root):
#   python -m warp.tools.scraper [--output warp/data] [--sets-images .config/images]
#
# Respects rate limits: 0.5s delay between requests.

from __future__ import annotations

import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

# ── Source URLs ────────────────────────────────────────────────────────────────

VGER_BASE   = 'https://vger.stobuilds.com'
STOWIKI_API = 'https://stowiki.net/w/api.php'
STOWIKI_IMG = 'https://stowiki.net/wiki/Special:FilePath/{filename}'

VGER_PAGES = {
    'starship_traits': f'{VGER_BASE}/starship-traits',
    'space_equipment': f'{VGER_BASE}/space-equipment',
    'ground_equipment':f'{VGER_BASE}/ground-equipment',
    'personal_traits': f'{VGER_BASE}/personal-traits',
}

# vger JSON API endpoints (returns JSON if Accept: application/json)
VGER_API = {
    'starship_traits': f'{VGER_BASE}/api/starship-traits',
    'space_equipment': f'{VGER_BASE}/api/space-equipment',
    'ground_equipment':f'{VGER_BASE}/api/ground-equipment',
    'personal_traits': f'{VGER_BASE}/api/personal-traits',
}

REQUEST_DELAY = 0.5   # seconds between requests

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build WARP icon+item database from external STO sources')
    parser.add_argument('--output', default='warp/data',
                        help='Output directory (default: warp/data)')
    parser.add_argument('--sets-images', default='.config/images',
                        help='SETS downloaded images directory')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip icon downloads (index only)')
    parser.add_argument('--stowiki-only', action='store_true',
                        help='Only scrape stowiki (skip vger)')
    parser.add_argument('--vger-only', action='store_true',
                        help='Only scrape vger (skip stowiki)')
    args = parser.parse_args()

    output_dir     = Path(args.output)
    icons_dir      = output_dir / 'icons'
    sets_img_dir   = Path(args.sets_images)
    output_dir.mkdir(parents=True, exist_ok=True)
    icons_dir.mkdir(parents=True, exist_ok=True)

    item_db: dict[str, dict] = {}

    # ── Step 1: Copy from SETS image cache ────────────────────────────────
    log.info('Step 1: importing from SETS image cache…')
    sets_count = import_from_sets_cache(sets_img_dir, icons_dir)
    log.info(f'  {sets_count} icons copied from SETS cache')

    # ── Step 2: Scrape vger.stobuilds.com ─────────────────────────────────
    if not args.stowiki_only:
        log.info('Step 2: scraping vger.stobuilds.com…')
        vger_items = scrape_vger(icons_dir, args.skip_download)
        item_db.update(vger_items)
        log.info(f'  {len(vger_items)} items from vger')

    # ── Step 3: Scrape stowiki.net ─────────────────────────────────────────
    if not args.vger_only:
        log.info('Step 3: scraping stowiki.net…')
        wiki_items = scrape_stowiki(item_db.keys(), icons_dir, args.skip_download)
        # Merge: add wiki data to existing entries, add new entries
        for name, data in wiki_items.items():
            if name in item_db:
                item_db[name].update({k: v for k, v in data.items() if v})
            else:
                item_db[name] = data
        log.info(f'  {len(wiki_items)} items from stowiki')

    # ── Step 4: Save ──────────────────────────────────────────────────────
    db_path = output_dir / 'item_db.json'
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(item_db, f, indent=2, ensure_ascii=False)
    log.info(f'Saved {len(item_db)} items → {db_path}')
    log.info(f'Icons directory: {icons_dir}  ({len(list(icons_dir.glob("*.png")))} PNGs)')


# ── SETS cache import ──────────────────────────────────────────────────────────

def import_from_sets_cache(sets_img_dir: Path, icons_dir: Path) -> int:
    """
    Copies existing SETS icons (already downloaded) to warp/data/icons/.
    Filenames are kept as-is (quote_plus encoded).
    Returns count of copied files.
    """
    if not sets_img_dir.exists():
        log.warning(f'  SETS images dir not found: {sets_img_dir}')
        return 0
    count = 0
    for png in sets_img_dir.glob('*.png'):
        dest = icons_dir / png.name
        if not dest.exists():
            import shutil
            shutil.copy2(png, dest)
            count += 1
    return count


# ── vger scraper ───────────────────────────────────────────────────────────────

def scrape_vger(icons_dir: Path, skip_download: bool) -> dict[str, dict]:
    """
    Scrapes vger.stobuilds.com for item listings.
    Tries JSON API first; falls back to HTML parsing.
    """
    import requests
    session = requests.Session()
    session.headers.update({
        'Accept': 'application/json, text/html',
        'User-Agent': 'SETS-WARP/0.1 (github.com/STOCD/SETS)'
    })

    items: dict[str, dict] = {}

    # Map vger slot name → SETS build_key
    SLOT_CATEGORY = {
        'starship_traits': 'starship_traits',
        'space_equipment': 'space',
        'ground_equipment':'ground',
        'personal_traits': 'traits',
    }

    for category, url in VGER_PAGES.items():
        log.info(f'  vger: {category}')
        try:
            # Try JSON endpoint
            api_url = VGER_API.get(category, url)
            resp = session.get(api_url, timeout=15)
            if resp.headers.get('content-type', '').startswith('application/json'):
                data = resp.json()
                parsed = _parse_vger_json(data, category, SLOT_CATEGORY[category])
            else:
                # Fall back to HTML scraping
                parsed = _parse_vger_html(resp.text, category, SLOT_CATEGORY[category], url)

            # Download icons
            if not skip_download:
                for name, entry in parsed.items():
                    icon_url = entry.get('icon_url')
                    if icon_url:
                        _download_icon(session, icon_url, name, icons_dir)
                        time.sleep(REQUEST_DELAY)

            items.update(parsed)
        except Exception as e:
            log.warning(f'  vger {category} failed: {e}')
        time.sleep(REQUEST_DELAY)

    return items


def _parse_vger_json(data, category: str, slot_category: str) -> dict[str, dict]:
    """Parse vger JSON API response."""
    items = {}
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get('items') or data.get('data') or list(data.values())
    else:
        return items

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get('name') or entry.get('Name') or entry.get('title')
        if not name:
            continue
        items[name] = {
            'name':       name,
            'category':   slot_category,
            'wiki_url':   entry.get('url') or entry.get('wiki') or '',
            'icon_url':   entry.get('icon') or entry.get('image') or '',
            'rarity':     entry.get('rarity') or '',
            'type':       entry.get('type') or entry.get('slot') or '',
            'source':     f'vger:{category}',
        }
    return items


def _parse_vger_html(html: str, category: str, slot_category: str, base_url: str) -> dict[str, dict]:
    """Parse vger HTML page to extract item names and icon URLs."""
    items = {}
    try:
        from lxml import html as lhtml
        tree  = lhtml.fromstring(html)
        # vger pages have item cards — adapt selectors to actual structure
        # Common patterns: <div class="item-name">, <img class="item-icon">
        for card in tree.cssselect('.item-card, .trait-card, .equipment-card, [data-item-name]'):
            name_el = card.cssselect('.item-name, .name, h3, h4')
            name    = name_el[0].text_content().strip() if name_el else ''
            if not name:
                name = card.get('data-item-name', '')
            if not name:
                continue
            img_el   = card.cssselect('img')
            icon_url = ''
            if img_el:
                src = img_el[0].get('src') or img_el[0].get('data-src') or ''
                if src:
                    icon_url = urljoin(base_url, src)
            items[name] = {
                'name':     name,
                'category': slot_category,
                'icon_url': icon_url,
                'source':   f'vger_html:{category}',
            }
    except Exception as e:
        log.debug(f'vger HTML parse error: {e}')
    return items


# ── stowiki scraper ────────────────────────────────────────────────────────────

def scrape_stowiki(
    known_names: 'Iterable[str]',
    icons_dir: Path,
    skip_download: bool,
) -> dict[str, dict]:
    """
    Queries stowiki.net MediaWiki API for item pages and icon images.
    Only queries items already known by name (from SETS cache + vger).
    """
    import requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'SETS-WARP/0.1 (github.com/STOCD/SETS)'
    })

    items: dict[str, dict] = {}
    names = list(known_names)

    # Batch queries of 50 items each
    batch_size = 50
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        try:
            wiki_data = _query_stowiki_batch(session, batch)
            for name, data in wiki_data.items():
                items[name] = data
                if not skip_download and data.get('icon_url'):
                    _download_icon(session, data['icon_url'], name, icons_dir)
                    time.sleep(REQUEST_DELAY * 0.5)
        except Exception as e:
            log.warning(f'  stowiki batch {i//batch_size}: {e}')
        time.sleep(REQUEST_DELAY)

    # Also scrape Equipment categories for items not in SETS yet
    new_from_wiki = scrape_stowiki_categories(session, icons_dir, skip_download)
    items.update(new_from_wiki)

    return items


def _query_stowiki_batch(session, names: list[str]) -> dict[str, dict]:
    """Query MediaWiki API for page info + images for a batch of item names."""
    titles = '|'.join(names)
    params = {
        'action': 'query',
        'format': 'json',
        'titles': titles,
        'prop':   'info|images|categories',
        'inprop': 'url',
        'imlimit': 5,
    }
    resp = session.get(STOWIKI_API, params=params, timeout=15)
    data = resp.json()

    items: dict[str, dict] = {}
    for page_id, page in data.get('query', {}).get('pages', {}).items():
        if page_id == '-1':
            continue
        name     = page.get('title', '')
        wiki_url = page.get('fullurl', '')

        # Find icon — usually the first image matching the item name
        images = page.get('images', [])
        icon_url = ''
        for img in images:
            img_name = img.get('title', '')
            if name.lower().replace(' ', '_') in img_name.lower():
                icon_url = STOWIKI_IMG.format(filename=quote_plus(img_name[5:]))
                break

        items[name] = {
            'name':     name,
            'wiki_url': wiki_url,
            'icon_url': icon_url,
            'source':   'stowiki_api',
        }
    return items


def scrape_stowiki_categories(
    session, icons_dir: Path, skip_download: bool
) -> dict[str, dict]:
    """
    Scrapes stowiki category pages for items not already in SETS.
    Categories: Category:Space_equipment, Category:Ground_equipment, etc.
    """
    CATEGORIES = [
        'Category:Space_equipment',
        'Category:Ground_equipment',
        'Category:Personal_traits',
        'Category:Starship_traits',
        'Category:Reputation_traits',
    ]

    items: dict[str, dict] = {}
    for cat in CATEGORIES:
        params = {
            'action':  'query',
            'format':  'json',
            'list':    'categorymembers',
            'cmtitle': cat,
            'cmlimit': 500,
        }
        try:
            resp = session.get(STOWIKI_API, params=params, timeout=15)
            members = resp.json().get('query', {}).get('categorymembers', [])
            for m in members:
                name = m.get('title', '')
                if name and name not in items:
                    items[name] = {
                        'name':     name,
                        'wiki_url': f'https://stowiki.net/wiki/{quote_plus(name)}',
                        'icon_url': '',
                        'source':   f'stowiki_cat:{cat}',
                    }
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log.warning(f'  stowiki category {cat}: {e}')

    return items


# ── Icon downloader ────────────────────────────────────────────────────────────

def _download_icon(session, url: str, item_name: str, icons_dir: Path) -> bool:
    """Download a single icon and save as <quote_plus(item_name)>.png"""
    filename = quote_plus(item_name) + '.png'
    dest = icons_dir / filename
    if dest.exists() and dest.stat().st_size > 200:
        return True   # already downloaded

    try:
        resp = session.get(url, timeout=10, stream=True)
        if resp.status_code != 200:
            return False
        content = resp.content
        if len(content) < 100:
            return False
        dest.write_bytes(content)
        return True
    except Exception as e:
        log.debug(f'Icon download failed ({item_name}): {e}')
        return False


if __name__ == '__main__':
    main()
