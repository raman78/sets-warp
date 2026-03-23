# warp/tools/scraper.py  — v5  (final, structures confirmed)
#
# Confirmed SETS cargo structures:
#   equipment.json       list[{Page, name, rarity, type, ...}]         3987 items
#   traits.json          list[{Page, name, type, environment, ...}]     540 items
#   starship_traits.json list[{Page, name, short, type, obtained, ...}] 347 items
#   doffs.json           list[{spec, _pageName, shipdutytype, ...}]     273 items
#   ship_list.json       list[{Page, name, image, tier, type, ...}]     783 ships
#   boff_abilities.json  dict{space/ground/all: {profession: [rank_dicts]}}
#   modifiers.json       list[{modifier, type, stats, available}]       (not items)
#
# name fields contain HTML entities (&#34; = ", &quot; = ") — decoded on read.
# Page fields also contain entities — decoded for wiki URLs.
#
# Run from SETS root:
#   python -m warp.tools.scraper
#   python -m warp.tools.scraper --skip-icons --skip-vger

from __future__ import annotations
import json, re, time, argparse, logging, sys, shutil
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)-7s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

VGER_BASE  = 'https://vger.stobuilds.com'
VGER_PAGES = {
    'starship_traits': (f'{VGER_BASE}/starship-traits', 'starship_traits'),
    'personal_traits': (f'{VGER_BASE}/personal-traits', 'traits'),
    'space_equipment': (f'{VGER_BASE}/space-equipment', 'space'),
    'ground_equipment':(f'{VGER_BASE}/ground-equipment','ground'),
}

# equipment.json 'type' → WARP/SETS build_key
EQUIPMENT_TYPES = {
    'Body Armor':               'armor',
    'EV Suit':                  'ev_suit',
    'Experimental Weapon':      'experimental',
    'Ground Device':            'ground_devices',
    'Ground Weapon':            'weapons',
    'Hangar Bay':               'hangars',
    'Impulse Engine':           'engines',
    'Kit':                      'kit',
    'Kit Module':               'kit_modules',
    'Personal Shield':          'personal_shield',
    'Ship Aft Weapon':          'aft_weapons',
    'Ship Deflector Dish':      'deflector',
    'Ship Device':              'devices',
    'Ship Engineering Console': 'eng_consoles',
    'Ship Fore Weapon':         'fore_weapons',
    'Ship Science Console':     'sci_consoles',
    'Ship Secondary Deflector': 'sec_def',
    'Ship Shields':             'shield',
    'Ship Tactical Console':    'tac_consoles',
    'Ship Weapon':              'fore_weapons',
    'Singularity Engine':       'core',
    'Universal Console':        'uni_consoles',
    'Warp Engine':              'core',
}

# traits.json 'type' values → WARP category
TRAIT_TYPE_MAP = {
    'char':   'traits',
    'spec':   'traits',
    'boff':   'traits',
    'rep':    'rep_traits',
    'active': 'active_rep_traits',
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
}
REQ_DELAY  = 0.3
ICON_DELAY = 0.15
_NAME_RE   = re.compile(r'^[A-Z\"\'][A-Za-z0-9 \'\"\-\/\(\)\[\]:,\.&!]{1,89}$')


def main():
    ap = argparse.ArgumentParser(description='Build WARP item DB from SETS cargo + vger')
    ap.add_argument('--output',      default='warp/data')
    ap.add_argument('--cargo',       default='.config/cargo')
    ap.add_argument('--sets-images', default='.config/images')
    ap.add_argument('--skip-icons',  action='store_true')
    ap.add_argument('--skip-vger',   action='store_true')
    ap.add_argument('--skip-github', action='store_true')
    args = ap.parse_args()

    out_dir  = Path(args.output);    out_dir.mkdir(parents=True, exist_ok=True)
    icon_dir = out_dir / 'icons';    icon_dir.mkdir(exist_ok=True)
    cargo_dir = Path(args.cargo)
    img_dir   = Path(args.sets_images)

    import requests
    s = requests.Session(); s.headers.update(HEADERS)

    item_db: dict[str, dict] = {}
    icon_q:  dict[str, str]  = {}

    # 1 ── SETS images
    log.info('Step 1: SETS image cache')
    n = _copy_images(img_dir, icon_dir)
    log.info(f'  {n} icons copied')

    # 2 ── SETS cargo  (PRIMARY — 5000+ items already on disk)
    log.info('Step 2: SETS cargo cache')
    _read_cargo(cargo_dir, item_db)
    log.info(f'  {len(item_db)} items loaded')

    # 3 ── vger JS chunks  (icon URLs + any extra items)
    if not args.skip_vger:
        log.info('Step 3: vger JS chunks')
        vi, vc = _scrape_vger_js(s)
        new = 0
        for name, data in vi.items():
            if name not in item_db:
                item_db[name] = data; new += 1
            else:
                if data.get('icon_url') and not item_db[name].get('icon_url'):
                    item_db[name]['icon_url'] = data['icon_url']
                if data.get('wiki_url') and not item_db[name].get('wiki_url'):
                    item_db[name]['wiki_url'] = data['wiki_url']
        icon_q.update(vc)
        log.info(f'  {new} new, {len(vc)} icon URLs from vger')
    else:
        log.info('Step 3: vger skipped')

    # 4 ── GitHub fallback (only if cargo was empty)
    if not args.skip_github and len(item_db) < 100:
        log.info('Step 4: STOCD GitHub (fallback)')
        gi, gc = _scrape_github(s)
        for name, data in gi.items():
            if name not in item_db: item_db[name] = data
        icon_q.update({k: v for k, v in gc.items() if k not in icon_q})
        log.info(f'  {len(gi)} items from GitHub')
    else:
        log.info(f'Step 4: GitHub skipped ({len(item_db)} items already)')

    # 5 ── icon downloads
    for name, data in item_db.items():
        if data.get('icon_url') and not (icon_dir/(quote_plus(name)+'.png')).exists():
            icon_q[name] = data['icon_url']

    if not args.skip_icons and icon_q:
        log.info(f'Step 5: downloading {len(icon_q)} missing icons')
        ok = _download_icons(s, icon_q, icon_dir)
        log.info(f'  {ok}/{len(icon_q)} downloaded')
    else:
        log.info(f'Step 5: icons skipped ({len(icon_q)} queued)')

    # 6 ── save
    db = out_dir / 'item_db.json'
    db.write_text(json.dumps(item_db, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    log.info(f'Saved {len(item_db)} items → {db}')
    log.info(f'Icons: {len(list(icon_dir.glob("*.png")))} PNGs in {icon_dir}')
    _print_summary(item_db)


# ── Cargo readers ──────────────────────────────────────────────────────────────

def _h(s: str | None) -> str:
    """Decode HTML entities from cargo strings."""
    return unescape(str(s)) if s else ''


def _wiki_url(page: str) -> str:
    return f'https://stowiki.net/wiki/{quote_plus(_h(page).replace(" ", "_"))}'


def _read_cargo(cargo_dir: Path, out: dict):
    if not cargo_dir.exists():
        log.warning(f'  cargo not found: {cargo_dir}  (run SETS once first)')
        return
    files = sorted(cargo_dir.glob('*.json'))
    log.info(f'  files: {[f.name for f in files]}')

    # ── equipment.json ─────────────────────────────────────────────────────────
    # list[{Page, name, rarity, type, ...}]
    p = cargo_dir / 'equipment.json'
    if p.exists():
        rows = json.loads(p.read_text(encoding='utf-8'))
        n = 0
        for row in rows:
            name = _h(row.get('name'))
            if not name: continue
            eq_type = _h(row.get('type', ''))
            build_key = EQUIPMENT_TYPES.get(eq_type, 'space')
            # Determine space vs ground
            env = 'ground' if build_key in (
                'armor','ev_suit','ground_devices','weapons','kit',
                'kit_modules','personal_shield') else 'space'
            page = _h(row.get('Page') or name)
            out[name] = {
                'name':     name,
                'category': build_key,
                'env':      env,
                'rarity':   _h(row.get('rarity', '')),
                'type':     eq_type,
                'wiki_url': _wiki_url(page),
                'icon_url': _icon_url_from_cargo(row),
                'source':   'sets_cargo:equipment',
            }
            n += 1
        log.info(f'    equipment.json:       {n:4d} items')

    # ── traits.json ────────────────────────────────────────────────────────────
    # list[{Page, name, type, environment, description, icon_name}]
    # type: 'char','spec','boff','rep','active'
    # environment: 'space','ground','both'
    p = cargo_dir / 'traits.json'
    if p.exists():
        rows = json.loads(p.read_text(encoding='utf-8'))
        n = 0
        for row in rows:
            name = _h(row.get('name'))
            if not name: continue
            ttype = (row.get('type') or '').lower()
            env   = (row.get('environment') or '').lower()
            cat   = TRAIT_TYPE_MAP.get(ttype, 'traits')
            page  = _h(row.get('Page') or name)
            icon_name = row.get('icon_name') or ''
            icon_url  = _icon_url_from_name(icon_name) if icon_name else ''
            out[name] = {
                'name':     name,
                'category': cat,
                'env':      env,
                'rarity':   '',
                'type':     ttype,
                'wiki_url': _wiki_url(page),
                'icon_url': icon_url,
                'source':   f'sets_cargo:traits:{ttype}:{env}',
            }
            n += 1
        log.info(f'    traits.json:          {n:4d} traits')

    # ── starship_traits.json ───────────────────────────────────────────────────
    # list[{Page, name, short, type, detailed, obtained, basic, icon_name}]
    p = cargo_dir / 'starship_traits.json'
    if p.exists():
        rows = json.loads(p.read_text(encoding='utf-8'))
        n = 0
        for row in rows:
            name = _h(row.get('name'))
            if not name: continue
            page = _h(row.get('Page') or name)
            icon_name = row.get('icon_name') or ''
            icon_url  = _icon_url_from_name(icon_name) if icon_name else ''
            if name not in out:  # don't override equipment with same name
                out[name] = {
                    'name':     name,
                    'category': 'starship_traits',
                    'env':      'space',
                    'rarity':   '',
                    'type':     _h(row.get('type', '')),
                    'wiki_url': _wiki_url(page),
                    'icon_url': icon_url,
                    'source':   'sets_cargo:starship_traits',
                }
                n += 1
        log.info(f'    starship_traits.json: {n:4d} starship traits')

    # ── doffs.json ─────────────────────────────────────────────────────────────
    # list[{spec, _pageName, shipdutytype, department, description, white..gold}]
    # 'spec' is the doff specialization name
    p = cargo_dir / 'doffs.json'
    if p.exists():
        rows = json.loads(p.read_text(encoding='utf-8'))
        seen: set[str] = set()
        n = 0
        for row in rows:
            spec = _h(row.get('spec') or row.get('_pageName') or '')
            if not spec or spec in seen: continue
            seen.add(spec)
            env = (row.get('shipdutytype') or 'space').lower()
            out[spec] = {
                'name':     spec,
                'category': 'doffs',
                'env':      env,
                'rarity':   '',
                'type':     _h(row.get('department', '')),
                'wiki_url': _wiki_url(row.get('_pageName') or spec),
                'icon_url': '',
                'source':   'sets_cargo:doffs',
            }
            n += 1
        log.info(f'    doffs.json:           {n:4d} doff specs')

    # ── boff_abilities.json ────────────────────────────────────────────────────
    # dict{space/ground/all: {profession: [rank_dict, ...]}}
    # rank_dict: {ability_name: description}
    p = cargo_dir / 'boff_abilities.json'
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        n = 0
        seen: set[str] = set()
        # Try all env keys: space, ground, all — whichever has data
        for env_key, env_data in data.items():
            if not isinstance(env_data, dict): continue
            for profession, rank_list in env_data.items():
                if not isinstance(rank_list, list): continue
                for rank_item in rank_list:
                    # rank_item is a dict of {ability_name: description}
                    if not isinstance(rank_item, dict): continue
                    for ability_name in rank_item.keys():
                        name = _h(ability_name)
                        if not name or name in seen: continue
                        seen.add(name)
                        env = env_key if env_key in ('space', 'ground') else 'space'
                        out[name] = {
                            'name':     name,
                            'category': 'boff_abilities',
                            'env':      env,
                            'rarity':   '',
                            'type':     profession,
                            'wiki_url': '',
                            'icon_url': '',
                            'source':   'sets_cargo:boff',
                        }
                        n += 1
        log.info(f'    boff_abilities.json:  {n:4d} abilities')

    # ── ship_list.json ─────────────────────────────────────────────────────────
    # list[{Page, name, image, tier, type, ...}]
    p = cargo_dir / 'ship_list.json'
    if p.exists():
        rows = json.loads(p.read_text(encoding='utf-8'))
        n = 0
        for row in rows:
            name = _h(row.get('name'))
            if not name or name in out: continue
            page = _h(row.get('Page') or name)
            # image: "File:Fed Ship Achilles.png" → construct wiki URL
            img  = row.get('image') or ''
            icon_url = ''
            if img:
                img_name = img.replace('File:', '').strip()
                icon_url = f'https://stowiki.net/wiki/Special:FilePath/{quote_plus(img_name)}'
            out[name] = {
                'name':     name,
                'category': 'ships',
                'env':      'space',
                'rarity':   '',
                'type':     str(row.get('type', '')),
                'tier':     str(row.get('tier', '')),
                'wiki_url': _wiki_url(page),
                'icon_url': icon_url,
                'source':   'sets_cargo:ships',
            }
            n += 1
        log.info(f'    ship_list.json:       {n:4d} ships')


def _icon_url_from_cargo(row: dict) -> str:
    """Try to get icon URL from cargo row (icon_name field or image field)."""
    icon_name = row.get('icon_name') or row.get('icon') or ''
    if icon_name:
        return _icon_url_from_name(str(icon_name))
    return ''


def _icon_url_from_name(icon_name: str) -> str:
    """
    stowiki icon naming: 'icon_name' in cargo → File:<icon_name>.png
    e.g. "Iconequip_dualbeambank" → Special:FilePath/Iconequip_dualbeambank.png
    """
    if not icon_name: return ''
    clean = icon_name.strip()
    if not clean.lower().endswith('.png'):
        clean += '.png'
    return f'https://stowiki.net/wiki/Special:FilePath/{quote_plus(clean)}'


# ── vger JS scraper ────────────────────────────────────────────────────────────

def _scrape_vger_js(s) -> tuple[dict, dict]:
    items: dict[str, dict] = {}
    icons: dict[str, str]  = {}
    visited: set[str] = set()

    for page_key, (url, cat) in VGER_PAGES.items():
        log.info(f'  vger/{page_key}')
        try:
            r = s.get(url, timeout=20); r.raise_for_status()
            chunk_urls = _get_js_chunks(r.text, r.headers.get('link',''))
            log.info(f'    {len(chunk_urls)} chunks')
            for cu in chunk_urls:
                if cu in visited: continue
                visited.add(cu)
                try:
                    cr = s.get(cu, timeout=15)
                    if cr.status_code != 200: continue
                    ni, nc = _extract_from_js(cr.text, cat)
                    if ni:
                        items.update(ni); icons.update(nc)
                        log.info(f'    {cu.split("/")[-1]}: {len(ni)} items')
                except Exception as e:
                    log.debug(f'    chunk {cu}: {e}')
                time.sleep(0.1)
        except Exception as e:
            log.warning(f'  vger/{page_key}: {e}')
        time.sleep(REQ_DELAY)
    return items, icons


def _get_js_chunks(html: str, link_header: str) -> list[str]:
    from urllib.parse import urljoin
    seen: set[str] = set(); urls = []
    for m in re.finditer(r'<([^>]+\.js)>', link_header):
        u = urljoin(VGER_BASE, m.group(1).lstrip('.'))
        if u not in seen: seen.add(u); urls.append(u)
    for m in re.finditer(r'/_app/immutable/(?:nodes|chunks)/[A-Za-z0-9._-]+\.js', html):
        u = VGER_BASE + m.group(0)
        if u not in seen: seen.add(u); urls.append(u)
    urls.sort(key=lambda u: (0 if '/nodes/' in u else 1))
    return urls


def _extract_from_js(js: str, cat: str) -> tuple[dict, dict]:
    items: dict[str, dict] = {}
    icons: dict[str, str]  = {}
    _ICON_RE = re.compile(r'https?://\S+\.(png|jpg|webp)', re.I)
    _WIKI_RE = re.compile(r'https?://stowiki\.net/wiki/\S+')

    # JSON.parse blobs
    for m in re.finditer(r'JSON\.parse\("((?:[^"\\]|\\.)*)"\)', js):
        try:
            raw = json.loads(m.group(1).replace('\\"','"').replace('\\\\','\\'))
            ni, nc = _walk_json(raw, cat); items.update(ni); icons.update(nc)
        except Exception: pass

    # {name:"...", icon:"..."} JS object literals
    for m in re.finditer(
            r'\{[^{}]{0,600}name:"([A-Z\"\'][A-Za-z0-9 \'\"\-\/\(\)\[\]:,\.&!]{2,89})"[^{}]{0,600}\}',
            js):
        block, name = m.group(0), m.group(1)
        im = re.search(r'icon:"(https?://[^"]+)"', block)
        wm = re.search(r'(?:url|wiki|href):"(https?://[^"]+)"', block)
        iu = im.group(1) if im else ''
        wu = wm.group(1) if wm else ''
        if name not in items:
            items[name] = {'name':name,'category':cat,'icon_url':iu,'wiki_url':wu,
                           'source':'vger_js_obj'}
            if iu: icons[name] = iu

    # Flat string arrays
    for m in re.finditer(r'\[("(?:[^"\\]|\\.)*"(?:,"(?:[^"\\]|\\.)*"){4,})\]', js):
        parts = [p.replace('\\"','"') for p in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))]
        for i, p in enumerate(parts):
            if _NAME_RE.match(p) and len(p) >= 6 and p not in items:
                win = parts[max(0,i-4):i+8]
                iu  = next((u for u in win if _ICON_RE.match(u)), '')
                wu  = next((u for u in win if _WIKI_RE.match(u)), '')
                items[p] = {'name':p,'category':cat,'icon_url':iu,'wiki_url':wu,
                            'source':'vger_js_arr'}
                if iu: icons[p] = iu

    # name,icon pairs: "Item Name","https://...icon.png"
    for m in re.finditer(
            r'"([A-Z\"\'][A-Za-z0-9 \'\"\-\/\(\)\[\]:,\.&!]{5,89})"\s*,\s*"(https?://[^"]+\.(png|jpg|webp))"',
            js, re.I):
        name, iu = m.group(1), m.group(2)
        if name not in items:
            items[name] = {'name':name,'category':cat,'icon_url':iu,'wiki_url':'',
                           'source':'vger_js_pair'}
            icons[name] = iu

    return items, icons


def _walk_json(obj, cat: str, _d: int = 0) -> tuple[dict, dict]:
    items: dict[str, dict] = {}; icons: dict[str, str] = {}
    if _d > 12: return items, icons
    if isinstance(obj, dict):
        name = _h(obj.get('name') or obj.get('Name') or obj.get('title') or '')
        if 3 <= len(name) <= 90 and _NAME_RE.match(name):
            iu = next((v for k,v in obj.items()
                       if k.lower() in ('icon','image','iconurl','icon_url')
                       and isinstance(v,str) and v.startswith('http')), '')
            wu = obj.get('url') or obj.get('wiki') or obj.get('Page') or ''
            if wu and not str(wu).startswith('http'): wu = _wiki_url(str(wu))
            items[name] = {
                'name':name,'category':cat,
                'rarity': _h(obj.get('rarity') or obj.get('Rarity') or ''),
                'type':   _h(obj.get('type') or obj.get('Type') or ''),
                'icon_url':iu, 'wiki_url':str(wu), 'source':'json_walk',
            }
            if iu: icons[name] = iu
        for v in obj.values():
            i2,ic2 = _walk_json(v,cat,_d+1); items.update(i2); icons.update(ic2)
    elif isinstance(obj, list):
        for v in obj:
            i2,ic2 = _walk_json(v,cat,_d+1); items.update(i2); icons.update(ic2)
    return items, icons


# ── GitHub fallback ────────────────────────────────────────────────────────────

def _scrape_github(s) -> tuple[dict, dict]:
    items: dict[str, dict] = {}; icons: dict[str, str] = {}
    for fname, cat in [('equipment.json','space'),('traits.json','traits'),
                       ('starship_traits.json','starship_traits')]:
        url = f'https://raw.githubusercontent.com/STOCD/SETS/main/local/{fname}'
        try:
            r = s.get(url, timeout=15)
            if r.status_code != 200: continue
            ni, nc = _walk_json(r.json(), cat)
            if ni: log.info(f'  github {fname}: {len(ni)} items')
            items.update(ni); icons.update(nc)
        except Exception as e:
            log.debug(f'  github {fname}: {e}')
        time.sleep(REQ_DELAY)
    return items, icons


# ── Helpers ────────────────────────────────────────────────────────────────────

def _copy_images(src: Path, dst: Path) -> int:
    if not src.exists(): log.warning(f'  images not found: {src}'); return 0
    n = 0
    for p in src.glob('*.png'):
        d = dst / p.name
        if not d.exists(): shutil.copy2(p, d); n += 1
    return n


def _download_icons(s, queue: dict[str, str], dst: Path) -> int:
    ok = 0
    for i, (name, url) in enumerate(queue.items()):
        if i % 200 == 0 and i: log.info(f'  {i}/{len(queue)} ({ok} ok)')
        dest = dst / (quote_plus(name) + '.png')
        if dest.exists() and dest.stat().st_size > 200: ok += 1; continue
        try:
            r = s.get(url, timeout=12)
            if r.status_code == 200 and len(r.content) > 200:
                dest.write_bytes(r.content); ok += 1
        except Exception: pass
        time.sleep(ICON_DELAY)
    return ok


def _print_summary(db: dict):
    from collections import Counter
    cats = Counter(v.get('category','?') for v in db.values())
    log.info('Category breakdown:')
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        log.info(f'  {n:5d}  {cat}')
    srcs = Counter(v.get('source','').split(':')[0] for v in db.values())
    log.info('Sources:')
    for src, n in sorted(srcs.items(), key=lambda x: -x[1]):
        log.info(f'  {n:5d}  {src}')


if __name__ == '__main__':
    main()
