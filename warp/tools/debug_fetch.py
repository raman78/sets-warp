#!/usr/bin/env python3
"""
Debug helper — fetches raw responses and shows exactly what the servers return.
Run from SETS root: python -m warp.tools.debug_fetch
"""
import sys, json, requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

s = requests.Session()
s.headers.update(HEADERS)

print("=" * 60)
print("TEST 1: vger.stobuilds.com/starship-traits")
print("=" * 60)
try:
    r = s.get('https://vger.stobuilds.com/starship-traits', timeout=20)
    print(f"Status:       {r.status_code}")
    print(f"Content-Type: {r.headers.get('content-type','?')}")
    print(f"Content-Len:  {len(r.content)} bytes")
    print(f"Headers: {dict(list(r.headers.items())[:8])}")
    print()
    print("First 3000 chars of body:")
    print("-" * 40)
    print(r.text[:3000])
    print()
    # Check for JSON data blocks
    import re
    nuxt = re.search(r'<script[^>]+id=["\']__NUXT_DATA__', r.text)
    winuxt = re.search(r'window\.__NUXT__', r.text)
    json_scripts = re.findall(r'<script[^>]+type=["\']application/json["\']', r.text)
    print(f"Has __NUXT_DATA__: {bool(nuxt)}")
    print(f"Has window.__NUXT__: {bool(winuxt)}")
    print(f"JSON script tags: {len(json_scripts)}")
    # Find all script tags
    scripts = re.findall(r'<script([^>]*)>', r.text)
    print(f"All script tags: {scripts[:15]}")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST 2: stowiki.net Cargo API — space_items")
print("=" * 60)
try:
    params = {
        'action': 'cargoquery', 'format': 'json',
        'tables': 'Infobox_starship_equipment',
        'fields': 'Name,Type,Rarity,Page',
        'limit': 5, 'offset': 0,
    }
    r2 = s.get('https://stowiki.net/w/api.php', params=params, timeout=20)
    print(f"Status:       {r2.status_code}")
    print(f"Content-Type: {r2.headers.get('content-type','?')}")
    print(f"Content-Len:  {len(r2.content)} bytes")
    print()
    print("Body (first 2000):")
    print("-" * 40)
    print(r2.text[:2000])
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST 3: stowiki.net Cargo API — Starship_traits")
print("=" * 60)
try:
    params = {
        'action': 'cargoquery', 'format': 'json',
        'tables': 'Starship_traits',
        'fields': 'Name,Obtained,Page',
        'limit': 5,
    }
    r3 = s.get('https://stowiki.net/w/api.php', params=params, timeout=20)
    print(f"Status:       {r3.status_code}")
    print(f"Content-Type: {r3.headers.get('content-type','?')}")
    print(r3.text[:2000])
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST 4: stowiki basic API test")
print("=" * 60)
try:
    params = {'action': 'query', 'format': 'json', 'meta': 'siteinfo', 'siprop': 'general'}
    r4 = s.get('https://stowiki.net/w/api.php', params=params, timeout=20)
    print(f"Status: {r4.status_code}  CT: {r4.headers.get('content-type','?')}")
    print(r4.text[:500])
except Exception as e:
    print(f"ERROR: {e}")

