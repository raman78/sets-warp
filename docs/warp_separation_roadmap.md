# Roadmap: SETS-WARP Separation & Upstream Merge

## Goal

1. **Decouple WARP-specific logic from `src/`** so upstream SETS can be merged with minimal conflicts.
2. **Merge upstream SETS** into SETS-WARP to catch up with 738 commits of upstream development.
3. **Preserve 100% of SETS-WARP functionality** — no feature may be lost at any step.
4. **Contribute select improvements back upstream** after separation is stable.

## Guiding Principles

- **Zero Functionality Loss**: No SETS or WARP feature, no user setting may be lost.
- **Upstream is master for `src/`**: when merging, upstream code wins unless it directly breaks WARP — then we discuss before acting.
- **WARP is a plugin**: all WARP-specific logic lives in `warp/` and is injected at runtime via inheritance from `WarpSETS(SETS)`.
- **Small safe steps**: each phase is independently testable. No big-bang changes.
- **Talk before risky moves**: any conflict marked HIGH risk requires discussion before resolution.

## Testing & Verification Responsibility

**All testing, refactoring and auditing at every step is performed by Claude** — the user is not expected to run tests or inspect code manually.

Methods used (in order of preference):
1. Static code analysis — reading files, grep, diff
2. Log inspection — running the app in background, reading output
3. Any other tools available — bash, git, linters

**The user is asked to intervene only when truly necessary** (e.g. visual UI verification, confirming a specific click path, or checking something only visible in the running app). When that happens, Claude provides exact step-by-step instructions:
- What to launch / click / type
- What input to provide
- What the expected output or visual result should be
- What to report back if it differs

---

## Current State Analysis (2026-03-27)

### Branch divergence
- **738 commits** upstream ahead of us — upstream SETS has been actively developed since our fork
- **180 commits** we are ahead of upstream — our WARP additions
- Protected snapshot: branch `stable-1.8b` (created at Phase 0)

### What upstream SETS added since our fork (features we want)
| Area | Example commits |
|------|----------------|
| Skill tree (space + ground) | Multiple |
| Markdown export | `e4c9b99`, `f83d8d4`, `70d2ef8` |
| Settings page with UI scale | `0a018d5` |
| About sidebar | `eb0289e` |
| Picker improvements + relative position fix | `2dfd9c9`, `8fcccb0` |
| Save format preference (JSON/PNG) | `00e04a4` |
| Boff/console/temporal operative bug fixes | `7baa777`, `6368f8c`, `280cc7c` |
| Modifier data refinement | `14dfaf8`, `5e61133` |
| Legacy build conversion | `d2bb7d2` |
| Linux path fixes (`os.path.join` everywhere) | `9c5b384` |

We want ALL of these — they are valuable SETS improvements.

### Files in `src/` and their status

#### Files ENTIRELY OURS — no merge conflict possible
These files do not exist in upstream. Git keeps them automatically on merge.

| File | Lines | Notes |
|------|-------|-------|
| `src/setsdebug.py` | 60 | Our logging system — used everywhere |
| `src/syncmanager.py` | 604 | Our sync + GitHub cache fallback |
| `src/cargomanager.py` | 46 | Our cargo data manager |
| `src/downloader.py` | 196 | Our HTTP downloader |
| `src/imagemanager.py` | 141 | Our image manager |

No action needed for these. After merge, verify they still import correctly.

#### Files modified on BOTH sides — real conflict risk

| File | Upstream adds | We add | Risk |
|------|--------------|--------|------|
| `src/app.py` | Skill tree init, settings page, about sidebar, markdown export wiring, Picker fixes | CargoManager/ImageManager/Downloader setup, WARP injection, Cloudflare cookies | **HIGH** |
| `src/datafunctions.py` | Modifier refinements, legacy build fixes, skill loading (-470/+68 net diff) | Our data pipeline | **HIGH** |
| `src/iofunc.py` | `os.path.join` fixes, maintenance functions, save dialog improvements | Our utility additions | **MEDIUM** |
| `src/callbacks.py` | Boff fixes, console slot fix, skill unlock callback | `_save_session_slots`, `_restore_session_slots` | **LOW** — our additions are isolated |
| `src/buildupdater.py` | Skill tree integration, `convert_old_build` improvements | DC ships (`equipcannons`), item normalization, boff ability aliases | **MEDIUM** |
| `src/constants.py` | Minor upstream additions | `SEVEN_DAYS_IN_SECONDS`, `GITHUB_CACHE_URL`, species sets | **LOW** |
| `src/widgets.py` | Upstream also added `ImageLabel` (different impl) | Our `ImageLabel`, `TooltipLabel`, `reset_cache` improvements | **MEDIUM** — both added ImageLabel |
| `src/splash.py` | New splash layout | Our `splash_progress` | **LOW** |
| `src/subwindows.py` | Small changes | Small changes | **LOW** |
| `src/export.py` | Markdown export restructure | Minor | **LOW** |
| `src/textedit.py` | Minor | Minor | **LOW** |

### WARP-specific code still in `src/` (must move to `warp/` before merge)

| Code | Current location | Target |
|------|-----------------|--------|
| `_save_session_slots()` | `src/callbacks.py` | `warp/session.py` |
| `_restore_session_slots()` | `src/callbacks.py` | `warp/session.py` |
| Debug `print('[SETS]...')` statements | `src/app.py` | Remove entirely |
| WARP injection, install_mode, `_WARP_AVAILABLE` | `src/app.py` | **Already done** in `warp/app.py` (Phase 1) |

---

## Phases

### Phase 0 — Branch protection (do immediately) ✅

Create `stable-1.8b` branch from current HEAD before any work begins.

```bash
git checkout -b stable-1.8b
git push origin stable-1.8b
git checkout main
```

This is the safety net. If anything goes wrong at any later phase, we return here.

---

### Phase 1 — WARP entry point ✅ DONE

- `warp/app.py` created with `WarpSETS(SETS)` class
- Version strings, app naming, Windows taskbar ID moved out of `src/app.py`
- `main.py` updated to use `warp.app.WarpSETS`

---

### Phase 2 — Remove remaining WARP code from `src/` ✅ DONE

**Goal**: after this phase, `src/` contains zero WARP-specific logic.

#### 2.1 Session slot functions — REVISED (NO MOVE NEEDED)

**Decision**: `_save_session_slots()` and `_restore_session_slots()` stay in `src/callbacks.py`.
They are called from `select_ship()` and `tier_callback()` in `src/callbacks.py` itself — not only
from `warp_dialog.py`. Moving them to `warp/` would create a wrong `src/ → warp/` dependency.
These are SETS improvements (preserve equipment on ship switch), not WARP-exclusive.

#### 2.2 Clean debug prints from `src/app.py` ✅

- Removed all `print('[SETS]...')` statements added during development (lines 2, 4, 7, 12)

#### 2.3 Remove WARP code from `src/app.py` ✅

- Removed `_MODE_FILE`, `_get_install_mode`, `_save_install_mode`, `_WARP_AVAILABLE` from `src/app.py`
  (these now live exclusively in `warp/app.py`)
- Removed `inject_warp_buttons` import from `src/app.py`
- Removed `if _WARP_AVAILABLE: inject_warp_buttons(...)` from `SETS.setup_main_layout`
- Stored `menu_layout` as `self.widgets.menu_layout` — accessible to `WarpSETS` override
- Removed WARP Updates + Installation sections from `SETS.setup_settings_frame`
- Removed Uninstall section from `SETS.setup_settings_frame`
- Removed `_on_uninstall` and `_run_uninstall` methods from `SETS`
- Changed `create_main_window` app naming: `'sets-warp'/'SETS-WARP'` → `'SETS'/'STOCD'` (upstream defaults)
- Stored `scroll_layout` as `self.widgets.settings_scroll_layout` — accessible to `WarpSETS` override

**`warp/app.py` now implements**:
- `create_main_window()` override: sets app name back to `'sets-warp'/'SETS-WARP'` after super()
- `setup_main_layout()` override: calls `inject_warp_buttons(self, self.widgets.menu_layout)`
- `setup_settings_frame()` override: calls `_add_warp_settings_sections()` which appends
  WARP Updates, Installation, and Uninstall sections to `self.widgets.settings_scroll_layout`
- `_on_uninstall()` and `_run_uninstall()` methods moved here from `src/app.py`

**Checkpoint**: full smoke test — SETS opens, WARP dialog works, WARP CORE opens.

---

### Phase 3 — Upstream merge ✅ DONE (2026-03-28)

Commit: `b263f6d` — all 19 conflicting files resolved, merge committed to `main`.

**This is the big step. Do not rush. One file at a time.**

#### 3.1 Dry-run: map actual conflicts

```bash
git merge --no-commit --no-ff upstream/main
git status
git merge --abort
```

Review the conflict list. Classify each file:
- **AUTO**: safe to take upstream, our changes are absent or trivial to re-add
- **MANUAL**: needs careful line-by-line review

#### 3.2 LOW-risk files — take upstream, re-apply our additions

Strategy for each: `git checkout upstream/main -- <file>`, then re-add our specific lines.

- `src/__init__.py` — take upstream
- `src/textedit.py` — take upstream, check if our 6 added lines still needed
- `src/export.py` — take upstream (upstream restructured markdown export significantly)
- `src/splash.py` — take upstream, re-add `splash_progress` if still needed by our code
- `src/subwindows.py` — take upstream, re-add our additions if any remain after Phase 2

#### 3.3 MEDIUM-risk files — manual merge, one at a time

**`src/callbacks.py`** (LOW after Phase 2.1):
- After session functions moved out, our changes are minimal
- Take upstream version, verify nothing WARP-related remains

**`src/constants.py`**:
- Take upstream version
- Re-add: `SEVEN_DAYS_IN_SECONDS`, `GITHUB_CACHE_URL`, species sets (`'Klingon': {...}` etc.)
- These are used by `src/syncmanager.py` and `src/cargomanager.py`

**`src/iofunc.py`**:
- Take upstream as base (it has `os.path.join` fixes that we want everywhere)
- Identify our additions: compare `git diff upstream/main HEAD -- src/iofunc.py`
- Re-add our unique functions on top
- Verify `src/syncmanager.py` still imports correctly

**`src/buildupdater.py`**:
- Take upstream as base
- Re-apply our additions method by method:
  - DC ship support: `if ship_data['equipcannons'] == 'yes': self.widgets.ship['dc'].show()`
  - Item normalization: `item.setdefault('mark', '')` / `setdefault('modifiers', [...])`
  - Boff ability alias resolution block
  - Intel Holoship `uni_consoles += 1` special case
- These are in different methods than upstream's skill tree additions — conflict unlikely but check

**`src/widgets.py`**:
- Both sides added `ImageLabel`. Compare implementations:
  - Ours: aspect-ratio preserving, uses `setPixmap`, `resizeEvent`
  - Upstream: may differ — take the better one or merge best of both
- Keep our unique additions: `TooltipLabel`, `reset_cache` improvements, `progress_bar`/`progress_detail` fields
- Keep our extended import list

#### 3.4 HIGH-risk files — discuss before each

**`src/datafunctions.py`** (HIGH):
- Largest diff: upstream -470 lines from our version, +68 new upstream lines
- Means upstream restructured this file significantly
- Strategy:
  1. Read upstream version fully
  2. Read our version fully
  3. List what we have that upstream doesn't — identify each addition
  4. Take upstream as base, re-add our additions function by function
  5. **STOP and discuss** if any upstream restructure makes our additions incompatible

**`src/app.py`** (HIGH — do last):
- Upstream added: skill tree `__init__` calls, settings page setup, about sidebar, markdown export wiring
- We added: `CargoManager`, `ImageManager`, `Downloader` init, Cloudflare cookie config
- These are additive sections in `__init__` — manual merge should be feasible
- Strategy: take upstream `__init__` as base, insert our manager setup blocks after upstream's equivalent init calls
- **STOP and discuss** if upstream's `__init__` structure conflicts with our insertions

#### 3.5 Verify our-only files still work

After all file merges, test integration of our exclusive files:
- `src/syncmanager.py` — verify imports from new `src/iofunc.py` still work
- `src/cargomanager.py` — verify imports still valid
- `src/downloader.py`, `src/imagemanager.py` — verify no interface breakage

#### 3.6 Full integration test after merge

- SETS-WARP opens → splash, ship selector ✓
- Skill tree loads (new upstream feature) ✓
- Markdown export works (new upstream feature) ✓
- Settings page opens (new upstream feature) ✓
- WARP dialog → import a screenshot ✓
- WARP CORE → load folder, zoom/cursor, accept item ✓
- Export build (all formats) ✓

---

### Phase 4 — Stabilization after merge ✅ DONE (2026-03-28)

All `warp/app.py` overrides verified compatible with merged `src/app.py`. No code changes needed.

| Override | Result |
|----------|--------|
| `WarpSETS.__init__(theme, args, path, config, versions)` | ✅ signature unchanged |
| `create_main_window(argv=[])` | ✅ app name override works after super() |
| `setup_main_layout()` | ✅ `self.widgets.menu_layout` hook present (src/app.py L374) |
| `setup_settings_frame()` | ✅ `settings_scroll_layout` (L1221) + `settings_scroll_frame` (L1273) hooks present |
| `_add_warp_settings_sections()` + `adjustSize()` | ✅ scroll frame hook present |

`warp_dialog.py` → `src/` dependencies verified: `_save/_restore_session_slots`, `align_space_frame`, `slot_equipment_item`, `slot_trait_item`, `get_boff_spec`, `clear_ship`, `load_boff_stations`, `ImageManager.get_ship_image` — all present.

#### 4.1 Update `warp/app.py` overrides

Upstream's `SETS.__init__` will have changed. Review `WarpSETS`:
- Check method signatures still match
- Check that `super().__init__()` call in `WarpSETS.__init__` passes correct arguments
- Update any overridden methods whose upstream base changed

#### 4.2 Document maintained differences ✅ DONE (2026-03-28)

`docs/src_patches.md` created — complete file-by-file reference of every intentional difference
between our `src/` and upstream SETS, including re-application instructions for future merges.

#### 4.3 Release stabilization

- Run full test cycle
- Tag `v1.9b` or `v2.0b` depending on scope of changes
- Update CHANGELOG with upstream feature list we inherited

---

### Phase 5 — Upstream contribution ✅ DONE (2026-03-28)

Three PRs opened to `Shinga13/SETS` from fork `raman78/SETS`:

| PR | Branch | Change |
|----|--------|--------|
| [#1](https://github.com/Shinga13/SETS/pull/1) | `fix/item-normalization-legacy-builds` | `setdefault` guards for `mark`/`modifiers` in `load_equipment_cat` — fixes crash on legacy saves |
| [#2](https://github.com/Shinga13/SETS/pull/2) | `fix/intel-holoship-uni-console` | Intel Holoship `uni_consoles += 1` — fixes missing Universal Console slot |
| [#3](https://github.com/Shinga13/SETS/pull/3) | `fix/species-sets-expansion` | Caitian, Ferasan, Talaxian, Klingon cross-faction added to SPECIES dict |

**NOT submitted (SETS-WARP exclusive or requires our infrastructure):**
- DC ship support — logic lives in `warp/`, requires our `TooltipLabel` + `dual_cannons` icon
- Boff ability alias resolution — requires `item_aliases` infrastructure we built
- `GITHUB_CACHE_URL` fallback — requires `syncmanager.py`

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| `src/app.py` merge breaks SETS `__init__` | HIGH | HIGH | Do last, `stable-1.8b` fallback, test after each block |
| `src/datafunctions.py` upstream restructure incompatible | MEDIUM | HIGH | Full read of both sides before touching, discuss if unclear |
| Our `syncmanager` breaks with new `iofunc` | LOW | HIGH | Test immediately after iofunc merge |
| `src/widgets.py` duplicate `ImageLabel` causes runtime error | MEDIUM | MEDIUM | Compare both, pick one, keep unique additions |
| WARP dialog breaks after `_session_slots` move (Phase 2.1) | LOW | HIGH | Update imports in warp_dialog.py in same commit |
| Upstream skill tree conflicts with our ship loading | LOW | MEDIUM | Skill tree is new code, unlikely to touch our DC/normalization additions |
| `warp/app.py` override signature mismatch after merge | MEDIUM | MEDIUM | Phase 4.1 explicitly reviews this |
| Losing `stable-1.8b` as fallback | VERY LOW | CRITICAL | Push to origin immediately in Phase 0 |

---

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-03-27 | Upstream is master for `src/` on conflicts | 738 commits of upstream work is more current; we don't lose features, we re-add on top |
| 2026-03-27 | Full merge (not cherry-pick) from upstream | We want ALL upstream features, not selective ones |
| 2026-03-27 | WARP code lives in `warp/` only — injected via inheritance | Enables clean upstream sync going forward |
| 2026-03-27 | No monkey patching for `buildupdater` | Fragile — breaks silently on upstream signature changes; subclass override instead |
| 2026-03-27 | `stable-1.8b` branch before any work | Safety net for all phases |
| 2026-03-27 | Upstream PR for genuine fixes (Phase 5) | DC ships, item normalization are genuine SETS fixes, not WARP-specific |
