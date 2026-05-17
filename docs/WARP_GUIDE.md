# WARP & WARP CORE — User Guide

WARP reads your Star Trek Online screenshots and fills in your SETS build automatically.
WARP CORE lets you review, correct, and confirm what WARP found — and train the ML model to improve future recognition.

---

## Table of contents

1. [Preparing screenshots](#1-preparing-screenshots)
2. [Using WARP — import a build](#2-using-warp--import-a-build)
   - [Step 4 — Run recognition](#step-4--run-recognition)
   - [Step 4b — Automatic transfer to SETS](#step-4b--automatic-transfer-to-sets)
3. [WARP CORE — interface overview](#3-warp-core--interface-overview)
4. [Reviewing and correcting recognition](#4-reviewing-and-correcting-recognition)
5. [Confirming items and accepting results](#5-confirming-items-and-accepting-results)
6. [Community model — how it works](#6-community-model--how-it-works)
7. [Community sync details](#7-community-sync-details)
8. [Keyboard shortcuts](#8-keyboard-shortcuts)
9. [Tips and troubleshooting](#9-tips-and-troubleshooting)

---

## 1. Preparing screenshots

### What to capture

WARP reads the standard STO build screens. Open your ship/character loadout in-game and take full-screen screenshots of:

| Screen | Contains |
|--------|----------|
| Space Equipment | Weapons, shields, engines, deflector, devices, consoles |
| Ground Equipment | Ground weapons, armor, kit, kit modules, devices |
| Space Traits | Personal space traits, starship traits, reputation traits |
| Ground Traits | Personal ground traits, reputation traits |
| Bridge Officers | Boff seats and abilities (space or ground) |
| Specializations | Primary and secondary specialization trees |

> **Screenshot tip:** Use the default STO screenshot key (default: **Print Screen**) to save full-resolution screenshots. Cropped or resized images may reduce recognition accuracy.

### How many screenshots per folder

**One build per folder.** Each folder is one import session. You can mix screen types freely:

- **Separate screens** — one screenshot per game tab. WARP identifies each screen type automatically.
- **Mixed screen** — a single screenshot that combines multiple tabs (assembled view). WARP detects the layout automatically.
- **Partial** — only some screens (e.g. equipment only, or traits only). Slots for missing screens are left empty.

### Recommended folder structure

```
my_build/
    space_equipment.jpg
    space_traits.jpg
    boffs.jpg
    ground_equipment.jpg
```

<!-- screenshot: example folder with 4 screenshots in file manager -->

---

## 2. Using WARP — import a build

### Step 1 — Open the WARP dialog

Click the **WARP** button in the SETS menu bar (next to Export and Settings).

<!-- screenshot: SETS menu bar with WARP button highlighted -->

### Step 2 — Select screenshot folder

In the WARP dialog, click **Select folder** and navigate to the folder containing your screenshots.
The folder path appears in the field below the button.

### Step 3 — Choose build type

Use the **Build type** selector to match the SETS tab you want to fill:

| Build type | Fills |
|------------|-------|
| **Space Build** | Space equipment, consoles, boffs, space traits |
| **Ground Build** | Ground equipment, boffs, ground traits |
| **Space Skills** | Space skill tree point allocation |
| **Ground Skills** | Ground skill tree point allocation |

Choose the type that matches the screenshots you took. One import fills one tab.

### Step 4 — Run recognition

Click **Import**. WARP processes each screenshot through this pipeline:

```
   [screenshot.png]
        │
        ▼
  ┌─────────────────────┐
  │ 1. Screen classify  │  MobileNetV3-Small  →  SPACE_EQ / GROUND_EQ /
  └─────────────────────┘                        TRAITS / BOFFS / MIXED / …
        │
        ▼
  ┌─────────────────────┐
  │ 2. OCR              │  EasyOCR  →  ship_name / ship_type / ship_tier
  │    (space screens)  │              (anchored on T6-X2 / T6-X / Tx tokens)
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 3. ShipDB lookup    │  type-first match (783 ships)  →  ship profile
  │                     │  • exact type  • word-subset (boff-Jaccard tiebreak)
  │                     │  • fuzzy 0.68  • keyword fallback
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 4. Layout detection │  per build_type, in priority order:
  │                     │   • Strategy 0:  BOFF marker grid (BOFFS / MIXED)
  │                     │   • Strategy 1:  EQ geometry detector (OCR-anchored)
  │                     │   • Strategy 1G: Ground EQ geometry
  │                     │   • Strategy 2:  pixel analysis (legacy)
  │                     │   • Strategy 3:  learned anchors.json
  │                     │   • Strategy 4:  default calibration
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 5. Icon matching    │  per slot crop  →  RecognisedItem
  │                     │   template / histogram k-NN / ArcFace embed /
  │                     │   EfficientNet softmax / community pHash
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 6. Write to SETS    │  ship_data → align_space_frame → slot_equipment_item /
  │                     │  slot_trait_item; BOFF cluster→seat assignment
  └─────────────────────┘
```

A progress bar shows the current step. Recognition typically takes 5–30 seconds per screenshot, depending on the number of screens, image resolution, and whether your hardware is CPU- or GPU-accelerated.

### Step 4b — Automatic transfer to SETS

WARP does not stop at recognition — once a screenshot is processed, it writes the result directly into the SETS build. The transfer happens in `_apply_to_sets()` (warp_dialog.py) and follows this sequence:

```
RecognisedItem stream from importer
        │
        ├──► ship_type / ship_tier        ──► _resolve_and_apply_ship()
        │                                       • match against sets_app.cache.ships
        │                                       • set ship button text + image
        │                                       • populate tier combo (T6 / T6-X / T6-X2)
        │                                       • align_space_frame() — rebuilds the
        │                                         SPACE/GROUND tab to match ship's
        │                                         slot counts (consoles, devices,
        │                                         hangars, sec-def, …)
        │
        ├──► equipment & traits           ──► _import_equipment_and_traits()
        │                                       • SLOT_MAP translates WARP slot
        │                                         names → SETS build keys
        │                                       • slot_equipment_item / slot_trait_item
        │                                         writes the item into the right column
        │                                       • universal-console overflow handled
        │                                         (extra Universal consoles get queued)
        │
        ├──► BOFF abilities (Boff *)      ──► _write_boffs_to_build()
        │                                       • cluster abilities by Y proximity
        │                                       • match each cluster to a ship seat
        │                                         (profession + spec)
        │                                       • _slot_indices_from_x maps each
        │                                         ability to its rank position using
        │                                         X-gap analysis (handles empty mid-slots)
        │
        └──► virtual items (__empty__,    ──► silently skipped on the SETS write
             __inactive__)                       (still uploaded to HF for training)
```

After all items are written, the dialog switches the SETS tab to match the build type and runs `sets.autosave()`. A summary message box reports detected / imported / unmatched counts and the ship that WARP picked.

**If the ship is not recognised** — the ship dropdown is cleared, slots are left at their default profile, and you can pick the correct ship manually in SETS afterwards. The equipment items that WARP did identify are still imported into the default layout where possible.

### Step 5 — Review results

After import, a **Results summary** shows:
- Ship detected (with tier)
- Slots filled vs total
- Average recognition confidence
- Items that need manual review (low confidence)

Items with confidence below the threshold are flagged. Click **Open WARP CORE** to review and correct them.

<!-- screenshot: WARP dialog after successful import, showing results summary -->

---

## 3. WARP CORE — interface overview

WARP CORE opens as a separate window with three panels.

```
+------------------+------------------------------+----------------------+
|   LEFT PANEL     |       CENTER PANEL           |    RIGHT PANEL       |
|                  |                              |                      |
|  Screenshots     |   [canvas / screenshot]      |  Recognition Review  |
|  ----------      |                              |  ----------          |
|  screen1.png  [ok]|  (zoom with Ctrl+wheel)     |  Slot: Fore Weapon 1 |
|  screen2.png  [?] |  (bboxes drawn on items)    |  Item: Phaser Array  |
|  screen3.png  [ ] |                              |  Conf: 94%  [green]  |
|                  |                              |                      |
|  [progress bar]  |  +------------------------+  |  Slot: Console Sci 1 |
|  3/6 confirmed   |  |  Slot:  [combo box  ]  |  |  Item: ???  [red]    |
|                  |  |  Item:  [name field ]  |  |  Conf: 31%           |
|                  |  |  [  Accept (Enter)  ]  |  |                      |
|                  |  +------------------------+  |  [+ Add BBox] [- Rm] |
|                  |                              |  [x] Auto >= [0.75]  |
|                  |                              |  [  Accept (Enter) ] |
+------------------+------------------------------+----------------------+
```

### Toolbar

| Button | Action |
|--------|--------|
| **Detect Screen Types** | Classifies every screenshot in the folder using the MobileNetV3-Small screen classifier (Equipment / Traits / Boffs / Specializations / Mixed). Runs automatically when you open a folder — use the button to re-run it manually if you rename or replace files. Files you have already confirmed with a checkmark are skipped. |
| **Auto-Detect Slots** | Re-runs the full recognition pipeline on the **currently selected screenshot**. Items you have already confirmed are preserved and used as seeds for icon matching — only unconfirmed slots are re-processed. Use this after correcting a few items to let WARP retry the remaining ones with better context. |

---

### Left panel — Screenshots list

Lists every screenshot file in the folder. Each entry shows the detected screen type (e.g. *Space Equipment*, *Traits*, *Boffs*) and a **checkbox**:

- **Checked (✓)** — screen type confirmed by you; Detect Screen Types will not overwrite it
- **Unchecked** — screen type was auto-detected and is still tentative; may be updated on re-detect

**How to confirm a screen type:**
- **Tick the checkbox manually** next to the filename — confirms whatever type is currently shown.
- **Change the type via the Screen Type dropdown** (top of the center panel) — the correct type is set and the checkbox is ticked automatically.

Auto-detected types start unchecked. If the classifier guesses wrong, change the type in the dropdown and it will be confirmed immediately. Un-ticking a checkbox removes the manual override so the classifier can re-classify that file next time.

Click a filename to load it into the canvas.

#### Screenshot colour coding

| Colour | Meaning |
|--------|---------|
| White | No annotations yet |
| Light blue | Has annotations — in progress |
| Green | Marked Done — fully annotated and locked |

#### Marking a screenshot as Done

When you have finished annotating all items on a screenshot, click **✓ Mark Done** (below the progress bar) or press **Alt+D**. This:

- Locks the screenshot — no new bounding boxes can be added
- Saves the confirmed slot layout to the local layout database (used to improve auto-detection on similar screenshots)
- Colours the entry green in the list

If you need to make changes, click **↩ Back to Edit** (same button) or press **Alt+D** again to unlock it.

### Center panel — Canvas

Displays the current screenshot with coloured bounding boxes drawn over each detected item slot:

| Box colour | State | Meaning |
|------------|-------|---------|
| Red          | `pending`         | Detected but not yet reviewed — needs your attention |
| Green        | `confirmed (user)`| Accepted by you (Enter / autocomplete pick / Accept button) |
| Yellow/gold  | `confirmed (auto)`| Auto-accepted by the program because confidence ≥ Auto threshold (default 0.75). Persists across restarts so you can tell at a glance what *you* confirmed vs what the program did. Editing the name re-flags it as user-confirmed (green). |
| Cyan         | `text slot`       | Ship Name / Ship Type / Ship Tier — read by OCR, no icon matching, no confidence score |
| Grey (empty name) | `pending, no match` | The grid found this slot but the icon matcher had low confidence (< 0.35) or the match name had the wrong type for the slot. The bbox is kept so you can correct it manually — type the right name and Accept. |
| Gold crosshair | (drawing)       | While Alt+LMB drag is in progress — the bbox you're currently drawing |

Diagram:

```
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │░░░░░░░░░░░░░░│   │██████████████│   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
   │░░ pending  ░░│   │██  user OK ██│   │▓▓ auto OK  ▓▓│
   │░░░ (red)  ░░░│   │██ (green)  ██│   │▓▓ (yellow) ▓▓│
   └──────────────┘   └──────────────┘   └──────────────┘

   ┌──────────────┐   ┌──────────────┐
   │░░░░░░░░░░░░░░│   │┄┄┄┄┄┄┄┄┄┄┄┄┄┄│
   │░ text slot ░│   │  no match    │
   │░░░ (cyan) ░░│   │  (grey)      │
   └──────────────┘   └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
```

#### Ship Name / Ship Type / Ship Tier bboxes

Cyan bboxes are special — they are not matched against the item database. Instead, the text inside them is read by OCR and used to identify the ship and its tier. They behave differently from equipment bboxes:

- **No confidence score** — there is no "correct/incorrect" percentage; the OCR result is shown as the item name.
- **No autocomplete** — type the text as it appears in the screenshot if you need to correct it.
- **No duplicate warning** — Ship Type and Ship Tier bboxes are allowed to overlap each other (see below).

#### Ship Type and Ship Tier overlap

In STO screenshots the ship type label (e.g. *"Fleet Temporal Science Vessel"*) and the tier label (e.g. *"T6-X"*) appear close together and sometimes on adjacent lines.

When drawing these bboxes manually:

- **Ship Type** — draw the bbox over the **full ship type text**, even if it spans two lines. A two-line Ship Type bbox will physically overlap with the Ship Tier bbox. This overlap is **intentional and expected** — the duplicate warning does not trigger for cyan text slots.
- **Ship Tier** — draw a **separate, smaller bbox** that covers **only the tier token** (e.g. just the `T6-X` fragment). Do not include the ship type text in this bbox.

Example:

```
┌─────────────────────────────────────┐  ← Ship Type bbox (full 2 lines)
│ Fleet Temporal Science Vessel       │
│ T6-X                   ┌───────┐   │  ← Ship Tier bbox (tier token only)
└────────────────────────┤  T6-X ├───┘
                         └───────┘
```

If both bboxes are confirmed correctly, WARP will extract the ship class from Ship Type and the upgrade tier from Ship Tier independently.

| Action | How |
|--------|-----|
| Zoom in / out | **Ctrl + scroll wheel** (1× – 6×, anchored to cursor) |
| Select a box | **Left click** on the box — highlights it in the review list |
| Draw new box | **Alt + LMB drag** — hold Alt, click and drag over an item icon |
| Draw mode toggle | **Alt+A** button in the right panel — cursor stays as crosshair until toggled off |

Below the canvas is the **Annotate panel**:
- **Slot** — dropdown to select the slot type for the current box
- **Item** — text field with autocomplete; type the item name
- **Accept** — confirms the item (also triggered by Enter)

### Right panel — Recognition Review

Lists all items detected in the current screenshot, one row per slot. Each row shows:
- Slot name (e.g. "Fore Weapon 3")
- Recognised item name (or "???" if not matched)
- Confidence percentage, colour-coded:
  - **Green** ≥ 75% — confident match
  - **Yellow** 40–74% — uncertain, review recommended
  - **Red** < 40% — poor match, manual correction needed

At the bottom:
- **Add BBox** — enter draw mode to add a missing box (Alt+A)
- **Remove** — delete the selected box (Alt+R or Del)
- **Auto ≥ [threshold]** — automatically accept items above the threshold; adjust the spinner to change it (default 0.75)

<!-- screenshot: WARP CORE window with a loaded screenshot and mixed confirmed/pending items -->

---

## 4. Reviewing and correcting recognition

### Step-by-step user journey

The intended WARP CORE workflow, end to end:

```
   START: open WARP CORE → pick a folder of screenshots
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │ A. SCREEN TYPES                                 │
   │    On folder open, every file is classified     │
   │    (SPACE_EQ / GROUND_EQ / TRAITS / BOFFS /     │
   │     SPECIALIZATIONS / SPACE_MIXED / …).         │
   │    For each file: check the type next to the    │
   │    filename. If correct → tick the checkbox.    │
   │    If wrong → change it via the Screen Type     │
   │    dropdown above the canvas (auto-ticks).      │
   └─────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │ B. AUTO-DETECT SLOTS                            │
   │    Select a screenshot in the left panel.       │
   │    Click "Auto-Detect Slots" (toolbar) to run   │
   │    the recognition pipeline on it.              │
   │    Items appear in the right panel's review     │
   │    list, sorted by confidence (lowest first).   │
   │    Items above the Auto-accept threshold are    │
   │    auto-confirmed (yellow) immediately.         │
   └─────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │ C. REVIEW & CORRECT                             │
   │    For each remaining pending (red) item:       │
   │      • Click it → canvas highlights the box.    │
   │      • Correct name? Press Enter / click Accept │
   │        → box turns green.                       │
   │      • Wrong name? Type the right one (autocom- │
   │        plete) and Accept.                       │
   │      • Box in wrong place? Del to remove, then  │
   │        Alt+drag a new one on the right icon.    │
   │    For grey (empty-name) bboxes: type the name. │
   │    For missing slots: Alt+drag directly on the  │
   │    canvas to add a new bbox.                    │
   └─────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │ D. MARK DONE                                    │
   │    When the screenshot is fully reviewed, click │
   │    "✓ Mark Done" (or Alt+D).                    │
   │    Locks the screenshot, saves its layout to    │
   │    anchors.json (improves future detection on   │
   │    similar screens), colours the entry green    │
   │    in the file list.                            │
   └─────────────────────────────────────────────────┘
        │
        ▼
   Repeat B–D for the next screenshot in the folder.
        │
        ▼
   END: confirmed crops upload automatically to HF
        every 10 min in the background.
```

### Typical micro-loop (per item)

1. Pick the lowest-confidence red item in the review list (sorted automatically).
2. Compare what's in the Item field against the icon on the canvas.
3. **Correct** → **Enter**. **Wrong** → type the correct name, pick from autocomplete (instant accept, no Enter needed).
4. **Bbox in wrong position** → **Del** to remove, **Alt+drag** to redraw.
5. Move to the next item.

> **Tip:** Auto ≥ 0.75 already handles the easy cases. In practice, you usually only need to review red items (< 40%) and the occasional yellow-confidence item.

### Correcting item names

The Item field has **autocomplete** — start typing the item name and a dropdown shows matches from the STO item database. Select with arrow keys or mouse.

If the correct item is not in the autocomplete list, type the full name manually. It will still be saved as training data for that icon.

> **Tip:** You don't have to correct every item. Focus on red (< 40%) and yellow items. Green items (≥ 75%) are usually correct — Auto-accept handles these automatically if enabled.

### Adding a missing bounding box

If WARP missed a slot entirely (no box drawn over an item):

1. Hold **Alt** and drag directly on the canvas — or click **Add BBox** / press **Alt+A** to lock draw mode on.
2. The cursor changes to a gold crosshair.
3. Drag over the item icon to draw a box.
4. The box is recognised immediately. Correct the name if needed, then Accept.

### Annotating empty and inactive Bridge Officer slots

Bridge officer seats sometimes contain slots that are visually empty (nothing assigned yet) or inactive (locked/unavailable at this rank). Annotating these helps WARP correctly position recognised abilities even when some slots are unoccupied.

**How to annotate:**

1. Draw a bounding box over the empty or inactive slot icon as you would for a normal ability.
2. In the **Item** field, type `__empty__` (for an unoccupied slot) or `__inactive__` (for a locked/unavailable slot).
3. Press **Enter** to confirm.

The review list shows these with grey labels `[empty slot]` / `[inactive slot]`. They write nothing to your SETS build but are uploaded to the community dataset so the model learns to recognise these slot states.

> **Why bother?** WARP uses the position of all detected icons — including empty and inactive ones — to determine which ability belongs in which seat slot. Annotating gaps makes the positioning more accurate, especially for Commander seats (4 slots) where only 1–2 abilities are present.

### Removing a wrong bounding box

If a box covers the wrong area or a non-slot area:

1. Click the box on the canvas (or the item in the review list) to select it.
2. Press **Del** or **Backspace**, or click **Remove**.

<!-- screenshot: WARP CORE canvas with a red item selected and the Item field showing the correct name typed -->

---

## 5. Confirming items and accepting results

### Manual accept

With an item selected:
- Press **Enter**, or
- Click **Accept** in the right panel or bottom panel.

A confirmed item turns green in the review list and on the canvas.

### Auto-accept

Enable the **Auto ≥** checkbox to automatically accept items above the confidence threshold without pressing Enter for each one. The threshold spinner sets the cutoff (default 0.75 = 75%).

Auto-accept applies:
- When a screenshot is loaded (immediately marks high-confidence items)
- After drawing a new bounding box
- After running Auto-Detect

The checkbox state and threshold value are saved across sessions (per user).

### Selecting from the autocomplete dropdown

Choosing an item from the autocomplete dropdown confirms it immediately — no Enter needed. This is the fastest way to correct a wrong item: type a few letters, pick from the list, done.

### Duplicate warning

If you confirm an item into a slot that already has a confirmed item at the same position (>70% overlap), WARP CORE shows a warning. This prevents accidentally confirming the same physical slot twice.

---

## 6. Community model — how it works

### Architecture

WARP uses a central **EfficientNet-B0** icon classifier trained on crops contributed by all users.
There is no local training — the model is trained once per hour by the community pipeline and
automatically downloaded to your installation.

### Your contribution

Every time you confirm an item annotation in WARP CORE, the icon crop is queued for upload to
HuggingFace (`sets-sto/sto-icon-dataset`). The upload happens automatically in the background
**every 10 minutes** (first run 15 s after app start) when a HuggingFace token is configured.

**What is sent:** Only the icon crop image (small PNG, ~64×64 px) and its label (item name +
slot type). No screenshots, no ship names, no personal data.

### Model update

A new model is trained hourly on GitHub Actions using all community crops. Your installation
checks for updates **every 15 minutes** (rate-limit cache; uses `requests` with 5 s connect /
60 s read timeouts to survive Render free-tier cold-starts) and downloads the new
`icon_classifier.pt` automatically if a newer version is available.

| File | Source | Update interval |
|------|--------|----------------|
| `warp/models/icon_classifier.pt` (EfficientNet-B0 softmax) | `sets-sto/warp-knowledge` (HF) | 15 min check |
| `warp/models/icon_embedder.pt` (ArcFace embedder) | `sets-sto/warp-knowledge` (HF) | 15 min check |
| `warp/models/screen_classifier.pt` (MobileNetV3-Small) | `sets-sto/warp-knowledge` (HF) | 15 min check |
| `warp/models/community_anchors.json` (learned layouts) | `sets-sto/warp-knowledge` (HF) | 15 min check |
| `warp/models/ship_type_corrections.json` (OCR correction map) | `sets-sto/warp-knowledge` (HF) | 15 min check |

---

## 7. Community sync details

### What is sent to HuggingFace

- The icon crop image (small PNG, ~64×64 px) — just the item icon, cropped from your screenshot
- The item name and slot type you confirmed
- An anonymous installation ID (random UUID, generated at first launch, stored locally)

No username, account information, full screenshots, or ship names are ever transmitted.

### Privacy note for Ship Name bbox

If you draw a bbox over the ship name text in WARP CORE, the **position** (coordinates) is saved
locally in `annotations.json` for layout learning. The **text content** is never saved as a crop
and never uploaded — ship names are treated as personal data.

---

## 8. Keyboard shortcuts

### WARP CORE

| Shortcut | Action |
|----------|--------|
| **Enter** | Accept current item |
| **Del** / **Backspace** | Remove selected bounding box |
| **Alt + A** | Toggle Add BBox draw mode |
| **Alt + D** | Toggle Mark Done / Back to Edit |
| **Alt + R** | Remove selected bounding box |
| **Alt + LMB drag** | Draw new bounding box directly |
| **Ctrl + Wheel** | Zoom canvas in/out (1× – 6×, anchored to cursor) |

### Bounding box colours (canonical)

| Colour | Meaning |
|--------|---------|
| Red | Pending — awaiting your review |
| Green | User-confirmed (Enter / autocomplete pick / Accept) |
| Yellow / gold | Auto-confirmed by program (conf ≥ Auto threshold). Persists across restarts. |
| Cyan | Text slot (Ship Name / Type / Tier) — OCR, no icon matching |
| Grey (empty name) | Detected bbox without a usable match — type the name and Accept |
| Gold crosshair | Currently being drawn (Alt + LMB drag in progress) |

See section 3 for the full colour legend with diagram.

---

## 9. Tips and troubleshooting

### WARP didn't detect my ship

- Make sure at least one **Space Equipment** screenshot is included (ship name and type are read from the equipment screen).
- The ship name must be visible at the top of the screen in the screenshot.
- If OCR fails, WARP falls back to a keyword-based match using the slots it finds. Check the log for the detected ship name.

### Wrong ship was selected

After import, the ship dropdown in SETS shows what WARP detected. Click the dropdown and select the correct ship manually. The slot layout will update automatically.

### Recognition accuracy is low on first use

On a fresh install, WARP uses the community model. If you have unusual items, high-resolution screenshots, or a non-standard UI scale, accuracy may be lower initially. Run a few imports and confirm the results in WARP CORE — your confirmed crops are uploaded to the community dataset, and the model improves as more users contribute.

### "???" items after import

Items shown as "???" have a confidence below the minimum threshold (40%). They were not matched to any known item. To fix:
1. Open WARP CORE.
2. Find the ??? item in the review list (shown in red).
3. Type the correct item name in the Item field.
4. Accept.
5. Repeat for all ??? items — confirmed crops will be uploaded automatically and help improve future recognition.

### Import is slow

First import after a fresh install may take 30–60 seconds because EasyOCR initialises its language model. Subsequent imports are faster (model stays in memory while SETS-WARP is open).

On CPU-only hardware, the ML inference step adds 2–5 seconds per screenshot. This is normal.

### "Duplicate bbox" warning

This appears when two confirmed items overlap by more than 70% in the same screenshot. Usually this means you accidentally drew a bbox over an area that already has a confirmed item. Remove the duplicate using Del and re-confirm if needed.

### Training completes but accuracy doesn't improve

- Make sure you have confirmed items from multiple different screenshots, not just one.
- If you only have a handful of unique items, accuracy metrics may fluctuate — this is normal with small datasets.
- More data always helps. Confirm items from 5–10 screenshots before training for the best results.

### The WARP button is greyed out

WARP is only available in **SETS + WARP** installations. If you chose **SETS only** during setup, the WARP button is not present. To add WARP:
1. Delete `.config/install_mode.txt` in the SETS-WARP folder.
2. Relaunch — the setup window will appear and let you choose SETS + WARP.
3. The additional ~2 GB of dependencies will be downloaded automatically.
