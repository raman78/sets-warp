# WARP & WARP CORE — User Guide

WARP reads your Star Trek Online screenshots and fills in your SETS build automatically.
WARP CORE lets you review, correct, and confirm what WARP found — and train the ML model to improve future recognition.

---

## Table of contents

1. [Preparing screenshots](#1-preparing-screenshots)
2. [Using WARP — import a build](#2-using-warp--import-a-build)
3. [WARP CORE — interface overview](#3-warp-core--interface-overview)
4. [Reviewing and correcting recognition](#4-reviewing-and-correcting-recognition)
5. [Confirming items and accepting results](#5-confirming-items-and-accepting-results)
6. [Training the ML model](#6-training-the-ml-model)
7. [Community model sync](#7-community-model-sync)
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

Click **Import**. WARP processes each screenshot:

1. Classifies the screen type (Equipment / Traits / Boffs / etc.)
2. Detects the ship name, tier, and type from OCR (space screens)
3. Reads slot layout from pixel analysis and the ship database
4. Matches each slot's icon against the community and local model
5. Fills matching equipment/traits/boffs into your SETS build

A progress bar shows the current step. Recognition typically takes 5–30 seconds depending on the number of screenshots and your hardware.

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
| **Train Model** | Fine-tunes the local EfficientNet-B0 icon classifier on all confirmed crops in `warp/training_data/`. See [Training the ML model](#6-training-the-ml-model). |

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

### Center panel — Canvas

Displays the current screenshot with coloured bounding boxes drawn over each detected item slot:

| Box colour | Meaning |
|------------|---------|
| Red | Pending — not yet reviewed / confirmed |
| Green | Confirmed (accepted by user) |
| Cyan | Text slot (Ship Name / Ship Type / Ship Tier) — read by OCR, no icon matching |
| Yellow | Bbox currently being drawn (Alt+LMB drag) |

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

### Typical workflow

1. Open WARP CORE (from the WARP results dialog, or click the **WARP CORE** button in the menu bar).
2. Select the first screenshot in the left panel.
3. Look at the review list on the right — items are sorted by confidence (low first).
4. Click a red or yellow item to select it. The canvas highlights the corresponding box.
5. Check whether the recognised item is correct:
   - **Correct** → press **Enter** or click **Accept** to confirm.
   - **Wrong** → type the correct name in the Item field, then press Enter to confirm.
   - **Box in wrong position** → remove the box (Del) and draw a new one with Alt+drag.
6. Move through all items, then click the next screenshot in the left panel.

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

## 6. Training the ML model

### What training does

WARP uses an **EfficientNet-B0** image classifier to recognise item icons. Every time you confirm an annotation in WARP CORE, the icon crop and its label are saved to `warp/training_data/`. Clicking **Train Model** fine-tunes the classifier on all confirmed crops so it recognises those icons more accurately in future imports.

### When to train

Train after accumulating a meaningful number of new confirmations — typically:
- First time: after confirming 30–50 items across several screenshots
- Ongoing: after each session where you corrected 10+ items

There is no harm in training more often, but each training run takes 1–5 minutes (CPU-only).

### How to run Train Model

1. Click **Train Model** in the WARP CORE toolbar.
2. A progress dialog shows: epoch number, loss, and validation accuracy.
3. When training completes, a summary shows the final accuracy and number of classes.
4. The updated model is saved to `warp/models/icon_classifier.pt` and used immediately for the next import.

<!-- screenshot: WARP CORE Train Model progress dialog mid-training -->

### Training parameters (internal)

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNet-B0 |
| Input size | 64×64 px (item icon crop) |
| Epochs | 20 (early stop on plateau) |
| Optimizer | Adam, lr=1e-4 |
| Loss | Focal loss (handles class imbalance) |
| Batch size | 32 |
| Device | CPU (GPU if available) |

### Local vs community model

SETS-WARP has two model sources:

| Source | File | Updated by |
|--------|------|-----------|
| **Local model** | `warp/models/icon_classifier.pt` | You, via Train Model |
| **Community model** | Downloaded from HuggingFace | Automatically, from all users |

The local model always takes priority after you train. It is personalised to your screenshot style, resolution, and the specific items in your builds.

The community model is downloaded at startup if a newer version is available (checked once per 24 hours). It is used as the baseline before your local model exists, and as a fallback for items you have never confirmed.

---

## 7. Community model sync

### How it works

When you confirm an item in WARP CORE, the icon crop is sent anonymously to a shared community server (`sets-warp-backend` on Render). The server collects contributions from all users and merges them using majority vote per icon hash. A central EfficientNet model is retrained hourly from the merged dataset and published to HuggingFace.

Your SETS-WARP instance downloads the updated community model once per 24 hours in the background (8 seconds after launch) if a newer version is available.

### What is sent

- The icon crop image (64×64 px) — just the item icon, no screenshot, no personal data
- The item name you confirmed
- An anonymous installation ID (random UUID, generated at first launch, stored locally)

No username, account information, or screenshot content is ever transmitted.

---

## 8. Keyboard shortcuts

### WARP CORE

| Shortcut | Action |
|----------|--------|
| **Enter** | Accept current item |
| **Del** / **Backspace** | Remove selected bounding box |
| **Alt + A** | Toggle Add BBox draw mode |
| **Alt + R** | Remove selected bounding box |
| **Alt + LMB drag** | Draw new bounding box directly |
| **Ctrl + Wheel** | Zoom canvas in/out (1× – 6×, anchored to cursor) |

### Bounding box colours

| Colour | Meaning |
|--------|---------|
| Green | Confirmed (accepted) |
| Gold/yellow | Pending (awaiting review) |
| Grey | Skipped or low-confidence |
| Red | Error / removed |

---

## 9. Tips and troubleshooting

### WARP didn't detect my ship

- Make sure at least one **Space Equipment** screenshot is included (ship name and type are read from the equipment screen).
- The ship name must be visible at the top of the screen in the screenshot.
- If OCR fails, WARP falls back to a keyword-based match using the slots it finds. Check the log for the detected ship name.

### Wrong ship was selected

After import, the ship dropdown in SETS shows what WARP detected. Click the dropdown and select the correct ship manually. The slot layout will update automatically.

### Recognition accuracy is low on first use

On a fresh install, WARP uses the community model. If you have unusual items, high-resolution screenshots, or a non-standard UI scale, accuracy may be lower initially. Run a few imports and confirm the results in WARP CORE — after training the local model, accuracy improves significantly.

### "???" items after import

Items shown as "???" have a confidence below the minimum threshold (40%). They were not matched to any known item. To fix:
1. Open WARP CORE.
2. Find the ??? item in the review list (shown in red).
3. Type the correct item name in the Item field.
4. Accept.
5. Repeat for all ??? items, then run **Train Model**.

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
