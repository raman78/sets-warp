# SETS — User Guide

> For screenshot recognition and ML training, see **[WARP_GUIDE.md](WARP_GUIDE.md)**.

---

## Table of contents

1. [Screen layout](#1-screen-layout)
2. [Menu bar](#2-menu-bar)
3. [Sidebar — Ship & Character](#3-sidebar--ship--character)
4. [SPACE tab](#4-space-tab)
5. [GROUND tab](#5-ground-tab)
6. [SPACE SKILLS tab](#6-space-skills-tab)
7. [GROUND SKILLS tab](#7-ground-skills-tab)
8. [Settings](#8-settings)
9. [Saving and loading builds](#9-saving-and-loading-builds)
10. [Right-click context menu](#10-right-click-context-menu)
11. [Export](#11-export)

---

## 1. Screen layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SETS banner                                     │
├───────────────────────────────────────────────────────────────────────  ┤
│  [Save] [Open] [Clear Current Tab] [Clear All Tabs]                     │
│                [SPACE] [GROUND] [SPACE SKILLS] [GROUND SKILLS]          │
│                                    [Export] [Settings] [WARP] [WARP CORE]│
├──────────────────┬──────────────────────────────────────────────────────┤
│   SIDEBAR        │   BUILD AREA                                         │
│                  │                                                      │
│  [Ship image]    │  (contents change depending on active tab)           │
│  [Ship info]     │                                                      │
│  [Character]     │                                                      │
│                  │                                                      │
└──────────────────┴──────────────────────────────────────────────────────┘
```

The window is split into two areas:

- **Sidebar** (left) — ship selection, ship details, character settings
- **Build area** (right) — switches between Space, Ground, Space Skills, Ground Skills, and Settings depending on which tab is active

---

## 2. Menu bar

### Left group

| Button | Action |
|--------|--------|
| **Save** | Save the current build — writes both `.json` and `.png` simultaneously under the same filename. See [Saving and loading builds](#9-saving-and-loading-builds). |
| **Open** | Load a previously saved build from a `.json` or `.png` file. |
| **Clear Current Tab** | Clears all slots on the currently visible tab (Space or Ground). |
| **Clear All Tabs** | Clears all slots on all tabs. |

### Center group — tab switchers

| Button | Switches to |
|--------|------------|
| **SPACE** | Space equipment, boffs, traits, doffs |
| **GROUND** | Ground equipment, boffs, traits, doffs |
| **SPACE SKILLS** | Space skill tree point allocation |
| **GROUND SKILLS** | Ground skill tree point allocation |

### Right group

| Button | Action |
|--------|--------|
| **Export** | Opens the Export window — generates a shareable build summary (Markdown/text). |
| **Settings** | Opens the Settings tab in the build area. |
| **WARP** | Opens the WARP import dialog (SETS+WARP installation only). |
| **WARP CORE** | Opens the WARP CORE annotation and training window (SETS+WARP only). |

---

## 3. Sidebar — Ship & Character

The sidebar has two sections stacked vertically. The top section changes depending on the active tab; the bottom section (character) is always visible.

### Ship section (Space tab active)

| Field / Control | Description |
|-----------------|-------------|
| **Ship image** | Displays the selected ship's artwork. Updates automatically when a ship is picked. |
| **\<Pick Ship\> button** | Opens the ship selector dialog. Search and select your ship by name. |
| **Ship Tier** | Dropdown — sets the tier variant (T5, T5-U, T6, T6-X, T6-X2). Affects slot counts. |
| **Dual Cannons icon** | Appears next to Tier if the selected ship can equip dual cannons. |
| **Ship Info** | Opens a brief info panel with ship stats for the selected ship. |
| **Ship Name** | Free-text field — enter a custom name for your build's ship (optional). |
| **Build Description** | Multi-line text field — free-form notes about your build. Saved with the build file. |

### Ground sidebar (Ground tab active)

When the Ground tab is active, the top sidebar section shows a **Build Description** text field for ground-specific notes.

### Character section (always visible, bottom of sidebar)

| Field | Description |
|-------|-------------|
| **Name** | Captain name (free text). |
| **Elite Captain** | Checkbox — marks captain as Elite (affects some trait interactions). |
| **Captain Career** | Dropdown — Tactical / Engineering / Science. |
| **Faction** | Dropdown — Federation / Klingon / Romulan / Dominion / TOS Federation / etc. Changing faction updates the available species list. |
| **Species** | Dropdown — populated based on selected Faction. |
| **Primary Spec** | Dropdown — primary specialization (Temporal Operative, Miracle Worker, etc.). |
| **Secondary Spec** | Dropdown — secondary specialization. |

---

## 4. SPACE tab

The Space build area is divided into columns separated by vertical dividers.

### Weapons column

| Section | Max slots | Notes |
|---------|-----------|-------|
| **Fore Weapons** | 5 | Slot count determined by ship tier |
| **Aft Weapons** | 5 | Slot count determined by ship tier; hidden for ships with no aft |
| **Experimental Weapon** | 1 | Hidden if ship doesn't support it |
| **Devices** | 6 | Slot count varies by ship; T6-X2 adds +1 |
| **Hangars** | 2 | Hidden for non-carrier ships |

### Deflector / Engines column

| Section | Slots |
|---------|-------|
| **Deflector** | 1 |
| **Sec-Def** *(Secondary Deflector)* | 1 — Science ships only |
| **Engines** | 1 |
| **Warp Core** | 1 *(Warp Core for Federation-aligned, Singularity Core for Romulans)* |
| **Shield** | 1 |

### Consoles column

| Section | Max slots | Notes |
|---------|-----------|-------|
| **Universal Consoles** | 3 | T6-X adds +1 slot |
| **Engineering Consoles** | 5 | Ship-dependent count |
| **Science Consoles** | 5 | Ship-dependent count |
| **Tactical Consoles** | 5 | Ship-dependent count |

### Bridge Officers column

Six boff stations, each with:
- **Profession dropdown** — Tactical / Engineering / Science / Universal
- **Specialization dropdown** — Miracle Worker / Command / Intelligence / Pilot / Temporal / none
- **Rank slots** — 1–4 ability slots per seat (Ensign → Lieutenant → Lt. Commander → Commander)

Each rank slot is a dropdown to select the boff ability.

### Traits column

| Section | Slots |
|---------|-------|
| **Personal Space Traits** | Up to 10 |
| **Starship Traits** | 5 (T6-X2 adds +1) |
| **Reputation Traits** | 5 |
| **Active Reputation Traits** | 5 |

### Duty Officers (bottom strip)

**Space Duty Officers** — 6 doff slots displayed as a horizontal row at the bottom. Each slot is a text field with autocomplete.

---

## 5. GROUND tab

### Equipment column

| Section | Max slots |
|---------|-----------|
| **Kit Modules** | 6 |
| **Weapons** | 2 |
| **Devices** | 5 |

### Armor / Kit column

| Section | Slots |
|---------|-------|
| **Kit Frame** | 1 |
| **Armor** | 1 |
| **EV Suit** | 1 |
| **Shield** | 1 |

### Bridge Officers column

Four ground boff stations. Each station has a profession dropdown and 4 ability slots (ground abilities).

### Traits column

| Section | Slots |
|---------|-------|
| **Personal Ground Traits** | Up to 10 |
| **Reputation Traits** | 5 |
| **Active Reputation Traits** | 5 |

### Duty Officers (bottom strip)

**Ground Duty Officers** — 6 doff slots, same layout as space.

---

## 6. SPACE SKILLS tab

A scrollable skill tree for space combat. Skills are arranged in a grid with six columns (three skill categories × two columns each). Each skill node shows:

- Skill name
- Point allocation — click to add/remove points (up to 3 per node)
- Visual fill bar showing current allocation

The right panel shows a **point budget** — total points spent vs. available, broken down by category.

**Save Skills / Open Skills** buttons at the top work independently of the equipment build save.

---

## 7. GROUND SKILLS tab

Same layout as Space Skills but for ground combat skill tree.

---

## 8. Settings

Accessed via the **Settings** button in the menu bar. The settings area is a scrollable panel divided into sections.

### Settings

| Option | Values | Description |
|--------|--------|-------------|
| **UI Scale** | Slider (25–75) | Scales the entire UI. **Requires restart to take effect.** |
| **Default Mark** | Dropdown (blank, I–XV) | Pre-fills the Mark field when adding new items via the picker. |
| **Default Rarity** | Dropdown (Common → Ultra Rare) | Pre-fills the Rarity field in the item picker. |
| **Picker Position** | Absolute / Relative | Controls where the item picker window opens. *Absolute* = fixed screen position; *Relative* = near the clicked slot. |
| **Preferred Backup** | Auto / Manual | Whether cargo data backup runs automatically or only when you click *Backup Cargo Data*. |

### SETS-WARP Updates *(only shown in SETS+WARP installs)*

| Option | Description |
|--------|-------------|
| **Check for updates automatically** | Checkbox — enables background update check 8 seconds after launch. |
| **Installed version** | Shows the currently installed version label. |

### Installation

| Option | Description |
|--------|-------------|
| **SETS + WARP** checkbox | Checked = full installation with screenshot recognition and ML training (~2.5 GB). Unchecked = SETS-only (~500 MB), WARP buttons hidden. Changing this setting requires a restart — the installer runs automatically on next launch to add or disable WARP dependencies. |

Switching from SETS-only to SETS+WARP downloads ~2 GB of ML packages (PyTorch, EasyOCR, OpenCV) on the next launch. Switching from SETS+WARP to SETS-only hides the WARP buttons immediately after restart; installed ML packages remain on disk (disk space is not reclaimed automatically).

### Maintenance

| Button | Action |
|--------|--------|
| **Clear Cargo Data** | Deletes downloaded cargo JSON files. Restart to re-download fresh data from STO wiki / GitHub. |
| **Clear Cache** | Deletes the in-memory build cache. Restart to rebuild. |
| **Backup Cargo Data** | Creates a backup copy of cargo data to protect against download failures. |

### Compatibility

| Button | Action |
|--------|--------|
| **Convert Legacy Build Image** | Loads a build from an old-format `.png` build image (pre-JSON era). Use the regular **Open** button for legacy `.json` files. |

### About SETS (sidebar)

Shows a short description and links: **Website**, **Github**, **STOBuilds Discord**, **Downloads**.

---

## 9. Saving and loading builds

### Save

Click **Save** in the menu bar. SETS always writes **both formats simultaneously** under the same base filename:

- **`.json`** — full machine-readable build data (all slots, skills, boffs, traits, character info)
- **`.png`** — visual build card image, shareable as a screenshot

If files with that name already exist you are asked to confirm overwrite before both files are written.

> There is no "save format" setting — both files are always written together. To share just the image, send the `.png`. To reload a build in SETS, open the `.json`.

### Open

Click **Open** and select a `.json` or `.png` file. Both formats contain the full build data.

### Skills

Space and Ground skill trees are saved and loaded **separately** from equipment builds:
- Use **Save Skills** / **Open Skills** (available on the Skills tabs)
- Skills are also saved as both `.json` and `.png`

### Autosave

SETS autosaves the current build automatically after every change. The autosave file lives in `library/autosave/` and is loaded automatically on the next launch if no other file was explicitly opened.

---

## 10. Right-click context menu

Right-clicking any filled equipment slot opens a context menu:

| Action | Description |
|--------|-------------|
| **Copy Item** | Copies the item (name, mark, rarity, mods) to an internal clipboard. |
| **Paste Item** | Pastes the copied item into the clicked slot. |
| **Clear Slot** | Removes the item from the slot. |
| **Open Wiki** | Opens the item's STO wiki page in your browser. |
| **Edit Slot** | Opens the Item Editor dialog for detailed editing (name, mark, rarity, modifiers). |

---

## 11. Export

Click **Export** in the menu bar. The Export window generates a formatted build summary in Markdown suitable for posting on STOBuilds Reddit, Discord, or the SETS website.

The output includes ship name, tier, all equipment, boffs, traits, and character info. You can copy the text or save it to a file.
