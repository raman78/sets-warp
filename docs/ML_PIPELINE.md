# SETS-WARP ML Pipeline — Technical Reference

This document describes the full lifecycle of machine learning in SETS-WARP:
how models are trained locally, how data flows to the community server, how
the central model is built from contributions, and how updated models are
delivered back to all users.

---

## Overview

SETS-WARP uses two ML classifiers:

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `icon_classifier.pt` | EfficientNet-B0 | Matches item icon crops to item names |
| `screen_classifier.pt` | MobileNetV3-Small | Classifies screenshot type (SPACE_EQ, BOFFS, TRAITS, …) |

Both models exist in two variants — **local** (trained on your data only) and
**community** (trained on all users' data centrally). The local model always
takes priority after you train it; the community model is the baseline before
that.

---

## 1. Local training

### Trigger

You click **Train Model** in WARP CORE, or optionally configure automatic
training after each session.

### Input data

```
warp/training_data/
├── annotations.json          ← confirmed bbox + label records
├── crops/
│   ├── <sha256>.png          ← 64×64 px icon crop per confirmed item
│   └── ...
└── screen_types/
    ├── SPACE_EQ/
    │   └── <filename>.png    ← confirmed full screenshot per type
    └── ...
```

Every time you accept an item in WARP CORE, the system:
1. Crops the bounding box from the screenshot
2. Saves the crop as `warp/training_data/crops/<sha256>.png`
3. Appends the record to `annotations.json` (`slot`, `name`, `bbox`,
   `crop_sha256`, `confirmed_at`)

Screen type labels are saved separately when you change the screen type
dropdown in WARP CORE (stored in `screen_types/<TYPE>/<filename>.png`).

### Training process (icon classifier)

File: `sets-warp-backend/admin_train.py` (runs on GitHub Actions, hourly)

```
1. Download all confirmed crops from HF staging/<install_id>/crops/
2. Deduplicate by sha256 — one vote per install_id per crop hash
3. Stratified train/val split (same as before)
4. Fine-tune EfficientNet-B0:
     - Focal loss (handles class imbalance)
     - AdamW + cosine LR, P7 augmentation (ColorJitter, Flip, Affine)
     - P9 hard negatives: WeightedRandomSampler, cap 3×
     - Skip if < 10 new crops since last run (--skip-if-unchanged)
5. Upload icon_classifier.pt + model_version.json to HF sets-sto/warp-knowledge/models/
```

Local training was removed in v2.3. Crops uploaded by users are the contribution;
the central pipeline trains on all of them and distributes the result.

### Training process (screen classifier)

File: `warp/trainer/screen_type_trainer.py` (central only — not triggered locally)

Uses MobileNetV3-Small and full screenshot images resized to 224×224 px.
Saved to `warp/models/screen_classifier.pt` and distributed via ModelUpdater.

---

## 2. Community upload (local → HuggingFace staging)

### When it happens

WARP CORE runs a background sync timer every 5 minutes. On each tick it
checks for unsynced confirmed crops and uploads them if a HuggingFace token
is present in `warp/hub_token.txt`.

File: `warp/trainer/sync.py` — `SyncWorker`

### Rate limiting

At most **200 uploads per install_id per UTC day** to avoid hammering the HF
API. The counter is stored in memory per session.

### What is uploaded

**Icon crops** → `sets-sto/sto-icon-dataset` (HF Dataset):
```
staging/<install_id>/crops/<sha256>.png   ← icon crop image (64×64 px)
staging/<install_id>/annotations.jsonl    ← one JSON line per crop:
    {"crop_sha256": "...", "name": "Ablative Shell", "slot": "Science Console", ...}
```

**Screen type screenshots** → same repo:
```
staging/<install_id>/screen_types/<TYPE>/<sha256>.png
```

### What is NOT uploaded

- Full screenshots (only the 64×64 icon crop)
- Any personal information
- The local `.pt` model files
- Anything outside `warp/training_data/`

### Install ID

A random UUID generated at first launch, stored in
`warp/knowledge/install_id.txt`. It is anonymous — not linked to any account.
Its only purpose is democratic voting (1 install = 1 vote per crop hash).

### Deduplication

Each crop is identified by `sha256(crop_bytes)`. If a crop with that hash
already exists in staging for this install_id, it is not uploaded again.
This means the same icon appearing in multiple screenshots is uploaded exactly
once per install.

---

## 3. Central training (HuggingFace staging → trained model)

### Trigger

A GitHub Actions workflow runs `admin_train.py` on a schedule:

```yaml
# .github/workflows/train_central_model.yml  (sets-warp-backend repo)
on:
  schedule:
    - cron: '0 * * * *'   # every hour
  workflow_dispatch:       # manual trigger
```

Training is skipped (fast exit) when:
- No new crops since the last training run (`--skip-if-unchanged`)
- Fewer than 10 new crops arrived (`MIN_NEW_CROPS = 10`)

### Democratic voting

```
For each crop sha256:
    votes = {install_id: label for each user who uploaded that crop}
    winner = majority vote (most common label)
    if tie → first uploader's label wins
```

A single user cannot override the community. If User A labels a crop
"Ablative Shell" and three others label it "Ablative Field Projector",
the community label wins.

### Screen type dataset capping

To prevent dataset bloat for stable UI screens:

```
For each screen type class:
    if n_samples >= 30 and n_samples > 150:
        randomly keep 150 samples
        (avoids storing thousands of near-identical UI screenshots)
```

### Central training process (icon classifier)

File: `sets-warp-backend/admin_train.py` — `train()`

```
1. Download previous icon_classifier.pt from HF warp-knowledge (for fine-tuning)
2. Download all winning crops from staging via snapshot_download per install_id
   (bulk folder download — far fewer HTTP round-trips than per-file)
3. Stratified train/val split (same logic as local trainer)
4. Build EfficientNet-B0, replace head for n_classes
5. Load previous backbone weights (classifier.* keys stripped)
   LR = 3e-4 × 0.3 when fine-tuning, 3e-4 from scratch
6. Train with focal loss + cosine annealing + early stopping
7. Save: icon_classifier.pt, label_map.json, icon_classifier_meta.json,
         model_version.json, training_manifest.json
8. Upload all files to sets-sto/warp-knowledge/models/
```

`training_manifest.json` records the set of crop SHAs used in this run —
the next run's skip-if-unchanged check compares against this.

### Central training process (screen classifier)

File: `sets-warp-backend/admin_train.py` — `train_screen_classifier()`

Same flow with MobileNetV3-Small. Only runs if ≥ 7 screen type screenshots
are available. Fine-tunes from previous `screen_classifier.pt` backbone.

### model_version.json

Published after each successful central training run:

```json
{
  "version": "<git sha>",
  "trained_at": "2026-03-23T14:00:00Z",
  "n_classes": 2933,
  "val_acc": 0.87,
  "n_samples": 12450,
  "n_users": 18,
  "screen_trained_at": "2026-03-23T14:00:00Z",
  "screen_val_acc": 0.95,
  "screen_n_samples": 340
}
```

---

## 4. Community model delivery (HuggingFace → local)

### On first install (bootstrap)

File: `bootstrap.py` — `_download_community_model_bootstrap()`

During the first-run setup wizard (Step 5/5), the following files are
downloaded from `sets-sto/warp-knowledge`:

```
models/icon_classifier.pt
models/label_map.json
models/icon_classifier_meta.json
models/model_version.json
models/screen_classifier.pt
models/screen_classifier_labels.json
```

A `model_version_remote_cache.json` is written after download so that
`ModelUpdater` skips the next 24 h check (just downloaded, no need to check
again immediately).

### Ongoing updates (ModelUpdater)

File: `warp/trainer/model_updater.py`

Fired **15 seconds after app launch** (background daemon thread), and also
each time WARP CORE is opened.

```
1. Check rate limit: skip if last check was < 24 h ago
2. GET https://sets-warp-backend.onrender.com/model/version
   → returns {available, trained_at, n_classes, val_acc, ...}
3. Compare remote trained_at vs local model_version.json trained_at:
     remote > local  → download and install new model
     remote ≤ local  → skip (local model is current or user-trained)
4. If update needed: download all _MODEL_FILES from HF via hf_hub_download
5. Copy files to warp/models/ atomically
6. Call SETSIconMatcher.reset_ml_session() to reload immediately
7. Save check timestamp to model_version_remote_cache.json
```

**Local model always wins:** after you run Train Model locally, the local
`model_version.json` gets `trained_at = now()`. Since `now() > any past
remote trained_at`, ModelUpdater will never overwrite your local model until
the central model is trained again after your training time.

### screen_classifier fallback

If `screen_classifier.pt` is missing (e.g., first install before bootstrap
completes, or manual deletion), `_ensure_screen_classifier()` downloads it
immediately, bypassing the 24 h rate limit. This runs on every ModelUpdater
check.

---

## 5. Full pipeline diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER (WARP CORE)                                                   │
│                                                                     │
│  Confirm bbox → crop PNG + label saved locally                      │
│       │                                                             │
│       ▼                                                             │
│  warp/training_data/crops/<sha>.png                                 │
│  warp/training_data/annotations.json                                │
│       │                         │                                   │
│       │  [Train Model]          │  [sync timer, every 5 min]        │
│       ▼                         ▼                                   │
│  local icon_classifier.pt    HF staging (sets-sto/sto-icon-dataset) │
│  (used immediately)          staging/<install_id>/crops/<sha>.png   │
│       │                      staging/<install_id>/annotations.jsonl │
└───────┼──────────────────────────────┼──────────────────────────────┘
        │                              │
        │                              ▼
        │             ┌────────────────────────────────┐
        │             │  GitHub Actions (hourly cron)  │
        │             │  admin_train.py                │
        │             │                                │
        │             │  1. Democratic voting          │
        │             │  2. Screen type capping        │
        │             │  3. Download crops (bulk)      │
        │             │  4. Fine-tune EfficientNet-B0  │
        │             │  5. Fine-tune MobileNetV3-Small│
        │             │  6. Upload to warp-knowledge   │
        │             └───────────────┬────────────────┘
        │                             │
        │                             ▼
        │             HF Dataset: sets-sto/warp-knowledge
        │             models/icon_classifier.pt
        │             models/screen_classifier.pt
        │             models/model_version.json
        │             models/label_map.json
        │                             │
        │                             │  [ModelUpdater, 15s after launch,
        │                             │   at most once per 24 h]
        │                             ▼
        └──────────► warp/models/icon_classifier.pt   ◄── community model
                                                           (used until user
                                                            trains locally)
```

---

## 6. Priority rules

When WARP matches an icon, the following priority applies:

```
1. Community pHash knowledge override   (warp/knowledge/knowledge_cache.json)
   — exact perceptual hash match, highest confidence, instant
2. Template matching + HSV histogram    (SETS icon cache, wiki icons)
3. Local ML classifier                  (icon_classifier.pt)
4. Session examples                     (confirmed crops from current session,
                                         used only when phases 1-3 fail)
```

The local `.pt` and community `.pt` are the **same file** —
`warp/models/icon_classifier.pt`. It is either:
- The community model (downloaded from HF, used on fresh install)
- Your locally-trained model (overwrites community after Train Model)
- A community update (overwrites local only if `remote trained_at > local trained_at`)

---

## 7. Data stored on HuggingFace

| Repo | Path | Contents |
|------|------|----------|
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/crops/` | Icon crop PNGs (64×64) |
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/annotations.jsonl` | Label records |
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/screen_types/<TYPE>/` | Screen type PNGs |
| `sets-sto/warp-knowledge` | `models/` | Trained .pt files, label_map.json, model_version.json |
| `sets-sto/warp-knowledge` | `knowledge.json` | pHash → item_name community overrides |
| `sets-sto/warp-knowledge` | `models/training_manifest.json` | SHA set from last training run |

---

## 8. Key files (local)

| File | Purpose |
|------|---------|
| `warp/models/icon_classifier.pt` | Active icon classifier (local or community) |
| `warp/models/screen_classifier.pt` | Active screen type classifier |
| `warp/models/label_map.json` | `{index: item_name}` for icon classifier |
| `warp/models/screen_classifier_labels.json` | `{index: screen_type}` for screen classifier |
| `warp/models/model_version.json` | `trained_at`, `source`, `val_acc` of current model |
| `warp/models/model_version_remote_cache.json` | Timestamp of last remote version check |
| `warp/training_data/annotations.json` | All confirmed annotations |
| `warp/training_data/crops/<sha>.png` | Confirmed icon crops |
| `warp/training_data/screen_types/<TYPE>/<file>.png` | Confirmed screen type screenshots |
| `warp/knowledge/install_id.txt` | Anonymous UUID identifying this installation |
| `warp/hub_token.txt` | HuggingFace write token (optional, for uploads) |
