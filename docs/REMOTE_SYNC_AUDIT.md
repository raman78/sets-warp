# Remote Sync — Capacity Audit

> Snapshot before opening the program to a wider user base. Verifies the
> upload and download paths and estimates how many concurrent users the
> current infrastructure can carry. No code changes were made as part of
> this audit — findings and recommendations are listed at the bottom.

---

## 1. Channels in use

WARP talks to **two separate backends**:

| Channel | Target | Used for | Auth |
|---------|--------|----------|------|
| HF Hub (direct) | `sets-sto/sto-icon-dataset` | Training crops, annotations.jsonl, screen-type screenshots, anchor grids | Shared write token `warp/hub_token.txt` |
| HF Hub (direct, read-only) | `sets-sto/warp-knowledge` | Model + label-map + community_anchors + ship_type_corrections download | Public read (optional token for higher rate limits) |
| Render service | `sets-warp-backend.onrender.com` | `POST /contribute` (pHash knowledge), `GET /knowledge`, `GET /model/version` | Server-side `HF_TOKEN` (never on client) |

---

## 2. Upload path

### 2.1 Crops + annotations + screen-types + anchors (HF Hub direct)

```
WARP CORE confirm
  └─► TrainingDataManager.add_annotation()
         • annotations.json + crop PNG written under warp/training_data/
         • per-sha label cache (.sync_uploaded_labels.json) tracks last-sent
           (slot|name) to skip no-op rewrites
  ▼
SyncManager.check_and_upload()         (registered in warp_button.py)
  • interval:        every 10 min
  • startup delay:   15 s after app launch
  • daily rate cap:  1000 new files per install_id (corrections are free)
  ▼
SyncWorker.run() — single HF commit per cycle, contains:
  • staging/<install_id>/crops/<sha>.png         (only when sha unseen)
  • staging/<install_id>/annotations.jsonl       (last-wins sha dedup)
  • staging/<install_id>/screen_types/<stype>/<sha>.png
  • staging/<install_id>/anchors_grid_<sha8>.json
```

**Validated invariants** (`warp/trainer/sync.py`):

- Crops are stored under `staging/<install_id>/` — no two users can collide.
- `.sync_uploaded_hashes.json` and `.sync_uploaded_labels.json` cache the
  sha→sent and sha→label state locally so `list_repo_files` is only called
  once per install (bootstrap), not on every cycle.
- `_append_staging_annotations_to_ops` performs per-sha last-wins dedup so
  correction events overwrite stale labels instead of accumulating.
- One commit per 10-min cycle bundles all changes — N crops + N annotations
  cost exactly one HF commit, not N.

### 2.2 pHash knowledge (Render proxy)

```
WARP user confirms an unusual icon
  └─► WARPSyncClient.contribute(crop_bgr, item_name, …)
          • non-blocking thread
          • local rate-limit:  200 contributions / install_id / day
          • circuit breaker:   on 503/network error, back off 5 min
  ▼
POST https://sets-warp-backend.onrender.com/contribute
  • 60 s read timeout (covers Render free-tier cold-start ~50 s)
  • backend appends to contributions/YYYY-MM-DD/<uuid>.{png,json}
  • admin merge (manual / scheduled) folds approved contributions into
    knowledge.json
```

---

## 3. Download path

### 3.1 Models (`ModelUpdater`)

```
ModelUpdater().check_and_update()        (called at app launch + WARP CORE open)
  • polling cadence: 15 min  (_CHECK_INTERVAL_HOURS = 0.25)
  • rate-limit cache: warp/models/model_version_remote_cache.json
  • requests with (connect=5 s, read=60 s) timeouts
  ▼
GET https://sets-warp-backend.onrender.com/model/version
  └─► remote = {trained_at, n_classes, val_acc, available}
  ▼
If remote.trained_at > local.trained_at OR embedder stale:
  hf_hub_download from sets-sto/warp-knowledge for each of:
    icon_classifier.pt              (EfficientNet-B0 softmax — required)
    label_map.json                  (required)
    icon_classifier_meta.json
    model_version.json
    screen_classifier.pt            (MobileNetV3-Small)
    screen_classifier_labels.json
    community_anchors.json          (optional, P11)
    ship_type_corrections.json      (optional)
    icon_embedder.pt                (optional, ArcFace)
    embedder_label_map.json         (optional)
    icon_embedder_meta.json         (optional)
    embedding_index.npz             (optional)
  ▼
shutil.copy2 to warp/models/ (only after ALL required files OK)
  ▼
SETSIconMatcher.reset_ml_session()
LayoutDetector.reset_community_anchors_cache()
TextExtractor.load_corrections(...)
```

**Network resilience:**

- Retry schedule: 1 min → 5 min → 15 min → 60 min on network failure.
- Failure does **not** save the rate-limit timestamp → next app launch retries
  immediately instead of waiting 15 min.
- Atomic-ish file install: all temp paths gathered first, only copied to
  `warp/models/` once every required file downloaded successfully.

### 3.2 pHash knowledge (`WARPSyncClient`)

- Polling cadence: re-download after `KNOWLEDGE_MAX_AGE_HOURS = 24`.
- 5 s connect / 15 s read timeout (cache fallback exists).
- Used as override layer in `icon_matcher` before template matching.

---

## 4. Capacity envelope

### 4.1 HF Hub (sets-sto/sto-icon-dataset)

| Constraint | Value (source) | Headroom at 100 users | Headroom at 1000 users |
|------------|----------------|----------------------|------------------------|
| Commits per user token | ~1000/day (HF docs) | 100 × 144 cycles/day × 1 commit ≈ 14 400 commits/day on one shared token | Same shared token would hit the daily ceiling |
| Concurrent commits | Tolerated, no per-second hard limit documented | Fine — 100 users staggered over 10 min = ~10/min | Risky — 1000 users × 1 cycle / 10 min = 100/min if startup not staggered |
| `list_repo_files` size | Grows linearly with crops | 100 × 50 crops = 5 000 files → list ≈ 1–2 s | 1000 × 50 = 50 000 files → list ≈ 10–20 s |
| Storage | Free public datasets, no published cap | Negligible (~64×64 PNG ≈ 4 KB) | ~200 MB at 1000 × 50 crops |

**Key observation — single shared HF token.** All clients use the same
`hub_token.txt` (admin write token). HF's rate limits apply per token, not
per user. The current path therefore scales as N total commits across the
whole user base, not N per user.

### 4.2 Render free-tier backend (`sets-warp-backend`)

| Constraint | Free tier | At 100 active users | At 1000 active users |
|------------|-----------|--------------------|----------------------|
| Cold start | ~50 s after 15 min idle | Handled by 60 s read timeout | Same — once warm, stays warm |
| Concurrent requests | ~100 (default uvicorn workers) | OK | Bottleneck — request queue grows |
| Daily request budget | 750 hours/month CPU time | Fine | Will exhaust within a week if users actively contribute |
| Bandwidth | Generous but not unlimited | OK | Monitor — `/knowledge` is the heaviest endpoint |

**Render free tier comfortably supports ~100 concurrent users.** Beyond
that, the cold-start delay multiplies (queue forms), and CPU-time budget
becomes the constraint.

### 4.3 GitHub Actions training pipeline

- `admin_train.py` runs hourly. Each run reads `staging/*/` from HF and
  trains EfficientNet-B0. Step timeout: 6 h (GH default).
- At 1000 users × 50 crops, training set ≈ 50 000 images — still fits in
  a single Actions run on a `ubuntu-latest` runner.

---

## 5. Risk register

| Risk | Severity | Mitigation today | Recommendation |
|------|----------|------------------|----------------|
| HF token shared across all users — leak compromises every install | High | Token kept out of repo; `hub_token.txt` gitignored | Move uploads through Render proxy (like `/contribute` does today) so the token lives only server-side. Cuts the HF commit fan-out, eliminates client-side leakage. |
| HF rate limit on shared token at >500 active users | Medium | Per-sha + per-label cache keeps cycles ~no-op when nothing changed | Same — Render proxy lets us aggregate uploads server-side and commit in batches. |
| Render cold-start makes the first user of a session wait | Low | 60 s read timeout absorbs it; UI is non-blocking | Upgrade to Render Starter ($7/mo) — no sleep, ~5 s warm restart. |
| All clients start at the same time (e.g. weekend evening) → thundering herd at second 16 | Medium | Fixed 15 s startup delay; per-cycle de-duplication keeps payload small | Add jitter to startup delay (e.g. `15 s + uniform(0, 30 s)`) to spread commits over 30 s window. |
| `list_repo_files` cost grows with crops | Low | Cached locally after first call; subsequent syncs O(1) | None now. Revisit if user base passes 5 k installs. |
| Bad actor floods their own staging/ folder | Low | Per-install staging; client rate limit (1000/day); admin merge step is the gate | None — staging is the right design. |
| User edits a label N times → N HF commits | Low | Label cache (`.sync_uploaded_labels.json`) skips no-op rewrites | Already mitigated. |
| Free public dataset eventually hits HF size limits | Low | None published; HF informally hosts datasets up to ~100 GB | Monitor; archive older `staging/*` after admin merge to keep active size low. |

---

## 6. Bottom line

**For the first round of wider testing (a few dozen to ~100 concurrent
users), the current setup is fit for purpose.** Both upload and download
paths are non-blocking, idempotent, and cached. The retry/back-off logic
covers the Render cold-start. No code changes required to open the program
to a wider beta.

**Before scaling past ~500 active contributors,** two changes earn their
keep:

1. Move client uploads behind the Render proxy (same as `/contribute`
   already does for pHash), so the HF write token leaves the client side
   entirely and the backend can batch commits.
2. Add startup-delay jitter to spread the per-cycle uploads over a 30 s
   window.

Neither is urgent. Both can be done before v3.0 (public release) without
breaking compatibility — clients fall back to direct HF uploads if the
proxy is unreachable.
