# WARP CORE — UI Layout

## Window structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WARP CORE  [Open Folder] [Auto-Detect] [Train Model] [Sync]  [statusbar]  │
├──────────────────┬────────────────────────────┬────────────────────────────┤
│   LEFT PANEL     │      CENTER PANEL           │      RIGHT PANEL           │
│   min 400px      │      min 400px              │      min 400px             │
│                  │                             │                            │
│  Screenshots     │  ┌─────────────────────┐    │  Recognition Review        │
│  ─────────────   │  │                     │    │  ─────────────────────     │
│  □ screen1.png   │  │   SCROLL AREA       │    │  Screen type: SPACE_EQ     │
│  □ screen2.png   │  │  ┌───────────────┐  │    │  ─────────────────────     │
│  □ screen3.png   │  │  │               │  │    │  slot → name [conf%]       │
│  ...             │  │  │    CANVAS     │  │    │  slot → name [conf%]       │
│                  │  │  │  (screenshot) │  │    │  ...                       │
│                  │  │  │               │  │    │                            │
│                  │  │  └───────────────┘  │    │  ─────────────────────     │
│                  │  └─────────────────────┘    │  [+ Add BBox] [- Remove]   │
│  ─────────────   │                             │                            │
│  5/20 annotated  │  ┌─────────────────────┐    │  ─────────────────────     │
│  [progress bar]  │  │   BOTTOM PANEL      │    │  Slot:  [combo box   ]     │
│                  │  │   (Annotate icon)   │    │  Item:  [name field  ]     │
│                  │  │                     │    │  ☐ Auto ≥ [0.75]           │
│                  │  │   Slot: [combo ]    │    │  [    Accept (Enter)  ]    │
│                  │  │   Item: [name  ]    │    │                            │
│                  │  │   ☐Auto≥ [Accept]   │    │                            │
│                  │  └─────────────────────┘    │                            │
└──────────────────┴────────────────────────────┴────────────────────────────┘
```

## Current state (as built)

### LEFT PANEL
- Header: "Screenshots"
- File list (`QListWidget`) — one item per screenshot file, checkbox for screen type confirmation
- Progress label + progress bar at bottom

### CENTER PANEL (vertical stack)
- **Scroll area** containing `AnnotationWidget` (canvas)
  - Displays the current screenshot
  - Zoom: Ctrl+wheel, 1×–6×, zoom-to-cursor
  - Alt+LMB drag: draw new bbox without toggling the Add BBox button
  - Alt held over canvas area: cursor changes to draw crosshair (DRAW_BBOX_COLOR)
  - Confirmed bboxes shown with colour-coded borders (green=confirmed, yellow=pending, grey=skipped)
- **Bottom panel** (`QGroupBox` "Annotate Selected Icon", fixed height 120px)
  - Slot combo box (200px wide)
  - Item name field with autocomplete
  - Accept button (Enter shortcut) + Auto ≥ checkbox + threshold spinner

### RIGHT PANEL
- Header: "Recognition Review"
- Hint text (colour legend)
- Screen type badge
- Review list (`QListWidget`) — one row per detected/annotated item
- Summary label
- Add BBox button (Alt+A) + Remove BBox button (Alt+R / Del)

## Open question

The **Bottom Panel** (Slot / Item name / Accept) currently lives inside the Center Panel, below the canvas. An alternative layout would move it into the Right Panel, below the review list — keeping all annotation controls in one place. This has not been decided yet.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| Enter | Accept current item |
| Del / Backspace | Remove selected bbox (works from canvas or review list) |
| Alt+A | Toggle Add BBox mode |
| Alt+R | Remove selected bbox |
| Alt+LMB drag | Draw new bbox directly on canvas |
| Ctrl+wheel | Zoom canvas in/out (1×–6×) |

## Constants (annotation_widget.py)

| Constant | Value | Controls |
|---|---|---|
| `DRAW_BBOX_COLOR` | `QColor(255, 200, 0)` | Colour of drawn bbox rectangle, fill, and crosshair cursor |
| `DRAW_PEN_WIDTH` | 2 | Pen width for drawn/pending bbox |
| `SELECTED_PEN_WIDTH` | 3 | Pen width for selected bbox |
| `FONT_SIZE_BADGE` | 9 | (Unused — badge text removed, info shown via tooltip) |
