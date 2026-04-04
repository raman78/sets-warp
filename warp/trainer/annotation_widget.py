# warp/trainer/annotation_widget.py
# Interactive canvas for annotating STO screenshots.

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QWidget, QSizePolicy, QScrollArea
from PySide6.QtCore    import Qt, QRect, QPoint, QRectF, Signal, QSize
from PySide6.QtGui     import (
    QPainter, QPixmap, QColor, QPen, QBrush, QFont,
    QMouseEvent, QPaintEvent, QKeyEvent, QCursor
)

from warp.trainer.training_data import TrainingDataManager, Annotation, AnnotationState


# Colour scheme for annotation states
STATE_COLORS = {
    AnnotationState.PENDING:   QColor(255, 200,   0, 180),   # yellow
    AnnotationState.CONFIRMED: QColor( 60, 220, 100, 200),   # green
    AnnotationState.SKIPPED:   QColor(160, 160, 160, 120),   # grey
}

DRAW_PEN_WIDTH     = 2
SELECTED_PEN_WIDTH = 3
FONT_SIZE_BADGE    = 9

# Colour of the bbox being drawn (Add BBox / Alt+LMB).
DRAW_BBOX_COLOR = QColor(255, 200, 0)   # yellow — matches Add BBox button style

# Color for manipulation tools (Move/Resize)
MANIP_COLOR = QColor(100, 200, 255)  # Action blue


class AnnotationWidget(QWidget):
    """
    Screenshot viewer with interactive bbox annotation overlay.

    Signals:
        annotation_added(bbox: tuple)     — user finished drawing a new bbox
        item_selected(annotation: dict)   — user clicked an existing annotation
    """

    annotation_added = Signal(tuple)    # (x, y, w, h) in image coords
    item_selected    = Signal(dict)     # annotation dict
    item_deselected  = Signal()         # user clicked empty area
    bbox_changed     = Signal(int, tuple)  # (row, new_bbox) — Shift+LMB move/resize

    def __init__(self, data_manager: TrainingDataManager, parent=None):
        super().__init__(parent)
        self._data_mgr    = data_manager
        self._pixmap:   QPixmap | None = None
        self._img_path: Path | None    = None
        self._scale:    float          = 1.0
        self._offset_x: int            = 0
        self._offset_y: int            = 0
        self._user_scale: 'float | None' = None  # None = fit-to-window
        self._fit_scale: float          = 1.0   # computed once at load, stable
        self._zoom:     float          = 1.0   # 1.0–6.0
        self._zoom_ox:  float          = 0.0
        self._zoom_oy:  float          = 0.0
        self._mod_cursor_active: bool  = False  # True when setOverrideCursor is active

        # Mode flags — drawing/editing only active when explicitly enabled
        self._draw_mode_forced: bool = False   # set by + Add BBox / Edit BBox
        self._alt_draw: bool = False             # True when drawing via Alt+LMB
        self._locked: bool = False               # True when screenshot is marked Done

        # Drawing state
        self._drawing       = False
        self._draw_start:   QPoint | None = None
        self._draw_current: QPoint | None = None

        # Selection
        self._selected_idx: int = -1

        # All annotations for current image
        self._annotations: list[Annotation] = []

        # Pending new bbox (drawn but not yet confirmed)
        self._pending_bbox: tuple | None = None

        # Review items from trainer_window (replaces _annotations for drawing)
        # Each dict: {bbox, state, name, slot}
        self._review_items: list[dict] = []
        self._selected_row: int = -1      # row for full edit mode (with handles)
        self._highlighted_row: int = -1   # row for simple highlight (red dotted box)
        # Hover tooltip state
        self._hover_row:   int   = -1
        self._hover_timer: object = None

        # Drag/resize state — _annotations (legacy draw mode)
        self._drag_mode:  str | None = None   # 'move' | 'resize_NW' | etc.
        self._drag_start: QPoint | None = None
        self._drag_orig:  tuple | None = None  # original bbox at drag start
        # Drag/resize state — _review_items (Shift+LMB)
        self._drag_review_row: int = -1       # which review item is being dragged
        # Handle size in screen pixels
        self._HANDLE = 9

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #1a1a1a;")

    # ---------------------------------------------------------------- public API

    def load_image(self, path: Path):
        """Load a screenshot and its existing annotations."""
        self._img_path    = path
        self._pixmap      = QPixmap(str(path))
        self._annotations = self._data_mgr.get_annotations(path)
        self._selected_idx  = -1
        self._pending_bbox  = None
        self._drawing       = False
        self._highlighted_row = -1
        self._selected_row = -1
        self._zoom = 1.0
        self._zoom_ox = 0.0
        self._zoom_oy = 0.0
        self._user_scale  = None   # reset to fit-to-window on every image load
        self._compute_transform()
        self.adjustSize()
        self.setFocus()
        self.update()

    def refresh_annotations(self, path: Path):
        """Reload confirmed annotations from data manager (call after add_annotation)."""
        self._annotations = self._data_mgr.get_annotations(path)
        self.update()

    def confirm_current(self, slot: str, name: str):
        if self._pending_bbox is not None:
            self._data_mgr.add_annotation(image_path=self._img_path, bbox=self._pending_bbox, slot=slot, name=name, state=AnnotationState.CONFIRMED)
            self._annotations = self._data_mgr.get_annotations(self._img_path)
            self._pending_bbox = None
            self._selected_idx = len(self._annotations) - 1
        elif self._selected_idx >= 0:
            ann = self._annotations[self._selected_idx]
            ann.slot = slot; ann.name = name; ann.state = AnnotationState.CONFIRMED
            self._data_mgr.update_annotation(self._img_path, ann)
            self._annotations = self._data_mgr.get_annotations(self._img_path)
        self.update()

    def skip_current(self):
        if self._selected_idx >= 0:
            ann = self._annotations[self._selected_idx]
            ann.state = AnnotationState.SKIPPED
            self._data_mgr.update_annotation(self._img_path, ann)
        self._pending_bbox = None; self._selected_idx = -1; self.update()

    def all_confirmed(self) -> bool:
        if not self._annotations: return False
        return all(a.state in (AnnotationState.CONFIRMED, AnnotationState.SKIPPED) for a in self._annotations)

    def clear_highlight(self):
        self._highlighted_row = -1; self.update()

    def clear_pending(self):
        self._pending_bbox = None; self._drawing = False; self._draw_start = None; self._draw_current = None; self.update()

    def set_review_items(self, items: list[dict]):
        self._review_items = items; self.update()

    def set_selected_row(self, row: int):
        self._selected_row = row; self._highlighted_row = -1; self.update()

    def set_highlighted_row(self, row: int):
        self._highlighted_row = row; self.update()

    def set_draw_mode(self, enabled: bool):
        self._draw_mode_forced = enabled
        if not enabled: self._drawing = False; self._draw_start = None; self._draw_current = None
        self.update()

    def set_locked(self, locked: bool):
        """Lock/unlock drawing — used when screenshot is marked Done."""
        self._locked = locked
        if locked:
            self._draw_mode_forced = False
            self._drawing = False
            self._draw_start = None
            self._draw_current = None
            self._clear_mod_cursor()
        self.update()

    # ---------------------------------------------------------------- painting

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._pixmap:
            zw = int(self._pixmap.width()  * self._scale)
            zh = int(self._pixmap.height() * self._scale)
            painter.drawPixmap(self._offset_x, self._offset_y, zw, zh, self._pixmap)
        else:
            painter.fillRect(self.rect(), QColor("#1a1a1a")); painter.setPen(QColor("#888888")); painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded\nOpen a folder to start")
            return

        # Z-ORDER DRAWING:
        # 1. Background (unselected) items
        for idx, ri in enumerate(self._review_items):
            if idx == self._selected_row or idx == self._highlighted_row: continue
            self._draw_review_item(painter, ri.get('bbox'), ri.get('state'), ri.get('name',''), ri.get('slot',''), False, False)

        # 2. Highlighted item (Red Dashed)
        if self._highlighted_row != -1 and self._highlighted_row < len(self._review_items) and self._highlighted_row != self._selected_row:
            ri = self._review_items[self._highlighted_row]
            self._draw_review_item(painter, ri.get('bbox'), ri.get('state'), ri.get('name',''), ri.get('slot',''), False, True)

        # 3. Selected item (Full Edit with handles)
        if self._selected_row != -1 and self._selected_row < len(self._review_items):
            ri = self._review_items[self._selected_row]
            self._draw_review_item(painter, ri.get('bbox'), ri.get('state'), ri.get('name',''), ri.get('slot',''), True, False)

        # In-progress drawing (while dragging)
        if self._drawing and self._draw_start and self._draw_current:
            pen = QPen(DRAW_BBOX_COLOR, DRAW_PEN_WIDTH, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(DRAW_BBOX_COLOR.red(), DRAW_BBOX_COLOR.green(), DRAW_BBOX_COLOR.blue(), 30)))
            rect = QRect(self._draw_start, self._draw_current).normalized()
            painter.drawRect(rect)

    _STATE_COLOR = {
        'pending':   QColor(220,  80,  80, 220),
        'confirmed': QColor( 60, 220, 100, 220),
        'new':       QColor(220,  80,  80, 220),
    }
    # Text/fixed-value slots (Ship Name/Type/Tier) use cyan — visually distinct
    # from icon slots; signals "bbox saved for layout learning, no ML crop"
    _TEXT_SLOT_COLOR = QColor(0, 200, 220, 220)

    def _draw_review_item(self, painter: QPainter, bbox: tuple, state: str, name: str, slot: str, selected: bool, highlighted: bool):
        if not bbox: return
        try:
            from warp.trainer.training_data import NON_ICON_SLOTS
            is_text_slot = slot in NON_ICON_SLOTS
        except Exception:
            is_text_slot = False
        base_color = self._TEXT_SLOT_COLOR if is_text_slot else self._STATE_COLOR.get(state, QColor(200, 200, 200, 180))
        if highlighted and not selected:
            color = base_color; pw = SELECTED_PEN_WIDTH + 1; style = Qt.PenStyle.DashLine
        elif selected:
            color = base_color; pw = SELECTED_PEN_WIDTH; style = Qt.PenStyle.DashLine
        else:
            color = base_color; pw = DRAW_PEN_WIDTH; style = Qt.PenStyle.SolidLine
        pen = QPen(color, pw, style); painter.setPen(pen); painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 25))); rect = self._img_to_screen_rect(bbox); painter.drawRect(rect)
        if selected:
            h = self._HANDLE
            painter.setPen(QPen(color, 1))
            painter.setBrush(QBrush(color))
            for hx, hy in self._handle_positions(rect):
                painter.drawRect(QRect(hx - h // 2, hy - h // 2, h, h))

    # ---------------------------------------------------------------- mouse events

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton: return
        if self._locked: return
        pos = event.pos()
        # Alt+LMB drag — start drawing a new bbox without toggling Add BBox button
        alt_held   = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        # Shift+LMB — move or resize ANY existing review item via handles
        if shift_held and not alt_held:
            handle, target_row = self._handle_hit_test_all_reviews(pos)
            if handle is None:
                # If no handle hit, check if we hit the body of any bbox
                target_row = self._hit_test_review(pos)
                if target_row >= 0:
                    handle = 'move'
            
            if handle and target_row >= 0:
                self._drag_review_row = target_row
                self._drag_mode       = handle
                self._drag_start      = pos
                self._drag_orig       = self._review_items[target_row].get('bbox')
                self.setCursor(self._cursor_for_handle(handle))
                # Sync selection to the item being manipulated
                if target_row != self._selected_row:
                    ri = self._review_items[target_row]
                    self.item_selected.emit({'slot': ri.get('slot', ''), 'name': ri.get('name', ''), 'bbox': ri.get('bbox')})
                self.update()
                return

        if alt_held:
            self._drawing = True
            self._draw_start = pos
            self._draw_current = pos
            self._selected_idx = -1
            self._alt_draw = True  # flag: emitted via alt, not button
            self.setCursor(self._make_draw_cursor())
            self.update()
            return
        self._alt_draw = False
        if self._draw_mode_forced:
            if self._selected_idx >= 0:
                handle = self._handle_hit_test(pos, self._selected_idx)
                if handle:
                    self._drag_mode = handle; self._drag_start = pos; self._drag_orig = self._annotations[self._selected_idx].bbox; self.setCursor(self._cursor_for_handle(handle)); self.update(); return
            self._drawing = True
            self._draw_start = pos
            self._draw_current = pos
            self._selected_idx = -1
            self.setCursor(self._make_draw_cursor())
            self.update()
            return
        if self._selected_idx >= 0:
            handle = self._handle_hit_test(pos, self._selected_idx)
            if handle: self._drag_mode = handle; self._drag_start = pos; self._drag_orig = self._annotations[self._selected_idx].bbox; self.setCursor(self._cursor_for_handle(handle)); self.update(); return
        clicked = self._hit_test(pos)
        if clicked >= 0:
            self._selected_idx = clicked; self._pending_bbox = None; ann = self._annotations[clicked]
            self.item_selected.emit({'slot': ann.slot, 'name': ann.name, 'bbox': ann.bbox})
        else:
            row = self._hit_test_review(pos)
            if row >= 0:
                ri = self._review_items[row]
                self.item_selected.emit({'slot': ri.get('slot', ''), 'name': ri.get('name', ''), 'bbox': ri.get('bbox')})
            else:
                self._selected_idx = -1
                self.item_deselected.emit()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.pos()
        if self._drawing: self._draw_current = pos; self.update(); return

        # Review item drag (Shift+LMB) — takes priority over annotation drag
        if self._drag_review_row >= 0 and self._drag_mode and self._drag_start and self._drag_orig:
            dx = int((pos.x() - self._drag_start.x()) / self._scale)
            dy = int((pos.y() - self._drag_start.y()) / self._scale)
            ox, oy, ow, oh = self._drag_orig
            m = self._drag_mode
            if   m == 'move':      nx, ny, nw, nh = ox+dx, oy+dy, ow,    oh
            elif m == 'resize_NW': nx, ny, nw, nh = ox+dx, oy+dy, ow-dx, oh-dy
            elif m == 'resize_NE': nx, ny, nw, nh = ox,    oy+dy, ow+dx, oh-dy
            elif m == 'resize_SW': nx, ny, nw, nh = ox+dx, oy,    ow-dx, oh+dy
            elif m == 'resize_SE': nx, ny, nw, nh = ox,    oy,    ow+dx, oh+dy
            elif m == 'resize_N':  nx, ny, nw, nh = ox,    oy+dy, ow,    oh-dy
            elif m == 'resize_S':  nx, ny, nw, nh = ox,    oy,    ow,    oh+dy
            elif m == 'resize_W':  nx, ny, nw, nh = ox+dx, oy,    ow-dx, oh
            elif m == 'resize_E':  nx, ny, nw, nh = ox,    oy,    ow+dx, oh
            else:                  nx, ny, nw, nh = ox, oy, ow, oh
            if nw > 8 and nh > 8:
                self._review_items[self._drag_review_row]['bbox'] = (nx, ny, nw, nh)
            self.update()
            return

        if self._drag_mode and self._drag_start and self._drag_orig:
            dx = int((pos.x() - self._drag_start.x()) / self._scale); dy = int((pos.y() - self._drag_start.y()) / self._scale); ox, oy, ow, oh = self._drag_orig; m = self._drag_mode
            if m == 'move': nx, ny, nw, nh = ox + dx, oy + dy, ow, oh
            elif m == 'resize_NW': nx, ny, nw, nh = ox+dx, oy+dy, ow-dx, oh-dy
            elif m == 'resize_NE': nx, ny, nw, nh = ox,    oy+dy, ow+dx, oh-dy
            elif m == 'resize_SW': nx, ny, nw, nh = ox+dx, oy,    ow-dx, oh+dy
            elif m == 'resize_SE': nx, ny, nw, nh = ox,    oy,    ow+dx, oh+dy
            elif m == 'resize_N':  nx, ny, nw, nh = ox,    oy+dy, ow,    oh-dy
            elif m == 'resize_S':  nx, ny, nw, nh = ox,    oy,    ow,    oh+dy
            elif m == 'resize_W':  nx, ny, nw, nh = ox+dx, oy,    ow-dx, oh
            elif m == 'resize_E':  nx, ny, nw, nh = ox,    oy,    ow+dx, oh
            else: nx, ny, nw, nh = ox, oy, ow, oh
            if nw > 8 and nh > 8:
                ann = self._annotations[self._selected_idx]; self._data_mgr.update_annotation(self._img_path, ann, bbox=(nx, ny, nw, nh)); self._annotations = self._data_mgr.get_annotations(self._img_path)
            self.update(); return
        from PySide6.QtWidgets import QApplication as _QApp
        mods = _QApp.queryKeyboardModifiers()

        # 1. Modifiers have highest priority for cursor shape (e.g. forced draw/zoom mode)
        if mods & Qt.KeyboardModifier.AltModifier:
            self.setCursor(self._make_draw_cursor())
            self._cancel_hover_timer()
            from PySide6.QtWidgets import QToolTip
            QToolTip.hideText()
            return

        if mods & Qt.KeyboardModifier.ControlModifier:
            self.setCursor(self._make_zoom_cursor())
            self._cancel_hover_timer()
            return

        # 2. Check for handle/bbox hover ONLY IF Shift is held (Edit Mode)
        if mods & Qt.KeyboardModifier.ShiftModifier:
            handle, target_row = self._handle_hit_test_all_reviews(pos)
            if handle:
                self._set_mod_cursor(self._cursor_for_handle(handle))
                self._cancel_hover_timer()
                from PySide6.QtWidgets import QToolTip
                QToolTip.hideText()
                return

            hit_row = self._hit_test_review(pos)
            if hit_row >= 0:
                self._set_mod_cursor(self._make_edit_cursor())
                self._cancel_hover_timer()
                from PySide6.QtWidgets import QToolTip
                QToolTip.hideText()
                return

        # 3. Regular hover tooltip logic (no special cursor)
        hovered = self._hit_test_review(pos)
        if hovered != self._hover_row:
            self._hover_row = hovered
            self._cancel_hover_timer()
            if hovered >= 0:
                self._start_hover_timer(hovered)
            else:
                from PySide6.QtWidgets import QToolTip
                QToolTip.hideText()

        # Restore appropriate cursor if not over anything special
        if hovered < 0:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                self._set_mod_cursor(self._make_edit_cursor())
            else:
                self.unsetCursor()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton: return
        if self._drawing:
            self._drawing = False
            if self._draw_start and self._draw_current:
                screen_rect = QRect(self._draw_start, self._draw_current).normalized()
                if screen_rect.width() > 8 and screen_rect.height() > 8:
                    self._pending_bbox = self._screen_to_img_rect(screen_rect)
                    self.annotation_added.emit(self._pending_bbox)
            self._draw_start = None
            self._draw_current = None
            if getattr(self, '_alt_draw', False):
                self._alt_draw = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
        if self._drag_review_row >= 0:
            row      = self._drag_review_row
            new_bbox = self._review_items[row].get('bbox') if row < len(self._review_items) else None
            if new_bbox:
                self.bbox_changed.emit(row, new_bbox)
            self._drag_review_row = -1
            self._drag_mode  = None
            self._drag_start = None
            self._drag_orig  = None
            self.unsetCursor()
            self.update()
            return
        if self._drag_mode:
            self._drag_mode = None; self._drag_start = None
            self._drag_orig = None; self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete and self._selected_idx >= 0:
            ann = self._annotations[self._selected_idx]; self._data_mgr.remove_annotation(self._img_path, ann); self._annotations = self._data_mgr.get_annotations(self._img_path); self._selected_idx = -1; self.update()

    def _cancel_hover_timer(self):
        if self._hover_timer is not None:
            self._hover_timer.stop()
            self._hover_timer = None

    def _start_hover_timer(self, row: int):
        from PySide6.QtCore import QTimer
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(lambda: self._show_hover_tooltip(row))
        self._hover_timer.start(700)

    def _show_hover_tooltip(self, row: int):
        from PySide6.QtWidgets import QApplication as _QApp
        if row < 0 or row >= len(self._review_items) or (_QApp.queryKeyboardModifiers() & Qt.KeyboardModifier.ShiftModifier):
            from PySide6.QtWidgets import QToolTip
            QToolTip.hideText()
            return
        ri        = self._review_items[row]
        name      = ri.get('name') or '— unmatched —'
        slot      = ri.get('slot', '?')
        conf      = ri.get('conf', 0.0)
        state     = ri.get('state', 'pending')
        orig_name = ri.get('orig_name', '')
        if state == 'confirmed':
            lines = [f'<b>{slot}</b>', name, '<i>confirmed by user</i>']
            if conf > 0.0:
                color = ('#7effc8' if conf >= 0.85 else
                         '#e8c060' if conf >= 0.70 else '#ff9966')
                ml_text = orig_name if orig_name and orig_name != name else name
                lines.append(f'ML: <span style="color:{color}">{ml_text} ({conf:.1%})</span>')
            else:
                lines.append('<span style="color:#888">ML: unknown (previous session)</span>')
            text = '<br>'.join(lines)
        else:
            pct   = f'{conf:.1%}'
            color = ('#7effc8' if conf >= 0.85 else
                     '#e8c060' if conf >= 0.70 else '#ff9966')
            text  = (f'<b>{slot}</b><br>{name}'
                     f'<br>Confidence: <span style="color:{color}">{pct}</span>')
        from PySide6.QtWidgets import QToolTip
        from PySide6.QtGui import QCursor
        QToolTip.showText(QCursor.pos(), text, self)

    def _handle_positions(self, rect: QRect) -> list[tuple[int, int]]:
        l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom(); mx, my = (l + r) // 2, (t + b) // 2
        return [(l, t), (mx, t), (r, t), (l, my), (r, my), (l, b), (mx, b), (r, b), (mx, my)]

    def _handle_hit_test_review(self, pos: QPoint, row: int) -> str | None:
        """Hit-test handles on a review item bbox. Returns handle name or None."""
        if row < 0 or row >= len(self._review_items): return None
        bbox = self._review_items[row].get('bbox')
        if not bbox: return None
        rect = self._img_to_screen_rect(bbox)
        h = self._HANDLE + 2
        l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
        mx, my = (l + r) // 2, (t + b) // 2
        handles = [
            ('resize_NW', l, t), ('resize_N', mx, t), ('resize_NE', r, t),
            ('resize_W',  l, my),                     ('resize_E',  r, my),
            ('resize_SW', l, b), ('resize_S', mx, b), ('resize_SE', r, b),
        ]
        x, y = pos.x(), pos.y()
        for name, hx, hy in handles:
            if abs(x - hx) <= h and abs(y - hy) <= h: return name
        return None

    def _handle_hit_test_all_reviews(self, pos: QPoint) -> tuple[str | None, int]:
        """Hit-test handles for ALL review items. Returns (handle_name, row_index)."""
        # Prioritize selected row if it's under mouse
        if self._selected_row >= 0:
            h = self._handle_hit_test_review(pos, self._selected_row)
            if h: return h, self._selected_row
            
        # Then check others (top-most in list first)
        for idx in range(len(self._review_items)):
            if idx == self._selected_row: continue
            h = self._handle_hit_test_review(pos, idx)
            if h: return h, idx
        return None, -1

    def _handle_hit_test(self, pos: QPoint, ann_idx: int) -> str | None:
        if ann_idx < 0 or ann_idx >= len(self._annotations): return None
        try:
            from warp.trainer.training_data import NON_ICON_SLOTS
            if self._annotations[ann_idx].slot in NON_ICON_SLOTS: return None
        except Exception:
            pass
        rect = self._img_to_screen_rect(self._annotations[ann_idx].bbox); h = self._HANDLE + 2; l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom(); mx, my = (l + r) // 2, (t + b) // 2
        handles = [('resize_NW', l, t), ('resize_N', mx, t), ('resize_NE', r, t), ('resize_W', l, my), ('resize_E', r, my), ('resize_SW', l, b), ('resize_S', mx, b), ('resize_SE', r, b), ('move', mx, my)]
        x, y = pos.x(), pos.y()
        for name, hx, hy in handles:
            if abs(x - hx) <= h and abs(y - hy) <= h: return name
        return None

    @staticmethod
    def _make_draw_cursor() -> QCursor:
        """Create a crosshair cursor coloured with DRAW_BBOX_COLOR."""
        size = 12
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        p.setPen(QPen(DRAW_BBOX_COLOR, 2))
        cx = size // 2
        p.drawLine(cx, 0, cx, size - 1)   # vertical
        p.drawLine(0, cx, size - 1, cx)   # horizontal
        p.end()
        return QCursor(px, cx, cx)

    @staticmethod
    def _make_zoom_cursor() -> QCursor:
        """Magnifying glass cursor shown while Ctrl is held."""
        size = 28
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # White outer ring
        p.setPen(QPen(QColor(255, 255, 255, 220), 2))
        p.setBrush(QBrush(QColor(100, 200, 255, 60)))
        p.drawEllipse(2, 2, 16, 16)
        # Handle
        p.drawLine(16, 16, 24, 24)
        p.end()
        return QCursor(px, 10, 10)

    @staticmethod
    def _make_edit_cursor() -> Qt.CursorShape:
        """Use system move cursor for 'Edit Mode' (Shift) as requested."""
        return Qt.CursorShape.SizeAllCursor

    def _cursor_for_handle(self, handle: str) -> QCursor | Qt.CursorShape:
        # Use system resize cursors for precision as requested
        if handle == 'move': return self._make_edit_cursor()
        if handle in ('resize_NW', 'resize_SE'): return Qt.CursorShape.SizeFDiagCursor
        if handle in ('resize_NE', 'resize_SW'): return Qt.CursorShape.SizeBDiagCursor
        if handle in ('resize_N', 'resize_S'):   return Qt.CursorShape.SizeVerCursor
        if handle in ('resize_W', 'resize_E'):   return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.ArrowCursor

    def wheelEvent(self, event):
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl:
            if not self._pixmap: return
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            # Use _fit_scale computed at load time — stable, doesn't grow with widget
            fit_s = self._fit_scale
            old_s = self._scale
            new_s = max(fit_s, min(fit_s * 6.0, old_s * factor))
            if abs(new_s - old_s) < 0.0001: return
            # Find scroll area in parent chain
            from PySide6.QtWidgets import QAbstractScrollArea
            from PySide6.QtGui import QCursor
            sa = self.parent()
            while sa and not isinstance(sa, QAbstractScrollArea):
                sa = sa.parent()

            pw, ph = self._pixmap.width(), self._pixmap.height()
            lpos = self.mapFromGlobal(QCursor.pos())
            img_x, img_y = self._screen_to_img(lpos.x(), lpos.y())
            
            if sa:
                vp = sa.viewport()
                vp_pos = vp.mapFromGlobal(QCursor.pos())
                vp_cx, vp_cy = float(vp_pos.x()), float(vp_pos.y())
                # If cursor is outside image, zoom towards image center
                if not (0 <= img_x <= pw and 0 <= img_y <= ph):
                    img_x, img_y = pw / 2.0, ph / 2.0
                    vp_cx, vp_cy = vp.width() / 2.0, vp.height() / 2.0
            else:
                img_x, img_y = pw / 2.0, ph / 2.0
                vp_cx, vp_cy = self.width() / 2.0, self.height() / 2.0
                sa = None

            if new_s <= fit_s * 1.001:
                self._user_scale = None
            else:
                self._user_scale = new_s
            self._compute_transform()
            self.adjustSize()
            # Adjust scrollbars to keep image point under cursor
            if sa and self._user_scale is not None:
                sa.horizontalScrollBar().setValue(int(img_x * new_s - vp_cx))
                sa.verticalScrollBar().setValue(int(img_y * new_s - vp_cy))
            self.update()
            event.accept()
        else:
            super().wheelEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        from PySide6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)
        if self.parent():
            self.parent().installEventFilter(self)

    def hideEvent(self, event):
        super().hideEvent(event)
        from PySide6.QtWidgets import QApplication
        QApplication.instance().removeEventFilter(self)
        if self.parent():
            self.parent().removeEventFilter(self)

    def enterEvent(self, event):
        """Mouse entered canvas area — grab focus and show context cursor."""
        from PySide6.QtWidgets import QApplication, QLineEdit, QTextEdit, QAbstractSpinBox
        focused = QApplication.focusWidget()
        if not isinstance(focused, (QLineEdit, QTextEdit, QAbstractSpinBox)):
            self.setFocus()
        mods = QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.AltModifier:
            self._set_mod_cursor(self._make_draw_cursor())
        elif mods & Qt.KeyboardModifier.ControlModifier:
            self._set_mod_cursor(self._make_zoom_cursor())
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            self._set_mod_cursor(self._make_edit_cursor())
        else:
            self._clear_mod_cursor()  # clean up if modifier was released while mouse was outside

    def leaveEvent(self, event):
        """Mouse left canvas area — clear mod cursor if no active drag."""
        if self._drawing: return
        self._clear_mod_cursor()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        etype = event.type()
        # Viewport resize → recompute fit scale and resize widget
        if etype == QEvent.Type.Resize and obj is self.parent():
            if self._user_scale is None:
                self._compute_transform()
                self.adjustSize()
                self.update()
            return False
        if etype in (QEvent.Type.MouseMove, QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            # Only react when mouse is over this widget or its general area
            from PySide6.QtGui import QCursor as _QC
            gpos = _QC.pos()
            lpos = self.mapFromGlobal(gpos)
            
            # If mouse is over us and a key is pressed, grab focus if we don't have it
            # (but not if a text-input widget is currently being edited)
            if etype == QEvent.Type.KeyPress and self.rect().contains(lpos) and not self.hasFocus():
                from PySide6.QtWidgets import QApplication, QLineEdit, QTextEdit, QAbstractSpinBox
                focused = QApplication.focusWidget()
                if not isinstance(focused, (QLineEdit, QTextEdit, QAbstractSpinBox)):
                    self.setFocus()

            # Allow cursor change if mouse is within the viewport even if outside the image widget
            over_area = self.rect().contains(lpos)
            if not over_area and self.parent():
                viewport_rect = self.parent().rect()
                viewport_pos = self.parent().mapFromGlobal(gpos)
                if viewport_rect.contains(viewport_pos):
                    over_area = True

            if not over_area:
                return False

            if etype == QEvent.Type.MouseMove:
                return False

            key = event.key()
            if key == Qt.Key.Key_Alt and not event.isAutoRepeat():
                if etype == QEvent.Type.KeyPress:
                    self._set_mod_cursor(self._make_draw_cursor())
                else:
                    if not self._drawing:
                        self._clear_mod_cursor()
            elif key == Qt.Key.Key_Control and not event.isAutoRepeat():
                if etype == QEvent.Type.KeyPress:
                    self._set_mod_cursor(self._make_zoom_cursor())
                else:
                    self._clear_mod_cursor()
            elif key == Qt.Key.Key_Shift and not event.isAutoRepeat():
                if etype == QEvent.Type.KeyPress:
                    from PySide6.QtWidgets import QToolTip
                    QToolTip.hideText()
                    handle, _row = self._handle_hit_test_all_reviews(lpos)
                    self._set_mod_cursor(self._cursor_for_handle(handle) if handle else self._make_edit_cursor())
                else:
                    if not self._drawing:
                        self._clear_mod_cursor()
        return False

    def _set_mod_cursor(self, cursor):
        from PySide6.QtWidgets import QApplication
        if self._mod_cursor_active:
            QApplication.changeOverrideCursor(cursor)
        else:
            QApplication.setOverrideCursor(cursor)
            self._mod_cursor_active = True

    def _clear_mod_cursor(self):
        if self._mod_cursor_active:
            from PySide6.QtWidgets import QApplication
            QApplication.restoreOverrideCursor()
            self._mod_cursor_active = False

    def resizeEvent(self, event): self._compute_transform(); self.update()

    def sizeHint(self):
        if self._pixmap:
            if self._user_scale is None:
                vp = self.parent()
                if vp and vp.width() > 0:
                    return QSize(vp.width(), vp.height())
            return QSize(max(1, int(self._pixmap.width()  * self._scale)),
                         max(1, int(self._pixmap.height() * self._scale)))
        return QSize(800, 600)

    def _compute_transform(self):
        if not self._pixmap: return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if self._user_scale is None:
            # Compute fit scale from viewport (parent) size
            vp = self.parent()
            if vp and vp.width() > 0 and vp.height() > 0:
                vw, vh = vp.width(), vp.height()
            else:
                vw, vh = self.width() or pw, self.height() or ph
            self._fit_scale = min(1.0, min(vw / pw, vh / ph)) if pw and ph else 1.0
            self._scale = self._fit_scale
            ww = int(pw * self._scale)
            wh = int(ph * self._scale)
            self._offset_x = max(0, (vw - ww) // 2)
            self._offset_y = max(0, (vh - wh) // 2)
        else:
            self._scale = self._user_scale
            self._offset_x = 0
            self._offset_y = 0
        self._zoom = self._scale

    def _img_to_screen_rect(self, bbox: tuple) -> QRect:
        s = self._scale
        x, y, w, h = bbox
        return QRect(int(x*s)+self._offset_x, int(y*s)+self._offset_y,
                     max(4,int(w*s)), max(4,int(h*s)))

    def _screen_to_img(self, x: float, y: float) -> tuple[float, float]:
        """Convert screen (widget) pixel position to original image coordinate."""
        s = self._scale
        if not s: return 0.0, 0.0
        return (x - self._offset_x) / s, (y - self._offset_y) / s

    def _img_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Convert original image coordinate to screen (widget) pixel position."""
        s = self._scale
        return x * s + self._offset_x, y * s + self._offset_y

    def _screen_to_img_rect(self, rect: QRect) -> tuple:
        s = self._scale
        return (int((rect.x()-self._offset_x)/s), int((rect.y()-self._offset_y)/s),
                int(rect.width()/s), int(rect.height()/s))

    def _hit_test(self, pos: QPoint) -> int:
        for idx, ann in enumerate(self._annotations):
            if self._img_to_screen_rect(ann.bbox).contains(pos): return idx
        return -1

    def _hit_test_review(self, pos: QPoint) -> int:
        """Check click against _review_items (unconfirmed recognition results)."""
        for idx, ri in enumerate(self._review_items):
            bbox = ri.get('bbox')
            if bbox and self._img_to_screen_rect(bbox).contains(pos): return idx
        return -1
