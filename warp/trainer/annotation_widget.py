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
# Change this one value to update both the drawn rectangle and the cursor colour.
DRAW_BBOX_COLOR = QColor(255, 200, 0)   # yellow — matches Add BBox button style


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

        # Mode flags — drawing/editing only active when explicitly enabled
        self._draw_mode_forced: bool = False   # set by + Add BBox / Edit BBox
        self._alt_draw: bool = False             # True when drawing via Alt+LMB

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

        # Drag/resize state
        self._drag_mode:  str | None = None   # 'move' | 'resize_NW' | etc.
        self._drag_start: QPoint | None = None
        self._drag_orig:  tuple | None = None  # original bbox at drag start
        # Handle size in screen pixels
        self._HANDLE = 9

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #111;")

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
        self._compute_transform()
        if self._pixmap:
            self.resize(self._pixmap.width(), self._pixmap.height())
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

    # ---------------------------------------------------------------- painting

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._pixmap:
            zw = int(self._pixmap.width()  * self._scale)
            zh = int(self._pixmap.height() * self._scale)
            painter.drawPixmap(self._offset_x, self._offset_y, zw, zh, self._pixmap)
        else:
            painter.fillRect(self.rect(), QColor("#111")); painter.setPen(QColor("#555")); painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded\nOpen a folder to start")
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
        'pending':   QColor(200, 200, 200, 180),
        'confirmed': QColor( 60, 220, 100, 220),
        'new':       QColor(255, 220,   0, 200),
    }

    def _draw_review_item(self, painter: QPainter, bbox: tuple, state: str, name: str, slot: str, selected: bool, highlighted: bool):
        if not bbox: return
        if highlighted and not selected:
            color = QColor(255, 50, 50, 220); pw = SELECTED_PEN_WIDTH + 1; style = Qt.PenStyle.DashLine
        else:
            color = self._STATE_COLOR.get(state, QColor(200, 200, 200, 180)); pw = SELECTED_PEN_WIDTH if selected else DRAW_PEN_WIDTH; style = Qt.PenStyle.SolidLine
        pen = QPen(color, pw, style); painter.setPen(pen); painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 25))); rect = self._img_to_screen_rect(bbox); painter.drawRect(rect)
        # badge text removed — info shown via tooltip in review list
        if selected:
            h = self._HANDLE; painter.setPen(QPen(QColor(0, 0, 0, 180), 1)); painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
            for hx, hy in self._handle_positions(rect): painter.drawRect(hx - h//2, hy - h//2, h, h)

    # ---------------------------------------------------------------- mouse events

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton: return
        pos = event.pos()
        # Alt+LMB drag — start drawing a new bbox without toggling Add BBox button
        alt_held = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
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
            self._selected_idx = -1
            self.item_deselected.emit()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.pos()
        if self._drawing: self._draw_current = pos; self.update(); return
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
        if self._selected_idx >= 0:
            handle = self._handle_hit_test(pos, self._selected_idx)
            if handle: self.setCursor(self._cursor_for_handle(handle)); return
        # Preserve draw cursor if Alt is held
        from PySide6.QtWidgets import QApplication as _QApp
        if _QApp.queryKeyboardModifiers() & Qt.KeyboardModifier.AltModifier:
            self.setCursor(self._make_draw_cursor())
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
        if self._drag_mode:
            self._drag_mode = None; self._drag_start = None
            self._drag_orig = None; self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete and self._selected_idx >= 0:
            ann = self._annotations[self._selected_idx]; self._data_mgr.remove_annotation(self._img_path, ann); self._annotations = self._data_mgr.get_annotations(self._img_path); self._selected_idx = -1; self.update()

    def _handle_positions(self, rect: QRect) -> list[tuple[int, int]]:
        l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom(); mx, my = (l + r) // 2, (t + b) // 2
        return [(l, t), (mx, t), (r, t), (l, my), (r, my), (l, b), (mx, b), (r, b), (mx, my)]

    def _handle_hit_test(self, pos: QPoint, ann_idx: int) -> str | None:
        if ann_idx < 0 or ann_idx >= len(self._annotations): return None
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
    def _cursor_for_handle(handle: str) -> Qt.CursorShape:
        return {'move': Qt.CursorShape.SizeAllCursor, 'resize_NW': Qt.CursorShape.SizeFDiagCursor, 'resize_SE': Qt.CursorShape.SizeFDiagCursor, 'resize_NE': Qt.CursorShape.SizeBDiagCursor, 'resize_SW': Qt.CursorShape.SizeBDiagCursor, 'resize_N': Qt.CursorShape.SizeVerCursor, 'resize_S': Qt.CursorShape.SizeVerCursor, 'resize_W': Qt.CursorShape.SizeHorCursor, 'resize_E': Qt.CursorShape.SizeHorCursor}.get(handle, Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        from src.setsdebug import log as _sl
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl:
            if not self._pixmap: return
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            # Use _fit_scale computed at load time — stable, doesn't grow with widget
            fit_s = self._fit_scale
            old_s = self._scale
            new_s = max(fit_s, min(fit_s * 6.0, old_s * factor))
            _sl.info(f'AW.zoom fit={fit_s:.3f} old={old_s:.3f} new={new_s:.3f} delta={delta}')
            if abs(new_s - old_s) < 0.0001: return
            # Map cursor to widget coords
            try:
                pos = event.position()
                cx, cy = float(pos.x()), float(pos.y())
                sa = self.parent()
                vp = sa.viewport() if sa and hasattr(sa, 'viewport') else None
                if vp:
                    from PySide6.QtCore import QPoint
                    widget_pos = self.mapFrom(vp, QPoint(int(cx), int(cy)))
                    cx, cy = float(widget_pos.x()), float(widget_pos.y())
            except Exception:
                cx, cy = self.width() / 2.0, self.height() / 2.0
            img_x = (cx - self._offset_x) / old_s if old_s else 0.0
            img_y = (cy - self._offset_y) / old_s if old_s else 0.0
            if new_s <= fit_s * 1.001:
                self._user_scale = None
            else:
                self._user_scale = new_s
            self._compute_transform()
            if self._user_scale is not None:
                self._offset_x = int(cx - img_x * new_s)
                self._offset_y = int(cy - img_y * new_s)
            self.adjustSize()
            self.update()
            event.accept()
        else:
            super().wheelEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        from PySide6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)

    def hideEvent(self, event):
        super().hideEvent(event)
        from PySide6.QtWidgets import QApplication
        QApplication.instance().removeEventFilter(self)

    def enterEvent(self, event):
        """Mouse entered canvas area — if Alt held, show draw cursor."""
        from PySide6.QtWidgets import QApplication
        mods = QApplication.queryKeyboardModifiers()
        if mods & Qt.KeyboardModifier.AltModifier:
            self.setCursor(self._make_draw_cursor())

    def leaveEvent(self, event):
        """Mouse left canvas area — restore normal cursor."""
        if not self._drawing:
            self.unsetCursor()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        etype = event.type()
        if etype in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            # Only react when mouse is over this widget
            from PySide6.QtGui import QCursor as _QC
            if not self.rect().contains(self.mapFromGlobal(_QC.pos())):
                return False
            key = event.key()
            if key == Qt.Key.Key_Alt and not event.isAutoRepeat():
                if etype == QEvent.Type.KeyPress:
                    self.setCursor(self._make_draw_cursor())
                else:
                    if not self._drawing:
                        self.unsetCursor()
        return False

    def resizeEvent(self, event): self._compute_transform(); self.update()

    def sizeHint(self):
        if self._pixmap:
            return QSize(max(1,int(self._pixmap.width()*self._scale)),
                         max(1,int(self._pixmap.height()*self._scale)))
        return QSize(800, 600)

    def _compute_transform(self):
        if not self._pixmap: return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if self._user_scale is None:
            # Fit-to-window using stored _fit_scale
            # _fit_scale is set once at load_image, never changes
            self._scale = self._fit_scale
            ww = int(pw * self._scale)
            wh = int(ph * self._scale)
            self._offset_x = max(0, (self.width()  - ww) // 2)
            self._offset_y = max(0, (self.height() - wh) // 2)
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

    def _screen_to_img_rect(self, rect: QRect) -> tuple:
        s = self._scale
        return (int((rect.x()-self._offset_x)/s), int((rect.y()-self._offset_y)/s),
                int(rect.width()/s), int(rect.height()/s))

    def _hit_test(self, pos: QPoint) -> int:
        for idx, ann in enumerate(self._annotations):
            if self._img_to_screen_rect(ann.bbox).contains(pos): return idx
        return -1
