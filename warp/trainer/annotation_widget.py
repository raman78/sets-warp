# warp/trainer/annotation_widget.py
# Interactive canvas for annotating STO screenshots.
#
# Features:
#   - Displays screenshot scaled to fit available space
#   - Shows detected/pending bounding boxes as overlays
#   - User can draw new bboxes by click+drag (left mouse button)
#   - Click on existing bbox to select it
#   - Delete key removes selected bbox
#   - Confirmed boxes are green, pending are yellow, skipped are grey
#
# Coordinate system:
#   All annotations are stored in original image pixel coordinates.
#   The widget transforms between screen coordinates and image coordinates
#   automatically based on current zoom/scale.

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QWidget, QSizePolicy, QScrollArea
from PySide6.QtCore    import Qt, QRect, QPoint, QRectF, Signal, QSize
from PySide6.QtGui     import (
    QPainter, QPixmap, QColor, QPen, QBrush, QFont,
    QMouseEvent, QPaintEvent, QKeyEvent
)

from warp.trainer.training_data import TrainingDataManager, Annotation, AnnotationState


# Colour scheme for annotation states
STATE_COLORS = {
    AnnotationState.PENDING:   QColor(255, 200,   0, 180),   # yellow
    AnnotationState.CONFIRMED: QColor( 60, 220, 100, 200),   # green
    AnnotationState.SKIPPED:   QColor(160, 160, 160, 120),   # grey
    # CANDIDATE boxes are intentionally hidden — they clutter the screen
}

DRAW_PEN_WIDTH   = 2
SELECTED_PEN_WIDTH = 3
FONT_SIZE_BADGE  = 9


class AnnotationWidget(QWidget):
    """
    Screenshot viewer with interactive bbox annotation overlay.

    Signals:
        annotation_added(bbox: tuple)     — user finished drawing a new bbox
        item_selected(annotation: dict)   — user clicked an existing annotation
    """

    annotation_added = Signal(tuple)    # (x, y, w, h) in image coords
    item_selected    = Signal(dict)     # annotation dict

    def __init__(self, data_manager: TrainingDataManager, parent=None):
        super().__init__(parent)
        self._data_mgr    = data_manager
        self._pixmap:   QPixmap | None = None
        self._img_path: Path | None    = None
        self._scale:    float          = 1.0
        self._offset_x: int            = 0
        self._offset_y: int            = 0

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

        self._highlight_bbox: tuple | None = None
        self._draw_mode_forced: bool = False

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self._highlight_bbox = None
        self._drawing       = False
        self._compute_transform()
        self.update()

    def confirm_current(self, slot: str, name: str):
        """
        Confirm the currently selected / pending annotation with slot+name.
        If a new bbox was drawn, saves it. If an existing annotation is selected,
        updates it.
        """
        if self._pending_bbox is not None:
            ann = self._data_mgr.add_annotation(
                image_path=self._img_path,
                bbox=self._pending_bbox,
                slot=slot,
                name=name,
                state=AnnotationState.CONFIRMED,
            )
            self._annotations = self._data_mgr.get_annotations(self._img_path)
            self._pending_bbox = None
            self._selected_idx = len(self._annotations) - 1

        elif self._selected_idx >= 0:
            ann = self._annotations[self._selected_idx]
            ann.slot  = slot
            ann.name  = name
            ann.state = AnnotationState.CONFIRMED
            self._data_mgr.update_annotation(self._img_path, ann)
            self._annotations = self._data_mgr.get_annotations(self._img_path)

        self.update()

    def skip_current(self):
        """Mark currently selected annotation as skipped."""
        if self._selected_idx >= 0:
            ann = self._annotations[self._selected_idx]
            ann.state = AnnotationState.SKIPPED
            self._data_mgr.update_annotation(self._img_path, ann)
        self._pending_bbox = None
        self._selected_idx = -1
        self.update()

    def all_confirmed(self) -> bool:
        """Returns True if all non-skipped annotations are confirmed."""
        if not self._annotations:
            return False
        return all(
            a.state in (AnnotationState.CONFIRMED, AnnotationState.SKIPPED)
            for a in self._annotations
        )

    def highlight_bbox(self, bbox: tuple):
        """Highlight a specific bbox from recognition (shown as orange dashed rect)."""
        self._highlight_bbox = bbox
        self.update()

    def set_draw_mode(self, enabled: bool):
        """Enable or disable manual bbox drawing mode."""
        self._draw_mode_forced = enabled
        if not enabled:
            self._drawing = False
            self._draw_start = None
            self._draw_current = None
        self.update()

    # ---------------------------------------------------------------- painting

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw image
        if self._pixmap:
            self._compute_transform()
            painter.drawPixmap(self._offset_x, self._offset_y,
                               int(self._pixmap.width()  * self._scale),
                               int(self._pixmap.height() * self._scale),
                               self._pixmap)
        else:
            painter.fillRect(self.rect(), QColor("#111"))
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No image loaded\nOpen a folder to start")
            return

        # Draw annotations
        for idx, ann in enumerate(self._annotations):
            self._draw_annotation(painter, ann, selected=(idx == self._selected_idx))

        # Draw in-progress bbox
        if self._drawing and self._draw_start and self._draw_current:
            pen = QPen(QColor(255, 255, 0), DRAW_PEN_WIDTH, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 255, 0, 30)))
            rect = QRect(self._draw_start, self._draw_current).normalized()
            painter.drawRect(rect)

        # Draw pending bbox (drawn, awaiting confirmation)
        if self._pending_bbox:
            prect = self._img_to_screen_rect(self._pending_bbox)
            pen   = QPen(QColor(255, 180, 0), DRAW_PEN_WIDTH + 1, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 180, 0, 50)))
            painter.drawRect(prect)
            painter.setPen(QColor(255, 180, 0))
            painter.setFont(QFont("", FONT_SIZE_BADGE, QFont.Weight.Bold))
            painter.drawText(prect.topLeft() + QPoint(2, -3), "NEW")

        # Draw recognition highlight (from review panel selection)
        if self._highlight_bbox:
            hrect = self._img_to_screen_rect(self._highlight_bbox)
            pen   = QPen(QColor(126, 200, 227), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(126, 200, 227, 25)))
            painter.drawRect(hrect)

    def _draw_annotation(self, painter: QPainter, ann: Annotation, selected: bool):
        # Skip CANDIDATE annotations — they clutter the view
        if ann.state == AnnotationState.CANDIDATE:
            return
        color = STATE_COLORS.get(ann.state, QColor(200, 200, 200, 150))
        pen   = QPen(color, SELECTED_PEN_WIDTH if selected else DRAW_PEN_WIDTH)
        if selected:
            pen.setStyle(Qt.PenStyle.DashLine)

        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))

        rect = self._img_to_screen_rect(ann.bbox)
        painter.drawRect(rect)

        # Label badge
        if ann.name or ann.slot:
            badge = ann.name or ann.slot
            painter.setPen(color)
            painter.setFont(QFont("", FONT_SIZE_BADGE))
            painter.drawText(rect.bottomLeft() + QPoint(2, 12), badge[:24])

    # ---------------------------------------------------------------- mouse events

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            # Check if clicking existing annotation
            clicked = self._hit_test(pos)
            if clicked >= 0:
                self._selected_idx = clicked
                self._pending_bbox = None
                ann = self._annotations[clicked]
                self.item_selected.emit({
                    "slot": ann.slot,
                    "name": ann.name,
                    "bbox": ann.bbox,
                })
            else:
                # Start drawing new bbox
                self._drawing      = True
                self._draw_start   = pos
                self._draw_current = pos
                self._selected_idx = -1
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drawing:
            self._draw_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            if self._draw_start and self._draw_current:
                screen_rect = QRect(self._draw_start, self._draw_current).normalized()
                if screen_rect.width() > 8 and screen_rect.height() > 8:
                    self._pending_bbox = self._screen_to_img_rect(screen_rect)
                    self.annotation_added.emit(self._pending_bbox)
            self._draw_start   = None
            self._draw_current = None
            self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete and self._selected_idx >= 0:
            ann = self._annotations[self._selected_idx]
            self._data_mgr.remove_annotation(self._img_path, ann)
            self._annotations = self._data_mgr.get_annotations(self._img_path)
            self._selected_idx = -1
            self.update()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Forward to bottom controls via parent
            pass

    def resizeEvent(self, event):
        self._compute_transform()
        self.update()

    # ---------------------------------------------------------------- coordinate transforms

    def sizeHint(self):
        """Report 1:1 image size so QScrollArea sets scrollbars correctly."""
        if self._pixmap:
            return QSize(self._pixmap.width(), self._pixmap.height())
        return QSize(800, 600)

    def _compute_transform(self):
        """
        Always display at original 1:1 pixel scale (no stretching).
        Centre the image in the widget when the widget is larger than the image.
        When the widget is smaller the scroll area handles panning.
        """
        if not self._pixmap:
            return
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        ww = self.width()
        wh = self.height()
        self._scale    = 1.0
        self._offset_x = max(0, (ww - pw) // 2)
        self._offset_y = max(0, (wh - ph) // 2)

    def _img_to_screen_rect(self, bbox: tuple) -> QRect:
        x, y, w, h = bbox
        sx = int(x * self._scale) + self._offset_x
        sy = int(y * self._scale) + self._offset_y
        sw = max(4, int(w * self._scale))
        sh = max(4, int(h * self._scale))
        return QRect(sx, sy, sw, sh)

    def _screen_to_img_rect(self, rect: QRect) -> tuple:
        x = int((rect.x() - self._offset_x) / self._scale)
        y = int((rect.y() - self._offset_y) / self._scale)
        w = int(rect.width()  / self._scale)
        h = int(rect.height() / self._scale)
        return (x, y, w, h)

    def _hit_test(self, pos: QPoint) -> int:
        """Returns index of annotation under cursor, or -1."""
        for idx, ann in enumerate(self._annotations):
            rect = self._img_to_screen_rect(ann.bbox)
            if rect.contains(pos):
                return idx
        return -1
