"""
warp/folder_picker.py

Two-pane folder picker:
  Left  — directory tree (selectable).
  Right — image files in the selected directory (preview only, no selection).

Replaces QFileDialog(FileMode.Directory) which on PySide6 + non-native
mode combined with a NameFilter became unselectable (NoSelection
workaround disabled folder clicks too).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QDir, QModelIndex
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTreeView, QListView,
    QAbstractItemView, QPushButton, QLabel, QFileSystemModel,
)

from warp.style import (
    apply_dark_style, primary_btn_style, secondary_btn_style,
    FG, MBG, BC,
)

DEFAULT_IMAGE_FILTERS = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')


class FolderPickerDialog(QDialog):
    """Split-pane folder picker with image preview."""

    def __init__(
        self,
        parent=None,
        title: str = 'Select Folder',
        start_dir: str | None = None,
        image_filters: tuple[str, ...] = DEFAULT_IMAGE_FILTERS,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        apply_dark_style(self)

        self._selected: Path | None = None

        # ── Models ───────────────────────────────────────────────────────────
        self._dir_model = QFileSystemModel(self)
        self._dir_model.setFilter(
            QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.NoDot
        )
        self._dir_model.setRootPath('')

        self._file_model = QFileSystemModel(self)
        self._file_model.setFilter(QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self._file_model.setNameFilters(list(image_filters))
        self._file_model.setNameFilterDisables(False)  # hide non-matching

        # ── Views ────────────────────────────────────────────────────────────
        self._tree = QTreeView(self)
        self._tree.setModel(self._dir_model)
        for col in range(1, self._dir_model.columnCount()):
            self._tree.hideColumn(col)
        self._tree.setHeaderHidden(True)
        self._tree.selectionModel().currentChanged.connect(self._on_dir_selected)
        self._tree.doubleClicked.connect(self._on_tree_double_click)

        self._list = QListView(self)
        self._list.setModel(self._file_model)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setUniformItemSizes(True)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._list)
        splitter.setSizes([400, 500])

        # ── Bottom bar ───────────────────────────────────────────────────────
        self._path_label = QLabel('—', self)
        self._path_label.setStyleSheet(
            f'color:{FG};background:{MBG};border:1px solid {BC};'
            f'border-radius:3px;padding:4px 8px;'
        )

        self._open_btn = QPushButton('Open', self)
        self._open_btn.setStyleSheet(primary_btn_style())
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._accept)

        cancel_btn = QPushButton('Cancel', self)
        cancel_btn.setStyleSheet(secondary_btn_style())
        cancel_btn.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._path_label, 1)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._open_btn)

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)
        root.addLayout(btn_row)

        # ── Pre-select start directory ───────────────────────────────────────
        if start_dir:
            p = Path(start_dir).expanduser()
            if p.is_dir():
                idx = self._dir_model.index(str(p))
                if idx.isValid():
                    self._tree.expand(idx)
                    self._tree.setCurrentIndex(idx)
                    self._tree.scrollTo(
                        idx, QAbstractItemView.ScrollHint.PositionAtCenter
                    )

    def _on_dir_selected(self, current: QModelIndex, _previous: QModelIndex):
        if not current.isValid():
            return
        path = self._dir_model.filePath(current)
        self._selected = Path(path)
        self._path_label.setText(path)
        self._file_model.setRootPath(path)
        self._list.setRootIndex(self._file_model.index(path))
        self._open_btn.setEnabled(True)

    def _on_tree_double_click(self, idx: QModelIndex):
        if idx.isValid() and self._dir_model.isDir(idx):
            self._accept()

    def _accept(self):
        if self._selected and self._selected.is_dir():
            self.accept()

    def selected_folder(self) -> Path | None:
        return self._selected


def pick_folder(
    parent,
    title: str = 'Select Folder',
    start_dir: str | None = None,
    image_filters: tuple[str, ...] = DEFAULT_IMAGE_FILTERS,
) -> Path | None:
    """Show the picker modally; return chosen folder or None on cancel."""
    dlg = FolderPickerDialog(parent, title=title, start_dir=start_dir,
                             image_filters=image_filters)
    if dlg.exec():
        return dlg.selected_folder()
    return None
