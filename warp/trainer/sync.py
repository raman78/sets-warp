# warp/trainer/sync.py
# Synchronises training data (annotations + icon crops) with Hugging Face Hub.
#
# Storage model:
#   - HF Dataset repo: "sets-sto/sto-icon-dataset"  (public, community)
#   - Structure:
#       data/
#           annotations.jsonl      — one JSON per line: {slot, name, crop_sha256, contributor}
#           crops/
#               <sha256>.png       — deduplicated by content hash
#
# Deduplication:
#   Each crop is identified by SHA-256 of its pixel content.
#   If a crop already exists in the dataset, it is not re-uploaded.
#
# Privacy:
#   No personal data is uploaded — only icon crops and their item names.
#   No screenshot filenames, usernames, or system paths are included.
#
# Requires: huggingface_hub (pip install huggingface-hub)
#           User must have a free HF account and provide a write token.

from __future__ import annotations

import json
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QCheckBox
)

from warp.trainer.training_data import TrainingDataManager

logger = logging.getLogger(__name__)

# Hugging Face dataset repository ID
HF_DATASET_REPO  = "sets-sto/sto-icon-dataset"
HF_REPO_TYPE     = "dataset"
ANNOTATIONS_FILE = "data/annotations.jsonl"
CROPS_DIR        = "data/crops"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class SyncWorker(QThread):
    """
    Uploads or downloads training data in background.

    Signals:
        progress(percent, message)
        finished(success: bool)
    """

    progress = Signal(int, str)
    finished = Signal(bool)

    def __init__(
        self,
        data_manager: TrainingDataManager,
        hf_token: str,
        mode: str = "upload",   # "upload" | "download" | "both"
    ):
        super().__init__()
        self._mgr   = data_manager
        self._token = hf_token
        self._mode  = mode

    def run(self):
        try:
            if self._mode in ("upload", "both"):
                self._upload()
            if self._mode in ("download", "both"):
                self._download()
            self.finished.emit(True)
        except Exception as e:
            logger.error(f"Sync error: {e}")
            self.finished.emit(False)

    # ---------------------------------------------------------------- upload

    def _upload(self):
        """Upload new confirmed crops + annotations to HF Hub."""
        from huggingface_hub import HfApi
        api = HfApi(token=self._token)

        # Ensure repo exists
        api.create_repo(
            repo_id=HF_DATASET_REPO,
            repo_type=HF_REPO_TYPE,
            exist_ok=True,
            private=False,
        )

        confirmed = self._mgr.get_confirmed_crops()
        if not confirmed:
            self.progress.emit(100, "Nothing to upload.")
            return

        # Fetch existing crop hashes to avoid re-uploading
        self.progress.emit(5, "Checking existing dataset…")
        existing_hashes = self._fetch_existing_hashes(api)

        new_annotations: list[dict] = []
        total = len(confirmed)

        for idx, item in enumerate(confirmed):
            pct = 5 + int(85 * idx / total)
            self.progress.emit(pct, f"Uploading {idx+1}/{total}…")

            crop_path = Path(item["path"])
            if not crop_path.exists():
                continue

            # Content-hash deduplication
            sha = self._file_sha256(crop_path)
            crop_hf_path = f"{CROPS_DIR}/{sha}.png"

            if sha not in existing_hashes:
                api.upload_file(
                    path_or_fileobj=str(crop_path),
                    path_in_repo=crop_hf_path,
                    repo_id=HF_DATASET_REPO,
                    repo_type=HF_REPO_TYPE,
                )
                existing_hashes.add(sha)

            new_annotations.append({
                "slot":       item["slot"],
                "name":       item["name"],
                "crop_sha256": sha,
            })

        if new_annotations:
            self.progress.emit(92, "Appending annotations…")
            self._append_annotations(api, new_annotations)

        self.progress.emit(100, f"Uploaded {len(new_annotations)} annotations.")

    def _fetch_existing_hashes(self, api) -> set[str]:
        """Returns set of SHA-256 hashes already in the dataset."""
        try:
            files = api.list_repo_files(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
            )
            return {
                Path(f).stem
                for f in files
                if f.startswith(CROPS_DIR) and f.endswith(".png")
            }
        except Exception:
            return set()

    def _append_annotations(self, api, new_entries: list[dict]):
        """
        Downloads current annotations.jsonl, appends new entries, re-uploads.
        Uses a simple dedup by crop_sha256 to avoid duplicates.
        """
        import io

        # Try to download existing
        existing_lines: list[str] = []
        existing_hashes: set[str] = set()
        try:
            content = api.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=ANNOTATIONS_FILE,
                repo_type=HF_REPO_TYPE,
            )
            with open(content) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            existing_hashes.add(d.get("crop_sha256", ""))
                        except Exception:
                            pass
                        existing_lines.append(line)
        except Exception:
            pass  # File doesn't exist yet — that's fine

        # Append new, deduplicated
        combined = list(existing_lines)
        for entry in new_entries:
            if entry["crop_sha256"] not in existing_hashes:
                combined.append(json.dumps(entry))
                existing_hashes.add(entry["crop_sha256"])

        # Upload merged file
        content_bytes = "\n".join(combined).encode("utf-8")
        api.upload_file(
            path_or_fileobj=io.BytesIO(content_bytes),
            path_in_repo=ANNOTATIONS_FILE,
            repo_id=HF_DATASET_REPO,
            repo_type=HF_REPO_TYPE,
        )

    # ---------------------------------------------------------------- download

    def _download(self):
        """
        Download annotations.jsonl from HF Hub and update local crop index
        with community-contributed names (useful for improving local model).
        """
        self.progress.emit(10, "Downloading community annotations…")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=ANNOTATIONS_FILE,
                repo_type=HF_REPO_TYPE,
            )
            count = 0
            with open(path) as f:
                for line in f:
                    if line.strip():
                        count += 1
            self.progress.emit(100, f"Downloaded {count} community annotations.")
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            self.progress.emit(100, "Download failed — dataset may be empty.")

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _file_sha256(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()[:32]   # first 32 chars is sufficient


# ---------------------------------------------------------------------------
# HF Token setup dialog
# ---------------------------------------------------------------------------

class HFTokenDialog(QDialog):
    """
    Simple dialog to enter and save a Hugging Face write token.
    Token is stored in QSettings (not in plaintext files).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hugging Face Token")
        self.setFixedWidth(440)
        self._token = ""
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        info = QLabel(
            "To share training data with the community, you need a free\n"
            "Hugging Face account and a write-access token.\n\n"
            "Get your token at: https://huggingface.co/settings/tokens\n"
            "(Create a token with 'write' role)"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        lay.addWidget(info)

        lay.addWidget(QLabel("Hugging Face Token:"))
        self._token_edit = QLineEdit()
        self._token_edit.setPlaceholderText("hf_…")
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        lay.addWidget(self._token_edit)

        self._show_cb = QCheckBox("Show token")
        self._show_cb.toggled.connect(
            lambda on: self._token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        lay.addWidget(self._show_cb)

        btns = QHBoxLayout()
        btn_ok     = QPushButton("Save & Continue")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self._on_ok)
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        lay.addLayout(btns)

    def _on_ok(self):
        self._token = self._token_edit.text().strip()
        if not self._token:
            return
        self.accept()

    def get_token(self) -> str:
        return self._token
