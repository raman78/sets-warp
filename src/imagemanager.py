from os import listdir as os__listdir
from pathlib import Path
from PySide6.QtGui import QImage
from time import time
from urllib.parse import quote_plus, unquote_plus

from .cargomanager import CargoManager
from .constants import SEVEN_DAYS_IN_SECONDS
from .downloader import Downloader
from .iofunc import get_cached_cargo_data
from .setsdebug import log
from .splash import _state as _splash_state


class ImageManager():
    """Manages icons and ship images"""

    def __init__(
            self, images_dir: Path, ship_images_dir: Path, cargo_cache: CargoManager,
            downloader: Downloader, sync_manager=None):
        """
        Parameters:
        - :param images_dir: path to directory storing icons
        - :param ship_images_dir: path to directory storing ship images
        - :param cargo_cache: used to access cache
        - :param downloader: used to download icons and ship images
        - :param sync_manager: SyncManager instance for on-demand downloads
        """
        self._images_dir: Path = images_dir
        self._ship_images_dir: Path = ship_images_dir
        self._cargo_cache: CargoManager = cargo_cache
        self._downloader: Downloader = downloader
        self._sync: object = sync_manager   # SyncManager, set later via set_sync_manager()
        self.empty = QImage()
        self.image_set: set[str] = set()
        self.failed_images: dict[str, int] = dict()

    def set_sync_manager(self, sync_manager) -> None:
        """Called from datafunctions after SyncManager is constructed."""
        self._sync = sync_manager

    def get_downloaded_icons(self) -> set[str]:
        """
        Returns set containing all images currently in the images folder.
        """
        return set(map(lambda x: unquote_plus(x)[:-4], os__listdir(str(self._images_dir))))

    def download_images(self, skill_cache: dict[str, dict], threaded_worker=None, ship_images: list[str] = []):
        """
        Downloads wiki-only assets: Boff Abilities and Skill Icons.
        Item Icons and Ship Images are handled by SyncManager (GitHub-backed).
        """
        # Retry expired failures
        now = time()
        retry = [n for n, ts in self.failed_images.items() if now - ts >= SEVEN_DAYS_IN_SECONDS]
        for n in retry:
            del self.failed_images[n]
        no_retry = set(self.failed_images)
        available = self.get_downloaded_icons() | no_retry

        boff_images = sorted(
            self._cargo_cache.boff_abilities['all'].keys() - available)
        skill_images = sorted(
            self.get_skill_icons(skill_cache) - available)

        total = len(boff_images) + len(skill_images)
        log.info(f'download_images: wiki-only — boff={len(boff_images)}, skill={len(skill_images)}')

        if total == 0:
            _splash_state.update({'text': 'Downloading Images: up to date', 'current': 0, 'total': 0, 'hidden': True})
            return

        if self._sync is None:
            log.warning('download_images: no SyncManager — wiki downloads skipped')
            return

        def _splash(text, current, n):
            _splash_state.update({'text': text, 'current': current, 'total': total, 'hidden': False})

        if boff_images:
            failed = self._sync.download_wiki_group(
                'Boff Abilities', boff_images,
                suffix='_icon_(Federation).png', on_splash=_splash)
            self.failed_images.update(failed)

        if skill_images:
            failed = self._sync.download_wiki_group(
                'Skill Icons', skill_images,
                suffix='.png', on_splash=_splash)
            self.failed_images.update(failed)

        _splash_state.update({'text': 'Downloading: complete', 'current': total, 'total': total, 'hidden': False})

    def get_skill_icons(self, skill_cache: dict[str, dict]) -> set[str]:
        """
        Extracts skill icon names from skill cache.

        Parameters:
        - :param skill_cache: contains ground and space skill tree
        """
        icons = set()
        for rank_group in skill_cache['space']:
            for skill_group in rank_group:
                for skill_node in skill_group['nodes']:
                    icons.add(skill_node['image'])
        for skill_group in skill_cache['ground']:
            for skill_node in skill_group['nodes']:
                icons.add(skill_node['image'])
        return icons

    def get_ship_image(self, image_name: str, threaded_worker):
        """
        Tries to load ship image from local filesystem. If not available or corrupt,
        downloads and stores it. Passes the image back using the provided signal.

        Parameters:
        - :image_name: filename of the image
        - :param threaded_worker: thread object supplying signals
        """
        image_path = self._ship_images_dir / quote_plus(image_name)
        log.debug(f'ImageManager.get_ship_image: {image_name!r} -> {image_path}')
        image = QImage(str(image_path))
        if not image.isNull():
            log.debug(f'ImageManager.get_ship_image: loaded from disk '
                     f'size={image.width()}x{image.height()}')
            threaded_worker.result.emit((image,))
            return
        # Not on disk or corrupt — delete stale file and download
        if image_path.exists():
            log.warning(f'ImageManager.get_ship_image: file exists but QImage cannot load it '
                     f'(size={image_path.stat().st_size}B) — deleting and re-downloading')
            image_path.unlink(missing_ok=True)
        log.debug(f'ImageManager.get_ship_image: downloading...')
        if self._sync is not None:
            self._sync.download_one(image_name, 'ship')
        else:
            self._downloader.download_ship_image(image_name, {})
        image = QImage(str(image_path))
        log.debug(f'ImageManager.get_ship_image: after download null={image.isNull()} '
                 f'size={image.width()}x{image.height()}')
        threaded_worker.result.emit((image,))
