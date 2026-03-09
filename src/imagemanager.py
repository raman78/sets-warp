from os import listdir as os__listdir
from pathlib import Path
from PySide6.QtGui import QImage
from threading import Lock
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
            downloader: Downloader):
        """
        Parameters:
        - :param images_dir: path to directory storing icons
        - :param ship_images_dir: path to directory storing ship images
        - :param cargo_cache: used to access cache
        - :param downloader: used to download icons and ship images
        """
        self._images_dir: Path = images_dir
        self._ship_images_dir: Path = ship_images_dir
        self._cargo_cache: CargoManager = cargo_cache
        self._downloader: Downloader = downloader
        self.empty = QImage()
        self.image_set: set[str] = set()
        self.failed_images: dict[str, int] = dict()

    def get_downloaded_icons(self) -> set[str]:
        """
        Returns set containing all images currently in the images folder.
        """
        return set(map(lambda x: unquote_plus(x)[:-4], os__listdir(str(self._images_dir))))

    def download_images(self, skill_cache: dict[str, dict], threaded_worker=None, ship_images: list[str] = []):
        """
        Ensures that all required images are downloaded to disk. Updates `failed_images`.
        Reports progress directly to _splash_state (thread-safe).
        """
        no_retry_images = set()
        retry_images = list()
        now = time()
        for image_name, timestamp in self.failed_images.items():
            if now - timestamp < SEVEN_DAYS_IN_SECONDS:
                no_retry_images.add(image_name)
            else:
                retry_images.append(image_name)
        for image_name in retry_images:
            del self.failed_images[image_name]
        available_images = self.get_downloaded_icons() | no_retry_images

        ultimate_skill_icons = {'Focused Frenzy', 'Probability Manipulation', 'EPS Corruption'}
        image_set = self.image_set | ultimate_skill_icons
        images = image_set - available_images - self._cargo_cache.boff_abilities['all'].keys()
        boff_images = self._cargo_cache.boff_abilities['all'].keys() - available_images
        skill_images = self.get_skill_icons(skill_cache) - available_images

        # Ship images — skip already downloaded (valid file on disk, size > 100B)
        downloaded_ship = set(
            unquote_plus(f) for f in os__listdir(str(self._ship_images_dir))
            if (self._ship_images_dir / f).stat().st_size > 100
        ) if self._ship_images_dir.exists() else set()
        # compare decoded filenames with original names
        missing_ships = [name for name in ship_images if name not in downloaded_ship]
        log.info(f'download_images: ship images on disk={len(downloaded_ship)}, '
                 f'total={len(ship_images)}, missing={len(missing_ships)}')

        batches = [
            ('Icons',        list(images),        'icon',  None),
            ('Boff Abilities', list(boff_images), 'icon',  '_icon_(Federation).png'),
            ('Skill Icons',  list(skill_images),  'icon',  '.png'),
            ('Ship Images',  missing_ships,       'ship',  ''),
        ]
        # filter empty batches
        batches = [(label, lst, itype, suffix) for label, lst, itype, suffix in batches if lst]

        total_files = len(images) + len(boff_images) + len(skill_images) + len(missing_ships)
        total_steps = len(batches)

        log.info(f'download_images: total={total_files} (icons={len(images)}, '
                 f'boff={len(boff_images)}, skill={len(skill_images)}, '
                 f'ships={len(missing_ships)})')

        if total_files == 0:
            _splash_state.update({'text': 'Downloading Images: up to date', 'current': 0, 'total': 0, 'hidden': True})
            return

        # global counter across all batches for the progress bar
        counter = [0]
        lock = Lock()

        for step_idx, (label, image_list, itype, suffix) in enumerate(batches):
            step_num = step_idx + 1
            batch_size = len(image_list)

            # show initial state for this step before download starts
            _splash_state.update({
                'text': f'Downloading {label}',
                'current': counter[0],
                'total': total_files,
                'hidden': False,
            })
            log.info(f'download_images: step {step_num}/{total_steps} — {label} ({batch_size} files)')

            def on_progress(
                    _label=label, _step_num=step_num, _total_steps=total_steps,
                    _batch_size=batch_size):
                """Called by download threads — writes to _splash_state only (no Qt)."""
                with lock:
                    counter[0] += 1
                    c = counter[0]
                # Count within this batch = how many of the global counter belong here.
                # We track it simply as global counter for the bar and recompute batch done.
                _splash_state.update({
                    'text': f'Downloading {_label}',
                    'current': c,
                    'total': total_files,
                    'hidden': False,
                })

            kwargs = {'on_progress': on_progress, 'image_type': itype}
            if suffix is not None:
                kwargs['image_suffix'] = suffix

            failed = self._downloader.download_image_list(image_list, **kwargs)
            self.failed_images.update(failed)
            log.info(f'download_images: step {step_num} done, failed={len(failed)}')

        _splash_state.update({
            'text': f'Downloading: complete',
            'current': total_files,
            'total': total_files,
            'hidden': False,
        })

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
        log.info(f'ImageManager.get_ship_image: {image_name!r} -> {image_path}')
        image = QImage(str(image_path))
        if not image.isNull():
            log.info(f'ImageManager.get_ship_image: loaded from disk '
                     f'size={image.width()}x{image.height()}')
            threaded_worker.result.emit((image,))
            return
        # Not on disk or corrupt — delete stale file and download
        if image_path.exists():
            log.info(f'ImageManager.get_ship_image: file exists but QImage cannot load it '
                     f'(size={image_path.stat().st_size}B) — deleting and re-downloading')
            image_path.unlink(missing_ok=True)
        log.info(f'ImageManager.get_ship_image: downloading...')
        self._downloader.download_ship_image(image_name, {})
        image = QImage(str(image_path))
        log.info(f'ImageManager.get_ship_image: after download null={image.isNull()} '
                 f'size={image.width()}x{image.height()}')
        threaded_worker.result.emit((image,))
