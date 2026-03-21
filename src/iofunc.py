from datetime import datetime
import json
from json import load as json__load, JSONDecodeError
import os
from pathlib import Path
from shutil import copyfile as shutil__copyfile, rmtree as shutil__rmtree
import sys
from threading import Thread
from urllib.parse import quote_plus, unquote_plus
from webbrowser import open as webbrowser_open

from PySide6.QtGui import QIcon, QImage
from PySide6.QtWidgets import QFileDialog
import requests
from requests.cookies import create_cookie as requests__create_cookie
from lxml import html as lxml_html
from lxml.cssselect import CSSSelector

from .constants import WIKI_IMAGE_URL, WIKI_URL
from .textedit import compensate_json
from .setsdebug import log


class ReturnValueThread(Thread):
    def __init__(self, target, args: tuple = tuple()):
        super().__init__(target=target, args=args)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args)

    def join(self):
        super().join()
        return self._return


def browse_path(self, default_path: str = None, types: str = 'Any File (*.*)', save=False) -> str:
    """
    Opens file dialog prompting the user to select a file.

    Uses Qt's own (non-native) dialog so that:
    - Ctrl+L opens a path bar for typing / pasting a directory path directly
    - Hidden directories (starting with .) are toggleable via Ctrl+H

    Parameters:
    - :param default_path: path that the file dialog opens at
    - :param types: string containing all file extensions and their respective names that are
    allowed.
    Format: "<name of file type> (*.<extension>);;<name of file type> (*.<extension>);; [...]"
    Example: "Logfile (*.log);;Any File (*.*)"
    """
    if default_path is None or default_path == '':
        default_path = self.app_dir
    default_path = os.path.abspath(default_path)
    if not os.path.exists(os.path.dirname(default_path)):
        default_path = self.app_dir

    # Use QFileDialog instance so DontUseNativeDialog works on Wayland too.
    # Static methods (getSaveFileName/getOpenFileName) may ignore the flag
    # when QT_QPA_PLATFORM=wayland; instantiating directly always works.
    from PySide6.QtCore import QDir
    dialog = QFileDialog(self.window)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.Option.HideNameFilterDetails, True)
    # Always show hidden files and dirs (e.g. .config on NTFS or any mount)
    dialog.setFilter(QDir.Filter.AllEntries | QDir.Filter.Hidden | QDir.Filter.NoDotAndDotDot)
    dialog.setNameFilter(types)

    # When there is only one filter: hide the entire "Files of type" row
    # (label + combo) so the grid layout doesn't shift the filename field.
    if ';;' not in types:
        from PySide6.QtWidgets import QComboBox, QLabel
        from PySide6.QtCore import QTimer

        def _hide_row():
            c = dialog.findChild(QComboBox, 'fileTypeCombo')
            if c is None:
                cs = dialog.findChildren(QComboBox)
                c = cs[-1] if cs else None
            if c:
                c.setVisible(False)
            for lbl in dialog.findChildren(QLabel):
                if 'type' in lbl.text().lower():
                    lbl.setVisible(False)
                    break

        QTimer.singleShot(0, _hide_row)

    # Restore last used directory (shared between save and open)
    last_dir = self.settings.value('last_dialog_dir', '')
    if last_dir and os.path.isdir(last_dir):
        dialog.setDirectory(last_dir)
    else:
        dialog.setDirectory(os.path.dirname(default_path))

    if save:
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.selectFile(os.path.basename(default_path))
        if dialog.exec():
            files = dialog.selectedFiles()
            file = files[0] if files else ''
            if file:
                self.settings.setValue('last_dialog_dir', os.path.dirname(file))
                filter = dialog.selectedNameFilter()
                selected_extension = filter.rpartition('.')[2][:-1]
                if selected_extension and file.rpartition('.')[2].lower() != selected_extension:
                    file += f".{selected_extension}"
        else:
            file = ''
    else:
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if dialog.exec():
            files = dialog.selectedFiles()
            file = files[0] if files else ''
            if file:
                self.settings.setValue('last_dialog_dir', os.path.dirname(file))
        else:
            file = ''
    return file


def get_cargo_data(self, filename: str, url: str, ignore_cache_age=False) -> dict | list:
    """
    Retrieves cargo data for specific table. Downloads cargo data from wiki if cargo cache is empty.
    Updates cargo cache.

    Parameters:
    - :param filename: filename of cache file
    - :param url: url to cargo table
    - :param ignore_cache_age: True if cache of any age should be accepted
    """
    filepath = os.path.join(self.config['config_subfolders']['cargo'], filename)
    cargo_data = None

    # try loading from cache
    if os.path.exists(filepath) and os.path.isfile(filepath):
        last_modified = os.path.getmtime(filepath)
        if (datetime.now() - datetime.fromtimestamp(last_modified)).days < 7 or ignore_cache_age:
            try:
                return load_json(filepath)
            except json.JSONDecodeError:
                pass

    # download cargo data if loading from cache failed or data should be updated
    try:
        cargo_data = self.downloader.download_cargo_table(url, filename)
        if cargo_data is not None:
            auto_backup_cargo_file(self, filename)
            store_json(cargo_data, filepath)
            return cargo_data
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        if ignore_cache_age:
            backup_path = os.path.join(self.config['config_subfolders']['backups'], filename)
            auto_backup_path = os.path.join(
                    self.config['config_subfolders']['auto_backups'], filename)
            if self.settings.value('pref_backup', type=int) == 0:
                backup_paths = (auto_backup_path, backup_path)
            else:
                backup_paths = (backup_path, auto_backup_path)
            for path in backup_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    try:
                        cargo_data = load_json(path)
                        store_json(cargo_data, filepath)
                        return cargo_data
                    except json.JSONDecodeError:
                        pass
            sys.stderr.write(f'[Error] Cargo table could not be retrieved ({filename})\n')
            sys.exit(1)
        else:
            return get_cargo_data(self, filename, url, ignore_cache_age=True)


def get_cached_cargo_data(self, filename: str) -> dict | list:
    """
    Retrieves cached cargo data from filename. Returns empty dict when cache is too old or
    corrupted.

    Parameters:
    - :param filename: name of the cache file
    """
    filepath = os.path.join(self.config['config_subfolders']['cache'], filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        last_modified = os.path.getmtime(filepath)
        if (datetime.now() - datetime.fromtimestamp(last_modified)).days < 7:
            try:
                return load_json(filepath)
            except json.JSONDecodeError:
                pass
    return {}


def store_to_cache(self, data, filename: str):
    """
    Stores data to cache file with filename.

    Parameters:
    - :param data: data that will be stored
    - :param filename: filename of the cache file
    """
    filepath = os.path.join(self.config['config_subfolders']['cache'], filename)
    store_json(data, filepath)


def retrieve_image(
        self, name: str, image_folder_path: str, signal=None, url_override: str = '') -> QImage:
    """
    Downloads image or fetches image from cache.

    Parameters:
    - :param name: name of the item
    - :param image_folder_path: path to the image folder
    - :param signal: signal that is emitted to chance splash when downloading image (optional)
    - :param url_override: non default image url (optional)
    """
    filename = get_image_file_name(name)
    filepath = os.path.join(image_folder_path, filename)
    # log.debug(f'retrieve_image: {name!r} -> {filepath}')
    image = QImage(filepath)
    if image.isNull():
        log.info(f'retrieve_image: cache miss, downloading {name!r}')
        if signal is not None:
            signal.emit(f'Downloading Image: {name}')
        image = download_image(self, name, image_folder_path, url_override)
        log.info(f'retrieve_image: download done {name!r} null={image.isNull()}')
    return image


def download_image(self, name: str, image_folder_path: str, url_override: str = ''):
    """
    Downloads image from wiki and stores it in images folder. Returns the image.

    Parameters:
    - :param name: name of the item
    - :param image_folder_path: path to the image folder
    - :param url_override: non default image url (optional)
    """
    filepath = os.path.join(image_folder_path, get_image_file_name(name))
    if url_override == '':
        image_url = f'{WIKI_IMAGE_URL}{name.replace(" ", "_")}_icon.png'
    else:
        image_url = url_override
    image_response = requests.get(image_url)
    if image_response.ok:
        content = image_response.content
        if len(content) >= _MIN_PNG_SIZE and content[:8] == _PNG_HEADER:
            with open(filepath, 'wb') as f:
                f.write(content)
            img = QImage(filepath)
            if not img.isNull():
                return img
            log.info(f'download_image: {name!r} saved but QImage could not load it')
        else:
            log.info(f'download_image: {name!r} response OK but not a valid PNG '
                     f'(size={len(content)}, header={content[:8]!r})')
    else:
        log.info(f'download_image: {name!r} HTTP {image_response.status_code}')
    self.cache.images_failed[name] = int(datetime.now().timestamp())
    return QImage()


def get_ship_image(self, image_name: str, threaded_worker):
    """
    Tries to fetch ship image from local filesystem, downloads it otherwise. Returns the image.
    Handles special characters (apostrophes etc.) in ship names correctly.

    Parameters:
    - :image_name: filename of the image (e.g. "Ar'kala Tactical Warbird.jpg")
    - :param threaded_worker: thread object supplying signals
    """
    from urllib.parse import quote as url_quote
    image_path = os.path.join(
            self.config['config_subfolders']['ship_images'], quote_plus(image_name))
    log.debug(f'get_ship_image: {image_name!r} -> {image_path}')

    # Try loading from disk first
    image = QImage(image_path)
    if not image.isNull():
        log.debug(f'get_ship_image: loaded from disk OK size={image.width()}x{image.height()}')
        threaded_worker.result.emit((image,))
        return

    # Build wiki URL: spaces → underscores, special chars properly encoded
    encoded_name = url_quote(image_name.replace(' ', '_'), safe='._-')
    image_url = WIKI_IMAGE_URL + encoded_name
    log.info(f'get_ship_image: not on disk, downloading {image_url!r}')
    try:
        image_response = requests.get(image_url, timeout=15)
        if image_response.ok:
            raw = image_response.content
            log.info(f'get_ship_image: response {image_response.status_code} '
                     f'size={len(raw)} header={raw[:8]!r}')
            with open(image_path, 'wb') as f:
                f.write(raw)
            image = QImage(image_path)
            if image.isNull():
                log.info(f'get_ship_image: saved but QImage cannot load it '
                         f'(possibly not a supported format)')
            else:
                log.info(f'get_ship_image: download OK size={image.width()}x{image.height()}')
        else:
            log.info(f'get_ship_image: HTTP {image_response.status_code} for {image_url!r}')
    except Exception as e:
        log.info(f'get_ship_image: exception {e!r}')
    threaded_worker.result.emit((image,))


def load_image(image_name: str, image: QImage, image_folder_path: str) -> QImage:
    """
    Retrieves image from images folder and returns it. Assumes the image exists.

    Parameters:
    - :param image_name: name of the image
    - :param image: preconstructed (empty) Image
    - :param image_folder_path: path to the image folder
    """
    image_path = os.path.join(image_folder_path, get_image_file_name(image_name))
    image.load(image_path)


def image(self, image_name: str) -> QImage:
    """
    Returns image from cache if cached, loads from disk if null, downloads if missing.
    Returns N/A placeholder if image cannot be found or downloaded.

    Parameters:
    - :param image_name: name of the image
    """
    def _na():
        na = getattr(self.cache, 'na_image', None)
        return na if na is not None else (
            self.cache.empty_image if hasattr(self.cache, 'empty_image') else QImage())

    img = self.cache.images.get(image_name)
    if img is None:
        log.info(f'image: {image_name!r} not in cache at all, returning N/A')
        return _na()
    if not img.isNull():
        return img

    # Image slot is null — try loading from disk by creating a fresh QImage
    # (QImage.load() in-place is unreliable in PySide6 — always use QImage(path))
    img_folder = self.config['config_subfolders']['images']
    img_path = os.path.join(img_folder, get_image_file_name(image_name))
    if os.path.exists(img_path):
        loaded = QImage(img_path)
        if not loaded.isNull():
            self.cache.images[image_name] = loaded
            log.debug(f'image: {image_name!r} loaded from disk OK')
            return loaded

    # File not on disk — attempt download (synchronous, user-triggered action)
    log.info(f'image: {image_name!r} not on disk, downloading...')
    loaded = retrieve_image(self, image_name, img_folder)
    if not loaded.isNull():
        self.cache.images[image_name] = loaded
        log.info(f'image: {image_name!r} downloaded OK')
        return loaded

    # Download failed — return N/A placeholder
    log.info(f'image: {image_name!r} not available, returning N/A placeholder')
    return _na()


def alt_image(self, image_name: str, image_suffix: str) -> QImage:
    """
    Returns image from cache if cached, loads and returns image if not cached. If `image_suffix` is
    not empty, tries to get alternate image first.

    Parameters:
    - :param image_name: name of the image
    - :param image_suffix: suffix to check in self.cache.alt_images
    """
    if image_name + image_suffix in self.cache.alt_images:
        return image(self, self.cache.alt_images[image_name + image_suffix])
    else:
        return image(self, image_name)


def auto_backup_cargo_file(self, filename: str):
    """
    Backs up given cargo data file to the auto backups folder

    Parameters:
    - :param filename: name of the file to back up
    """
    source_path = os.path.join(self.config['config_subfolders']['cargo'], filename)
    if os.path.exists(source_path):
        target_path = os.path.join(self.config['config_subfolders']['auto_backups'], filename)
        shutil__copyfile(source_path, target_path)


# --------------------------------------------------------------------------------------------------
# static functions
# --------------------------------------------------------------------------------------------------


# Minimum valid PNG file size in bytes (PNG header 8 + IHDR chunk 25 + IEND 12 = 45)
_PNG_HEADER = b'\x89PNG\r\n\x1a\n'
_MIN_PNG_SIZE = 67  # smallest valid 1x1 PNG


def _is_valid_png(filepath: str) -> bool:
    """Returns True if file exists, is non-empty, and starts with a valid PNG header."""
    try:
        size = os.path.getsize(filepath)
        if size < _MIN_PNG_SIZE:
            return False
        with open(filepath, 'rb') as f:
            header = f.read(8)
        return header == _PNG_HEADER
    except OSError:
        return False


def get_downloaded_icons(images_dir: Path) -> set:
    """
    Returns set of image names that are present AND valid (correct PNG header, non-empty).
    Invalid/corrupt files are deleted so they will be re-downloaded.
    """
    valid = set()
    try:
        files = os.listdir(str(images_dir))
    except OSError:
        return valid
    for filename in files:
        if not filename.endswith('.png'):
            continue
        filepath = os.path.join(str(images_dir), filename)
        if _is_valid_png(filepath):
            valid.add(unquote_plus(filename)[:-4])
        else:
            # corrupt / empty / HTML error page — remove so it gets re-downloaded
            try:
                os.remove(filepath)
                log.info(f'get_downloaded_icons: removed corrupt image {filename!r}')
            except OSError:
                pass
    return valid


def create_folder(path_to_folder):
    """
    Creates the folder at path_to_folder in case it does not exist.

    Parameters:
    - :param path_to_folder: absolute path to folder
    """
    if not os.path.exists(path_to_folder) and not os.path.isdir(path_to_folder):
        os.mkdir(path_to_folder)


def delete_folder_contents(path_to_folder):
    """
    Delets all files and folders within a folder.

    Parameters:
    - :param path_to_folder: absolute path to folder
    """
    if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
        shutil__rmtree(path_to_folder)
        os.mkdir(path_to_folder)


def copy_file(source_path, target_path):
    """
    Tries to copy file from `source_path` to `target_path`

    Parameters:
    - :param source_path: file to copy
    - :param target_path: location and name of the target file
    """
    if os.path.exists(source_path) and os.path.isfile(source_path):
        shutil__copyfile(source_path, target_path)


def get_asset_path(asset_name: str, app_directory: str) -> str:
    """
    returns the absolute path to a file in the asset folder

    Parameters:
    - :param asset_name: filename of the asset
    - :param app_directory: absolute path to app directory
    """
    fp = os.path.join(app_directory, 'local', asset_name)
    if os.path.exists(fp):
        return fp
    else:
        return ''


def load_icon(filename: str, app_directory: str) -> QIcon:
    """
    Loads icon from path and returns it.

    Parameters:
    - :param path: path to icon
    - :param app_directory: absolute path to the app directory
    """
    return QIcon(get_asset_path(filename, app_directory))


def load_json__new(file_path: Path) -> dict | list | None:
    """
    Loads json from path and returns dictionary or list. Returns `None` if no data could be found.

    Parameters:
    - :param path: path to json file
    """
    try:
        with file_path.open() as json_file:
            return json__load(json_file)
    except (OSError, JSONDecodeError):
        return None


def load_json(path: str) -> dict | list:
    """
    Loads json from path and returns dictionary or list.

    Parameters:
    - :param path: absolute path to json file
    """
    if not (os.path.exists(path) and os.path.isfile(path) and os.path.isabs(path)):
        raise FileNotFoundError(f'Invalid / not absolute path: {path}')
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def store_json(data: dict | list, path: str):
    """
    Stores data to json file at path. Overwrites file at target location. Raises ValueError if path
    is not absolute.

    Paramters:
    - :param data: dictionary or list that should be stored
    - :param path: target location; must be absolute path
    """
    if not os.path.isabs(path):
        raise ValueError(f'Path to file must be absolute: {path}')
    try:
        with open(path, 'w') as file:
            json.dump(data, file)
    except OSError as e:
        sys.stdout.write(f'[Error] Data could not be saved: {e}')


def fetch_json(url: str) -> dict | list:
    """
    Fetches json from url and returns parsed object. Raises `requests.exceptions.JSONDecodeError` if
    result cannot be decoded. Raises `requests.exceptions.Timeout` or 2 download attempts failed.

    Parameters:
    - :param url: URL to file
    """
    try:
        r = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        r = requests.get(url, timeout=10)
    r.encoding = 'utf-8'
    return json.loads(compensate_json(r.text))


class _LxmlElement:
    """
    Minimal wrapper around lxml HtmlElement that matches the requests_html Element API
    used by get_boff_data(): .find(css), .html, .text
    """
    def __init__(self, el):
        self._el = el

    @property
    def html(self) -> str:
        return lxml_html.tostring(self._el, encoding='unicode')

    @property
    def text(self) -> str:
        return self._el.text_content()

    def find(self, selector: str, first: bool = False):
        try:
            sel = CSSSelector(selector)
            results = [_LxmlElement(e) for e in sel(self._el)]
        except Exception:
            results = []
        if first:
            return results[0] if results else None
        return results


def fetch_html(url: str) -> _LxmlElement:
    """
    Fetches html from url and returns an element wrapper compatible with requests_html API.
    Uses lxml + requests — no Chromium, thread-safe.
    """
    import requests as _requests
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    response = _requests.get(url, timeout=30, headers=headers)
    response.raise_for_status()
    doc = lxml_html.fromstring(response.content)
    return _LxmlElement(doc)


def sanitize_file_name(txt, chr_set='extended') -> str:
    """
    Converts txt to a valid filename.

    Parameters:
    - :param txt: The path to convert.
    - :param chr_set:
        - 'printable':    Any printable character except those disallowed on Windows/*nix.
        - 'extended':     'printable' + extended ASCII character codes 128-255
        - 'universal':    For almost *any* file system.
    """
    FILLER = '-'
    MAX_LEN = 255  # Maximum length of filename is 255 bytes in Windows and some *nix flavors.

    # Step 1: Remove excluded characters.
    BLACK_LIST = set(chr(127) + r'<>:"/\|?*')
    white_lists = {
        'universal': {'-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'},
        'printable': {chr(x) for x in range(32, 127)} - BLACK_LIST,     # 0-32, 127 are unprintable,
        'extended': {chr(x) for x in range(32, 256)} - BLACK_LIST,
    }
    white_list = white_lists[chr_set]
    result = ''.join(x if x in white_list else FILLER for x in txt)

    # Step 2: Device names, '.', and '..' are invalid filenames in Windows.
    DEVICE_NAMES = (
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6',
            'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8',
            'LPT9', 'CONIN$', 'CONOUT$', '..', '.')
    if '.' in txt:
        name, _, ext = result.rpartition('.')
        ext = f'.{ext}'
    else:
        name = result
        ext = ''
    if name in DEVICE_NAMES:
        result = f'-{result}-{ext}'

    # Step 3: Truncate long files while preserving the file extension.
    if len(result) > MAX_LEN:
        result = result[:MAX_LEN - len(ext)] + ext

    # Step 4: Windows does not allow filenames to end with '.' or ' ' or begin with ' '.
    result = result.strip()
    while len(result) > 0 and result[-1] == '.':
        result = result[:-1]

    return result


def get_image_file_name(name: str) -> str:
    """
    Converts image name to valid file name
    """
    return f'{quote_plus(name)}.png'


def open_url(url: str):
    """
    Opens URL in webbrowser
    """
    webbrowser_open(url, new=2, autoraise=True)


def open_wiki_page(page_name: str):
    """
    Converts page name to URL and opens page in webbrowser.
    """
    open_url(WIKI_URL + page_name.replace(' ', '_'))


def read_env_file(path: Path, names: list[str]) -> dict[str, str]:
    """
    Reads given `names` from env file at `path` and returns dictionary containing them.

    Parameters:
    - :param path: path to env file
    - :param names: variables to read from env file
    """
    env_variables = dict()
    if not path.exists():
        return env_variables
    with path.open(encoding='utf-8') as env_file:
        for line in env_file:
            for identifier in names:
                if line.startswith(f'{identifier}='):
                    if line[-1] == '\n':
                        env_variables[identifier] = line[len(identifier) + 1:-1]
                    else:
                        env_variables[identifier] = line[len(identifier) + 1:]
                    break
    return env_variables


def cache_cargo_data(cache_file: Path, url: str, session: requests.Session) -> bool:
    """
    Obtains cargo data from `url` and stores it. Returns `True` on success, `False` on failure.

    Parameters:
    - :param cache_file: path to file that the cargo data should be stored to
    - :param url: url to request data from
    - :param session: request session to use for the request
    """
    try:
        response = session.get(url, timeout=10)
    except requests.exceptions.Timeout:
        sys.stdout.write(f'[Error] Requesting the following URL timed out:\n[Error] {url}\n')
        return False
    if response.ok:
        response.encoding = 'utf-8'
        try:
            cargo_data = json.loads(compensate_json(response.text))
            store_json(cargo_data, str(cache_file))
            return True
        except json.JSONDecodeError:
            sys.stdout.write(
                f'[Error] Decoding the response failed for the following URL:\n[Error] {url}\n')
    return False


def download_image_session(
        session: requests.Session, name: str, image_folder_path: Path,
        failed_images: dict[str, int], image_suffix: str = '_icon.png'):
    """
    Downloads image and saves raw bytes to disk. No QImage — safe to call from any thread.
    """
    if image_suffix == '':
        filepath = image_folder_path / quote_plus(name)
    else:
        filepath = image_folder_path / get_image_file_name(name)
    image_url = WIKI_IMAGE_URL + name.replace(' ', '_') + image_suffix
    image_response = session.get(image_url)
    if image_response.ok:
        with open(filepath, 'wb') as f:
            f.write(image_response.content)
    else:
        failed_images[name] = int(datetime.now().timestamp())


def download_images_list(
        images_list: list[str], env_variables: dict[str, str], images_path: Path,
        image_suffix: str = '_icon.png') -> dict[str, int]:
    """
    """
    requests_session = requests.Session()
    if 'SETS_CF_CLEARANCE' in env_variables:
        requests_session.cookies.set_cookie(
            requests__create_cookie(name='cf_clearance', value=env_variables['SETS_CF_CLEARANCE']))
    if 'SETS_USER_AGENT' in env_variables:
        requests_session.headers['User-Agent'] = env_variables['SETS_USER_AGENT']
    failed_images = dict()
    for image_name in images_list:
        download_image_session(
            requests_session, image_name, images_path, failed_images, image_suffix)
    return failed_images


def download_images_fast(
        images_list: list[str], env_variables: dict[str, str], images_dir: Path,
        image_suffix: str = '_icon.png'):
    """
    Downloads images using multiple threads.
    """
    total_threads = 16
    image_chunk_size = len(images_list) // total_threads
    while image_chunk_size < 4 and total_threads > 1:
        total_threads -= 1
        image_chunk_size = len(images_list) // total_threads
    threads: list[ReturnValueThread] = list()
    for thread_num in range(total_threads):
        if thread_num == total_threads - 1:
            images = images_list[image_chunk_size * thread_num:]
        else:
            images = images_list[image_chunk_size * thread_num:image_chunk_size * (thread_num + 1)]
        thread = ReturnValueThread(
            target=download_images_list, args=(images, env_variables, images_dir, image_suffix))
        thread.start()
        threads.append(thread)
    failed_images = dict()
    for thread in threads:
        failed_images.update(thread.join())
    print(failed_images)
