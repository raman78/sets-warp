import sys
print('[SETS] src/__init__.py: start', flush=True)
print('[SETS] src/__init__.py: importing .app', flush=True)
from .app import SETS
print('[SETS] src/__init__.py: .app imported OK', flush=True)

__all__ = [
    'app', 'buildupdater', 'callbacks', 'constants', 'datafunctions', 'iofunc', 'splash', 'style',
    'subwindows', 'textedit', 'widgetbuilder', 'widgets'
]
