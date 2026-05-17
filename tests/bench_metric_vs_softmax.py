#!/usr/bin/env python3
"""
bench_metric_vs_softmax.py — head-to-head benchmark of softmax classifier
vs ArcFace metric-learning embedder on confirmed training crops.

Methodology:
  • Walk warp/training_data/crops/ — filename encodes the GT label.
  • Stratified 80/20 train/test split per class (seed=42 for reproducibility).
  • Softmax: load icon_classifier.pt, classify each test crop directly.
  • Metric: load icon_embedder.pt + embedding_index.npz, k-NN against the
    train-fold portion of the gallery (test embeddings excluded — fair).
  • Report: top-1 accuracy overall + per class-frequency bucket + BOFF
    science confusion matrix (Jam Sensors vs neighbours).

Usage:
  python tests/bench_metric_vs_softmax.py
  python tests/bench_metric_vs_softmax.py --crops <dir>  # override crop dir
  python tests/bench_metric_vs_softmax.py --md results.md  # save markdown
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CROP_DIR_DEFAULT = ROOT / 'warp' / 'training_data' / 'crops'
MODELS_DIR       = ROOT / 'warp' / 'models'
TEST_FRAC        = 0.2
SEED             = 42

# Filename convention: <prefix>__<item_name>__<hash>.png — extract item_name.
# Some item names contain underscores; the trailing 12-hex-char hash + .png is
# the stable boundary.
NAME_RE = re.compile(r'^([^_]+(?:_[^_]+)*?)__(.+)__[0-9a-f]+\.png$', re.IGNORECASE)


def _label_from_filename(path: Path) -> str | None:
    """Extract the item-name component from a crop filename."""
    m = NAME_RE.match(path.name)
    return m.group(2) if m else None


def _norm(s: str) -> str:
    """Canonical form for label comparison. Softmax label_map.json holds
    pretty names ('Jam Targeting Sensors'); local crops use snake_case
    ('jam_targeting_sensors'). Normalize both sides before comparing."""
    return s.lower().strip().replace(' ', '_').rstrip('_') if s else s


def _load_crops(crop_dir: Path) -> tuple[list[Path], list[str]]:
    paths, labels = [], []
    for p in sorted(crop_dir.glob('*.png')):
        lbl = _label_from_filename(p)
        if lbl:
            paths.append(p)
            labels.append(lbl)
    return paths, labels


def _stratified_split(labels: list[str]) -> tuple[list[int], list[int]]:
    """Per-class 80/20 split. Classes with 1 sample go to train only."""
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_cls[lbl].append(i)
    rng = random.Random(SEED)
    train_idx, test_idx = [], []
    for lbl, idxs in by_cls.items():
        rng.shuffle(idxs)
        if len(idxs) < 2:
            train_idx.extend(idxs)
        else:
            n_test = max(1, int(round(len(idxs) * TEST_FRAC)))
            test_idx.extend(idxs[:n_test])
            train_idx.extend(idxs[n_test:])
    return train_idx, test_idx


# ── Preprocessing (must match training) ─────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    import cv2
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return np.transpose(x, (2, 0, 1))[None, ...]


# ── Softmax classifier ──────────────────────────────────────────────────────

def _load_softmax():
    pt = MODELS_DIR / 'icon_classifier.pt'
    lm = MODELS_DIR / 'label_map.json'
    if not (pt.exists() and lm.exists()):
        return None, None
    import torch
    import torch.nn as nn
    from torchvision.models import efficientnet_b0
    with open(lm, encoding='utf-8') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    n_classes = len(label_map)
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)
    model.load_state_dict(torch.load(str(pt), map_location='cpu', weights_only=True))
    model.eval()
    return model, label_map


def _classify_softmax(model, label_map, bgr) -> tuple[str, float]:
    import torch
    inp = torch.from_numpy(_preprocess(bgr))
    with torch.no_grad():
        out = model(inp)[0].numpy()
    probs = np.exp(out - out.max()); probs /= probs.sum()
    top = int(np.argmax(probs))
    return label_map.get(top, ''), float(probs[top])


# ── Metric embedder ─────────────────────────────────────────────────────────

def _load_embedder():
    """Load icon_embedder.pt and reconstruct a fresh gallery from the train
    fold ONLY (test crops are excluded from the gallery to avoid leakage)."""
    emb = MODELS_DIR / 'icon_embedder.pt'
    lm  = MODELS_DIR / 'embedder_label_map.json'
    if not (emb.exists() and lm.exists()):
        return None, None
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import efficientnet_b0
    with open(lm, encoding='utf-8') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    # Match architecture in admin_train_metric.py
    backbone = efficientnet_b0(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    # Embed dim is fixed at 256 in admin_train_metric.py
    embed_dim = 256

    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.proj = nn.Linear(in_features, embed_dim)
        def forward(self, x):
            return F.normalize(self.proj(self.backbone(x)), dim=1)

    model = Embedder()
    model.load_state_dict(torch.load(str(emb), map_location='cpu', weights_only=True))
    model.eval()
    return model, label_map


def _embed_batch(model, bgrs: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    import torch
    out = []
    for i in range(0, len(bgrs), batch_size):
        batch = np.concatenate([_preprocess(b) for b in bgrs[i:i+batch_size]], axis=0)
        with torch.no_grad():
            e = model(torch.from_numpy(batch)).numpy()
        out.append(e)
    return np.concatenate(out, axis=0).astype('float32') if out else np.zeros((0, 256), 'float32')


# ── Benchmark runner ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crops', type=str, default=str(CROP_DIR_DEFAULT))
    parser.add_argument('--md', type=str, default=None,
                        help='Save markdown report to this path')
    args = parser.parse_args()

    import cv2

    crop_dir = Path(args.crops)
    paths, labels = _load_crops(crop_dir)
    print(f'Loaded {len(paths)} crops from {crop_dir}')
    if not paths:
        print('No crops found — aborting.')
        return

    cls_counts = Counter(labels)
    print(f'{len(cls_counts)} unique classes')

    train_idx, test_idx = _stratified_split(labels)
    print(f'Split: {len(train_idx)} train, {len(test_idx)} test '
          f'(classes with 1 sample → train-only, not in test)')

    test_labels = [labels[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]

    softmax_model, softmax_map = _load_softmax()
    embedder, embed_map = _load_embedder()

    if softmax_model is None and embedder is None:
        print('ERROR: neither icon_classifier.pt nor icon_embedder.pt found in '
              f'{MODELS_DIR}. Train at least one model first.')
        return

    # ── Build gallery for metric (train fold only) ──────────────────────────
    gallery_emb = gallery_lbl = None
    embed_label_to_idx = None
    if embedder is not None:
        embed_label_to_idx = {_norm(v): k for k, v in embed_map.items()}
        print('\nEmbedding train fold for k-NN gallery...')
        train_imgs = [cv2.imread(str(paths[i])) for i in train_idx]
        train_imgs = [im for im in train_imgs if im is not None]
        gallery_emb = _embed_batch(embedder, train_imgs)
        gallery_lbl = np.array([embed_label_to_idx.get(_norm(l), -1) for l in train_labels],
                                dtype=np.int32)
        unresolved = int((gallery_lbl == -1).sum())
        if unresolved:
            print(f'  WARNING: {unresolved}/{len(gallery_lbl)} train labels not in '
                  f'embedder label_map (will be excluded from k-NN matches)')
        print(f'  Gallery: {len(gallery_emb)} embeddings, {gallery_emb.shape[1]}-d')

    # ── Run test set through both models ────────────────────────────────────
    print(f'\nRunning {len(test_idx)} test crops through both models...')
    soft_pred, soft_conf = [], []
    metr_pred, metr_conf = [], []
    test_imgs = []
    for i, ti in enumerate(test_idx):
        bgr = cv2.imread(str(paths[ti]))
        test_imgs.append(bgr)
        if softmax_model is not None:
            n, c = _classify_softmax(softmax_model, softmax_map, bgr)
        else:
            n, c = '', 0.0
        soft_pred.append(n); soft_conf.append(c)

    if embedder is not None and len(test_imgs) > 0:
        print('  Embedding test fold...')
        test_emb = _embed_batch(embedder, test_imgs)
        sims = test_emb @ gallery_emb.T  # (T, G)
        top = np.argmax(sims, axis=1)
        for i, t in enumerate(top):
            lbl_idx = int(gallery_lbl[t])
            metr_pred.append(embed_map.get(lbl_idx, ''))
            metr_conf.append(float(max(0.0, min(1.0, sims[i, t]))))
    else:
        metr_pred = [''] * len(test_idx)
        metr_conf = [0.0] * len(test_idx)

    # ── Accuracy ────────────────────────────────────────────────────────────
    def _bucket(n: int) -> str:
        if n == 1:  return '1'
        if n <= 3:  return '2-3'
        if n <= 10: return '4-10'
        return '11+'

    overall = {'soft': 0, 'metr': 0, 'n': 0}
    by_bucket: dict[str, dict] = defaultdict(lambda: {'soft': 0, 'metr': 0, 'n': 0})
    for i, true in enumerate(test_labels):
        b = _bucket(cls_counts[true])
        by_bucket[b]['n'] += 1
        overall['n'] += 1
        t = _norm(true)
        if _norm(soft_pred[i]) == t:
            overall['soft'] += 1
            by_bucket[b]['soft'] += 1
        if _norm(metr_pred[i]) == t:
            overall['metr'] += 1
            by_bucket[b]['metr'] += 1

    # ── BOFF science abilities confusion ────────────────────────────────────
    science_pairs = [
        'jam_targeting_sensors', 'tractor_beam', 'gravity_well',
        'photonic_officer', 'hazard_emitters', 'science_team',
    ]
    boff_rows = []
    for true in science_pairs:
        true_indices = [i for i, l in enumerate(test_labels) if l == true]
        if not true_indices:
            continue
        t = _norm(true)
        s_right = sum(1 for i in true_indices if _norm(soft_pred[i]) == t)
        m_right = sum(1 for i in true_indices if _norm(metr_pred[i]) == t)
        boff_rows.append((true, len(true_indices), s_right, m_right))

    # ── Print + optional MD ─────────────────────────────────────────────────
    lines = []
    lines.append('# Benchmark: softmax vs ArcFace metric-learning\n')
    lines.append(f'Crops: {len(paths)} ({len(cls_counts)} classes) — '
                 f'test fold: {overall["n"]} samples\n')
    lines.append('## Overall top-1 accuracy\n')
    lines.append('| Model    | Correct | Total | Accuracy |')
    lines.append('|----------|---------|-------|----------|')
    lines.append(f'| softmax  | {overall["soft"]:>7} | {overall["n"]:>5} | '
                 f'{overall["soft"]/overall["n"]:.1%} |')
    lines.append(f'| metric   | {overall["metr"]:>7} | {overall["n"]:>5} | '
                 f'{overall["metr"]/overall["n"]:.1%} |')
    lines.append('')
    lines.append('## Per class-frequency bucket\n')
    lines.append('| Bucket (crops/class) | N test | softmax | metric |')
    lines.append('|----------------------|--------|---------|--------|')
    for b in ['1', '2-3', '4-10', '11+']:
        d = by_bucket[b]
        if d['n'] == 0:
            continue
        s = d['soft']/d['n'] if d['n'] else 0
        m = d['metr']/d['n'] if d['n'] else 0
        lines.append(f'| {b:<20} | {d["n"]:>6} | {s:>6.1%} | {m:>6.1%} |')
    lines.append('')
    lines.append('## BOFF science abilities (confusable cluster)\n')
    lines.append('| Item | N test | softmax ✓ | metric ✓ |')
    lines.append('|------|--------|-----------|----------|')
    for true, n, s, m in boff_rows:
        lines.append(f'| {true:<30} | {n:>6} | {s:>9} | {m:>8} |')
    lines.append('')

    text = '\n'.join(lines)
    print('\n' + text)
    if args.md:
        Path(args.md).write_text(text, encoding='utf-8')
        print(f'\nMarkdown report saved to {args.md}')


if __name__ == '__main__':
    main()
