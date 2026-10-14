"""Ortak yardımcılar: TSV okuma, etiket sözlüğü, ses klibi yükleme, Dataset."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SEED = 42
CLIP_SEC = 30
SR_PANNS = 32000
SR_VGGISH = 16000
DATASET = "autotagging_moodtheme"
SPLITS = ("train", "validation", "test")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tag_name(tag: str) -> str:
    """'mood/theme---happy' -> 'happy'"""
    return tag.split("---", 1)[-1]


def read_split(tsv_path):
    """MTG-Jamendo TSV'sini okur.

    Sütunlar: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION, TAGS...
    Etiket sayısı satıra göre değiştiği için 5. sütundan sonrası etiket kabul edilir.
    """
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # başlık satırı
        for r in reader:
            if len(r) < 5:
                continue
            rows.append(
                {
                    "track_id": int(r[0].split("_")[-1]),
                    "path": r[3],
                    "duration": float(r[4]),
                    "tags": [tag_name(t) for t in r[5:] if t],
                }
            )
    return rows


def build_vocab(rows_by_split) -> list:
    return sorted({t for rows in rows_by_split.values() for r in rows for t in r["tags"]})


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Klibi n örneğe getirir: kısaysa tekrarlar (sessizlikle seyreltmez), uzunsa keser."""
    if len(y) == 0:
        raise ValueError("boş ses")
    if len(y) >= n:
        return y[:n]
    return np.resize(y, n)


def load_clip(path, duration: float) -> dict:
    """Şarkının ortasından CLIP_SEC saniyelik kesiti bir kez çözer, iki örnekleme hızında döndürür."""
    import librosa

    offset = max(0.0, duration / 2 - CLIP_SEC / 2)
    y, sr = librosa.load(str(path), sr=None, mono=True, offset=offset, duration=CLIP_SEC)
    out = {}
    for target in (SR_PANNS, SR_VGGISH):
        out[target] = y if sr == target else librosa.resample(y, orig_sr=sr, target_sr=target)
    return out


class ClipDataset(Dataset):
    """items: (track_id, dosya_yolu, süre) listesi."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tid, path, dur = self.items[i]
        try:
            clips = load_clip(path, dur)
            w32 = fit_length(clips[SR_PANNS], SR_PANNS * CLIP_SEC)
            w16 = clips[SR_VGGISH]
            return tid, w32.astype(np.float32), w16.astype(np.float32), True
        except Exception:
            return (
                tid,
                np.zeros(SR_PANNS * CLIP_SEC, dtype=np.float32),
                np.zeros(SR_VGGISH, dtype=np.float32),
                False,
            )


def collate(batch):
    tids = [b[0] for b in batch]
    w32 = np.stack([b[1] for b in batch])
    w16 = [b[2] for b in batch]
    oks = [b[3] for b in batch]
    return tids, w32, w16, oks
