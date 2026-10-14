#!/usr/bin/env python3
"""MTG-Jamendo için tar-tar embedding hattı (PANNs + VGGish, tek geçişte).

Her tar için: indir -> doğrula -> aç -> iki modelle embedding çıkar -> .npz yaz -> sesi sil.
Disk hiç dolmaz. Çıktıda var olan tar'lar atlanır, böylece Kaggle'da oturum/süre
dolsa bile kaldığı yerden devam edilir (--prev-dir ile önceki çıktıyı ver).

Çıktı düzeni (--out-dir):
    panns/<tar_adı>.npz     ids (N,), emb (N, 2048)
    vggish/<tar_adı>.npz    ids (N,), emb (N, 256)
    splits/{train,validation,test}.tsv, vocab.json, failed.txt
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import DATASET, SPLITS, ClipDataset, build_vocab, collate, read_split
from extract_panns import PannsEmbedder
from extract_vggish import VggishEmbedder

URLS = {
    "mtg-fast": "https://cdn.freesound.org/mtg-jamendo/{dataset}/{dtype}/{name}",
    "mtg": "https://essentia.upf.edu/documentation/datasets/mtg-jamendo/{dataset}/{dtype}/{name}",
}


def ensure_splits(mtg_dir: Path, split_idx: int, dataset: str) -> dict:
    d = mtg_dir / "data" / "splits" / f"split-{split_idx}"
    files = {s: d / f"{dataset}-{s}.tsv" for s in SPLITS}
    if not all(p.exists() for p in files.values()):
        print("Split dosyaları yok, split_filter_subset.py ile üretiliyor...")
        subprocess.run(
            [sys.executable, "scripts/split_filter_subset.py", "data/splits",
             "autotagging", dataset, "--category", "mood/theme"],
            cwd=mtg_dir, check=True,
        )
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        listing = sorted(x.name for x in d.glob("*")) if d.exists() else "klasör yok"
        raise SystemExit(f"Split dosyaları bulunamadı: {missing}\n{d} içeriği: {listing}")
    return files


def read_tar_list(mtg_dir: Path, dataset: str, dtype: str) -> list:
    f = mtg_dir / "data" / "download" / f"{dataset}_{dtype}_sha256_tars.txt"
    items = []
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            items.append((parts[1], parts[0]))  # (dosya adı, sha256)
    if not items:
        raise SystemExit(f"Tar listesi boş: {f}")
    return items


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha: str, retries: int = 3) -> None:
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if sha256_of(tmp) != sha:
                raise RuntimeError("sha256 uyuşmuyor")
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  indirme hatası ({attempt}/{retries}): {e}")
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def safe_extract(tar_path: Path, dest: Path) -> None:
    """Yol kaçışı ve link içeren üyeleri atlayarak açar."""
    dest = dest.resolve()
    with tarfile.open(tar_path) as tar:
        members = []
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if m.issym() or m.islnk() or not str(target).startswith(str(dest)):
                continue
            members.append(m)
        tar.extractall(dest, members=members)


def save_npz(path: Path, ids, emb) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.asarray(ids, dtype=np.int64), emb=np.asarray(emb, dtype=np.float32))
    os.replace(tmp, path)


def restore_prev(prev_dir, out: Path) -> None:
    if not prev_dir:
        return
    n = 0
    for f in Path(prev_dir).rglob("*.npz"):
        if f.parent.name in ("panns", "vggish"):
            dst = out / f.parent.name / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                n += 1
    print(f"Önceki çıktıdan {n} dosya geri yüklendi.")


def process_audio(audio_dir: Path, rows: dict, panns, vgg, args):
    items = []
    for p in sorted(audio_dir.rglob("*.mp3")):
        try:
            tid = int(p.name.split(".")[0])
        except ValueError:
            continue
        if tid in rows:
            items.append((tid, p, rows[tid]["duration"]))
    loader = DataLoader(
        ClipDataset(items), batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate, shuffle=False,
    )
    ids, pe, ve, failed = [], [], [], []
    for tids, w32, w16s, oks in tqdm(loader, desc="  embedding", leave=False):
        batch_p = panns(w32)
        for i, tid in enumerate(tids):
            if not oks[i]:
                failed.append(tid)
                continue
            try:
                v = vgg(w16s[i])
            except Exception:  # noqa: BLE001
                failed.append(tid)
                continue
            ids.append(tid)
            pe.append(batch_p[i])
            ve.append(v)
    return ids, pe, ve, failed, len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtg-dir", required=True, help="klonlanmış mtg-jamendo-dataset reposu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prev-dir", default=None, help="önceki turun çıktısı (devam için)")
    default_work = "/kaggle/temp" if Path("/kaggle/temp").exists() else "work"
    ap.add_argument("--work-dir", default=default_work)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--type", default="audio-low", dest="dtype")
    ap.add_argument("--source", default="mtg-fast", choices=list(URLS))
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=10.5, help="süre bütçesi; dolunca temiz çıkar")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-tars", type=int, default=0, help="test için ilk N tar")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", device)
    mtg, out, work = Path(args.mtg_dir), Path(args.out_dir), Path(args.work_dir) / "musicrec"
    for m in ("panns", "vggish", "splits"):
        (out / m).mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    restore_prev(args.prev_dir, out)

    files = ensure_splits(mtg, args.split, args.dataset)
    rows_by_split = {s: read_split(p) for s, p in files.items()}
    rows = {r["track_id"]: r for rs in rows_by_split.values() for r in rs}
    for s, p in files.items():
        shutil.copy2(p, out / "splits" / f"{s}.tsv")
    vocab = build_vocab(rows_by_split)
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1))
    print(f"{len(rows)} şarkı, {len(vocab)} etiket.")

    tars = read_tar_list(mtg, args.dataset, args.dtype)
    if args.limit_tars:
        tars = tars[: args.limit_tars]

    panns, vgg = PannsEmbedder(device), VggishEmbedder(device)
    last, finished = 0.0, 0
    for k, (name, sha) in enumerate(tars, 1):
        stem = name.removesuffix(".tar")
        if all((out / m / f"{stem}.npz").exists() for m in ("panns", "vggish")):
            finished += 1
            print(f"[{k}/{len(tars)}] {name}: zaten var, atlandı")
            continue
        if time.time() - t0 + last * 1.1 > args.max_hours * 3600:
            print("Süre bütçesi doldu, temiz çıkılıyor. Kalan tar'lar için bu çıktıyı --prev-dir ile ver.")
            break
        ts = time.time()
        tar_path = work / name
        audio_dir = work / f"audio_{stem}"
        try:
            print(f"[{k}/{len(tars)}] {name}: indiriliyor...")
            url = URLS[args.source].format(dataset=args.dataset, dtype=args.dtype, name=name)
            download(url, tar_path, sha)
            audio_dir.mkdir(exist_ok=True)
            safe_extract(tar_path, audio_dir)
            tar_path.unlink(missing_ok=True)
            ids, pe, ve, failed, total = process_audio(audio_dir, rows, panns, vgg, args)
            save_npz(out / "panns" / f"{stem}.npz", ids, pe)
            save_npz(out / "vggish" / f"{stem}.npz", ids, ve)
            if failed:
                with open(out / "failed.txt", "a") as f:
                    f.write("\n".join(map(str, failed)) + "\n")
            finished += 1
            last = time.time() - ts
            print(f"  {len(ids)}/{total} şarkı, {len(failed)} hata, {last / 60:.1f} dk, "
                  f"toplam geçen {(time.time() - t0) / 3600:.2f} sa")
        finally:
            tar_path.unlink(missing_ok=True)
            shutil.rmtree(audio_dir, ignore_errors=True)

    print(f"Bitti: {finished}/{len(tars)} tar tamam.")
    if finished == len(tars):
        print("TAMAM: tüm embedding'ler hazır.")


if __name__ == "__main__":
    main()
