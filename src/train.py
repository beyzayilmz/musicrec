#!/usr/bin/env python3
"""Dondurulmuş embedding'ler üzerine MLP eğitimi ve değerlendirme (çok etiketli).

Her iki model için aynı hiperparametreler ve aynı seed kullanılır (adil karşılaştırma).
Metrikler: macro ROC-AUC, macro PR-AUC, macro F1 (eşikler validation üzerinde etiket
başına seçilir, test'e uygulanır).

Örnek:
    python src/train.py --emb-dir /kaggle/input/musicrec-emb --model panns vggish
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from common import SPLITS, read_split, set_seed


def resolve_emb_dir(p: str) -> Path:
    p = Path(p)
    for cand in (p, p / "embeddings"):
        if (cand / "splits").exists():
            return cand
    raise SystemExit(f"{p} altında splits/ klasörü bulunamadı (pipeline.py çıktısını ver).")


def load_embeddings(emb_dir: Path, model: str):
    files = sorted((emb_dir / model).glob("*.npz"))
    if not files:
        raise SystemExit(f"{emb_dir / model} içinde .npz yok.")
    ids, xs = [], []
    for f in files:
        z = np.load(f)
        ids.append(z["ids"])
        xs.append(z["emb"])
    return np.concatenate(ids), np.concatenate(xs)


def make_xy(rows, id2idx, X, tag2idx):
    keep = [r for r in rows if r["track_id"] in id2idx]
    x = X[[id2idx[r["track_id"]] for r in keep]]
    y = np.zeros((len(keep), len(tag2idx)), dtype=np.float32)
    for i, r in enumerate(keep):
        for t in r["tags"]:
            if t in tag2idx:
                y[i, tag2idx[t]] = 1.0
    return x, y


def macro_metrics(y, p):
    pos = y.sum(0)
    valid = (pos > 0) & (pos < len(y))
    roc = roc_auc_score(y[:, valid], p[:, valid], average="macro")
    pr = average_precision_score(y[:, valid], p[:, valid], average="macro")
    return float(roc), float(pr)


def best_thresholds(y, p):
    thr = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        prec, rec, th = precision_recall_curve(y[:, j], p[:, j])
        if len(th) == 0:
            continue
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        thr[j] = th[int(np.argmax(f1))]
    return thr


@torch.no_grad()
def predict(net, x, device, bs=2048):
    net.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(torch.sigmoid(net(x[i:i + bs].to(device))).cpu())
    return torch.cat(out).numpy()


def run(model_name, args, device):
    set_seed(args.seed)
    emb_dir = resolve_emb_dir(args.emb_dir)
    vocab = json.loads((emb_dir / "vocab.json").read_text())
    tag2idx = {t: i for i, t in enumerate(vocab)}
    ids, X = load_embeddings(emb_dir, model_name)
    id2idx = {int(t): i for i, t in enumerate(ids)}

    data = {}
    for s in SPLITS:
        rows = read_split(emb_dir / "splits" / f"{s}.tsv")
        data[s] = make_xy(rows, id2idx, X, tag2idx)
        print(f"[{model_name}] {s}: {len(data[s][0])}/{len(rows)} şarkı embedding'e sahip")

    mean = data["train"][0].mean(0, keepdims=True)
    std = data["train"][0].std(0, keepdims=True) + 1e-6
    xt = {s: torch.tensor((data[s][0] - mean) / std, dtype=torch.float32) for s in SPLITS}
    yt = {s: data[s][1] for s in SPLITS}

    d, n_tags = xt["train"].shape[1], len(vocab)
    net = nn.Sequential(
        nn.Linear(d, args.hidden), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.hidden, n_tags)
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_w = None
    if args.pos_weight:
        pos = yt["train"].sum(0)
        neg = len(yt["train"]) - pos
        w = np.clip(neg / np.maximum(pos, 1), 1.0, args.pos_weight_max)
        pos_w = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    x_train = xt["train"].to(device)
    y_train = torch.tensor(yt["train"], device=device)
    g = torch.Generator().manual_seed(args.seed)
    best_pr, best_state, bad, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(x_train), generator=g).to(device)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(net(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        roc, pr = macro_metrics(yt["validation"], predict(net, xt["validation"], device))
        if pr > best_pr:
            best_pr, best_epoch, bad = pr, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{model_name}] epoch {epoch:3d} val ROC-AUC {roc:.4f} PR-AUC {pr:.4f}")
        if bad >= args.patience:
            print(f"[{model_name}] erken durdu (en iyi epoch {best_epoch})")
            break

    net.load_state_dict(best_state)
    pv, pt = predict(net, xt["validation"], device), predict(net, xt["test"], device)
    thr = best_thresholds(yt["validation"], pv)
    roc, pr = macro_metrics(yt["test"], pt)
    f1 = f1_score(yt["test"], (pt >= thr).astype(int), average="macro", zero_division=0)
    result = {"model": model_name, "roc_auc": roc, "pr_auc": pr, "f1_macro": float(f1),
              "best_epoch": best_epoch, "embedding_dim": int(d), "n_test": int(len(pt))}

    out = Path(args.out_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1))
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "vocab": vocab, "thr": thr}, out / "model.pt")
    with open(out / "per_tag.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n_test_pos", "roc_auc", "pr_auc"])
        for j, t in enumerate(vocab):
            npos = int(yt["test"][:, j].sum())
            if 0 < npos < len(pt):
                w.writerow([t, npos, f"{roc_auc_score(yt['test'][:, j], pt[:, j]):.4f}",
                            f"{average_precision_score(yt['test'][:, j], pt[:, j]):.4f}"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--model", nargs="+", default=["panns", "vggish"], choices=["panns", "vggish"])
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pos-weight", action="store_true", help="BCE'de etiket başına pos_weight kullan")
    ap.add_argument("--pos-weight-max", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [run(m, args, device) for m in args.model]
    print("\n=== Test sonuçları (macro) ===")
    print(f"{'model':8s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'F1':>8s}")
    for r in results:
        print(f"{r['model']:8s} {r['roc_auc']:8.4f} {r['pr_auc']:8.4f} {r['f1_macro']:8.4f}")
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
