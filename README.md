# musicrec

MTG-Jamendo mood/theme veri setinde VGGish ve PANNs (Cnn14) embedding'lerini karşılaştırma hattı. Modeller dondurulmuş öznitelik çıkarıcı olarak kullanılır, üzerine aynı MLP eğitilir.

## Yapı

```
src/common.py          TSV okuma, etiket sözlüğü, ses yükleme
src/extract_panns.py   PANNs Cnn14 -> 2048 boyut
src/extract_vggish.py  VGGish -> 256 boyut (pencere ortalaması + std)
src/pipeline.py        tar-tar indir, embedding çıkar, .npz yaz, sesi sil (devam edebilir)
src/train.py           MLP eğitimi, macro ROC-AUC / PR-AUC / F1
kaggle/                Kaggle notebook'ları (embedding çıkarımı, eğitim)
```

## Kaggle'da çalıştırma

1. Repoyu GitHub'a push et (aşağıda), `kaggle/embeddings_notebook.ipynb` dosyasındaki `REPO_URL` satırını kendi repona göre düzenle.
2. kaggle.com > New Notebook > File > Import Notebook ile `kaggle/embeddings_notebook.ipynb` dosyasını yükle.
3. Sağ panel: Accelerator **GPU T4 x2** (veya P100), Internet **On** (telefon doğrulaması gerekir).
4. Önce `--limit-tars 2` ile kısa bir deneme koş, süreyi ölç. Sonra o satırı kaldırıp **Save Version > Save & Run All (Commit)** ile çalıştır. Commit modu tarayıcı kapalıyken en fazla 12 saat koşar.
5. Bittiğinde Output sekmesinden `embeddings` klasörünü bir Kaggle Dataset olarak kaydet.
6. Süre yetmezse: bu dataset'i yeni commit'e Input olarak ekle, notebook'taki `--prev-dir` satırının yorumunu kaldır. Biten tar'lar atlanır.
7. `kaggle/train_notebook.ipynb` ile eğitimi koş (GPU gerekmez, dakikalar sürer). Input olarak embedding dataset'ini ekle.

## Yerelde

```bash
pip install -r requirements.txt
git clone https://github.com/MTG/mtg-jamendo-dataset /data/mtg
python src/pipeline.py --mtg-dir /data/mtg --out-dir embeddings --work-dir /data/work
python src/train.py --emb-dir embeddings --model panns vggish
```

## Notlar

- Her şarkıdan ortadaki 30 saniye alınır ve bir kez çözülür. 32 kHz (PANNs) ve 16 kHz (VGGish) kopyaları bellekte üretilir.
- Hazır split-0 dosyaları kullanılır, kendin bölme. Dosyalar yoksa `pipeline.py` `split_filter_subset.py` ile üretir.
- Eşikler validation üzerinde etiket başına F1'e göre seçilir, test'e uygulanır.
- Etiketler dengesiz. Düşük kalan etiketler için `train.py --pos-weight` dene.
- `ImportError` veya numpy sürüm uyuşmazlığı çıkarsa pip kurulumundan sonra çekirdeği (kernel) yeniden başlat.
- Bozuk dosyaların id'leri `embeddings/failed.txt` içine yazılır.
