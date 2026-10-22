# Deney sonuçları (MTG-Jamendo mood/theme, 56 etiket)

Tarih: 2026-10-22/23. Dondurulmuş embedding + MLP. Test: 4231 şarkı, train 9949, validation 3802.

## Ortak ayarlar

- Embedding: 30 sn pencere, şarkı başına en fazla 20 pencere. PANNs mean (2048), VGGish mean+std (256), concat = ikisi yan yana (2304).
- MLP: tek gizli katman, AdamW, BCE, erken durdurma validation PR-AUC (patience 10), eşikler validation'da etiket başına F1'e göre.
- Metrikler: macro ROC-AUC, PR-AUC, F1 (test'te pozitifi olan tüm etiketler). P@k: test şarkıları arasında kosinüs benzerliğiyle en yakın k komşudan kaçı sorguyla en az bir etiketi paylaşıyor (ham embedding üzerinde, eğitilmiş modelden bağımsız).
- Rastgele baseline: PR-AUC 0.0319 (ortalama prevalence), P@k 0.0865.

## Koşu 1: varsayılan (hidden 512, dropout 0.3, wd 1e-4, lr 1e-3), 5 seed

| model | ROC-AUC | PR-AUC | F1 | P@5 | P@10 |
|---|---|---|---|---|---|
| panns | 0.7480 ± 0.0004 | 0.1344 ± 0.0013 | 0.1628 ± 0.0037 | 0.4044 | 0.3631 |
| vggish | 0.7610 ± 0.0012 | 0.1365 ± 0.0019 | 0.1676 ± 0.0029 | 0.4134 | 0.3693 |
| concat | 0.7580 ± 0.0013 | 0.1419 ± 0.0010 | 0.1715 ± 0.0024 | 0.4238 | 0.3831 |

## Koşu 2: + --pos-weight, 5 seed

| model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| panns | 0.7573 ± 0.0006 | 0.1368 ± 0.0034 | 0.1708 ± 0.0037 |
| vggish | 0.7669 ± 0.0007 | 0.1424 ± 0.0011 | 0.1727 ± 0.0034 |
| concat | 0.7637 ± 0.0016 | 0.1454 ± 0.0037 | 0.1768 ± 0.0030 |

## Koşu 3: + düzenlileştirme (hidden 256, dropout 0.5, wd 1e-2, lr 3e-4, pos-weight), 3 seed

| model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| panns | 0.7602 ± 0.0011 | 0.1406 ± 0.0017 | 0.1712 ± 0.0023 |
| vggish | 0.7682 ± 0.0007 | 0.1425 ± 0.0003 | 0.1713 ± 0.0038 |
| concat | 0.7659 ± 0.0018 | 0.1465 ± 0.0003 | 0.1789 ± 0.0006 |

## Gözlemler

- pos-weight her modelde tutarlı küçük kazanç verdi (ROC-AUC +0.006-0.009, PR-AUC +0.002-0.006, F1 +0.005-0.008). Düzenlileştirme ek olarak yalnızca PANNs'ta PR-AUC'yi biraz artırdı (+0.004), gerisi seed gürültüsü içinde.
- Sıralama her koşuda aynı: concat PR-AUC ve F1'de, VGGish ROC-AUC'de önde, PANNs en geride. PR-AUC rastgele baseline'ın yaklaşık 4,5 katı.
- Etiket bazında (PANNs, seed 1, koşu 1): deep (ROC 0.935, PR 0.63), summer, children, christmas, epic, corporate, trailer iyi öğreniliyor. nature, slow, movie, travel, drama, background, documentary, hopeful, inspiring, emotional ROC-AUC 0.55-0.65 civarı, şansa yakın. Öznel duygu ve kullanım bağlamı etiketleri zayıf.
- Eşikler validation'da seçildiği için nadir etiketlerde test'e aktarılamıyor (ör. dramatic: F1 0.0000).
- --min-pos 10 etkisiz kaldı, test'te her etiketin en az 10 pozitifi var.
- F1 maskesi (test'te pozitifi olmayan etiketleri dışarıda bırakma) sonucu neredeyse hiç değiştirmedi.
- P@k seed'e bağlı değil, çünkü ham embedding üzerinde hesaplanıyor.

## Komutlar

```
python src/train.py --emb-dir $emb --model panns vggish concat --seeds 1 2 3 4 5 --out-dir runs_v2 --min-pos 10
python src/train.py --emb-dir $emb --model panns vggish concat --seeds 1 2 3 4 5 --out-dir runs_v2_pw --pos-weight
python src/train.py --emb-dir $emb --model panns vggish concat --seeds 1 2 3 --out-dir runs_v3 --pos-weight --hidden 256 --dropout 0.5 --weight-decay 1e-2 --lr 3e-4
```
