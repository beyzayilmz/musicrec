"""Log-mel spektrogram için sıfırdan eğitilen ResNet-18/34 (önceden eğitilmiş ağırlık YOK).

Standart ResNet-34 blok düzeni ([3, 4, 6, 3] BasicBlock, kanallar w-2w-4w-8w) korunur; yalnızca
giriş ve çıkış ses verisine uyarlanır:
- Giriş: (B, 1, mel, kare). Mel bantları ayrı ayrı normalize edilir (PANNs'taki bn0 gibi).
- Gövde (stem): ImageNet'teki 7x7/2 + maxpool yerine iki 3x3 konvolüsyon + 2x2 ortalama havuz.
  64 mel bandı ImageNet görüntüsünden çok küçük; agresif stem frekans çözünürlüğünü erken yok eder.
- Çıkış: frekans ekseni ortalanır, zaman ekseninde ortalama + max havuzlama -> 16w boyutlu embedding
  (w=64 için 1024). Bu vektör öneri sistemi için şarkı vektörü olarak kullanılır.
- Düzenlileştirme: stochastic depth (drop path, derinlikle doğrusal artar), her artık bloğun son
  BN'i sıfırla başlatılır (blok başlangıçta özdeşlik gibi davranır, sıfırdan eğitimi kararlı kılar).

w=64, 64 mel: ResNet-34 ~21.3 M, ResNet-18 ~11.2 M parametre.
"""
import torch
import torch.nn as nn

LAYERS = {"resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3)}


def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, dp):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.bn2.weight)
        self.act = nn.ReLU(inplace=True)
        self.dp = dp
        self.short = None
        if stride != 1 or cin != cout:
            # ResNet-D kısayolu: önce ortalama havuz, sonra 1x1 konv (stride'lı 1x1 bilginin 3/4'ünü atar)
            self.short = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True) if stride != 1 else nn.Identity(),
                nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        s = x if self.short is None else self.short(x)
        return self.act(s + drop_path(h, self.dp, self.training))


class AudioResNet(nn.Module):
    def __init__(self, n_tags, arch="resnet34", n_mels=64, width=64, drop_path_rate=0.1, dropout=0.5):
        super().__init__()
        w = width
        self.bn_mel = nn.BatchNorm2d(n_mels)
        self.stem = nn.Sequential(
            nn.Conv2d(1, w // 2, 3, 1, 1, bias=False), nn.BatchNorm2d(w // 2), nn.ReLU(inplace=True),
            nn.Conv2d(w // 2, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        depths = LAYERS[arch]
        total = sum(depths)
        blocks, cin, k = [], w, 0
        for stage, (n, cout) in enumerate(zip(depths, (w, 2 * w, 4 * w, 8 * w))):
            for i in range(n):
                stride = 2 if (i == 0 and stage > 0) else 1
                blocks.append(BasicBlock(cin, cout, stride, drop_path_rate * k / max(1, total - 1)))
                cin, k = cout, k + 1
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(16 * w, n_tags)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def embed(self, x):  # x: (B, 1, mel, kare)
        x = self.bn_mel(x.transpose(1, 2)).transpose(1, 2)
        h = self.blocks(self.stem(x))
        h = h.mean(2)  # frekans
        return torch.cat([h.mean(2), h.amax(2)], dim=1)  # zaman: ortalama + max -> (B, 16w)

    def forward(self, x, return_emb=False):
        e = self.embed(x)
        out = self.fc(self.drop(e))
        return (out, e) if return_emb else out


class EMA:
    """Ağırlıkların üstel hareketli ortalaması. Sıfırdan eğitimde değerlendirmeyi belirgin biçimde
    kararlı kılar ve genelde son/epoch-en-iyi ağırlıklardan daha iyi genelleşir. Tamponlar (BN
    istatistikleri) doğrudan kopyalanır."""

    def __init__(self, net, decay):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone().float() for k, v in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))  # başta hızlı ısınma
        for k, v in net.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}
