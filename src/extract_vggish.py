"""VGGish ile embedding çıkarımı.

Giriş: 16 kHz mono. Klip 0.96 sn'lik pencerelere bölünür, her pencere 128 boyutlu
embedding verir. Şarkı vektörü = pencerelerin ortalaması ve standart sapması
(128 + 128 = 256 boyut). Ağırlıklar torch.hub ile ilk çalıştırmada iner.
"""
import numpy as np
import torch


class VggishEmbedder:
    dim = 256

    def __init__(self, device: str):
        # postprocess=False: PCA/quantize adımı yok, ham 128 boyutlu embedding
        self.model = torch.hub.load(
            "harritaylor/torchvggish",
            "vggish",
            postprocess=False,
            device=device,
            trust_repo=True,
        )
        self.model.eval()

    @torch.no_grad()
    def __call__(self, wav16k: np.ndarray) -> np.ndarray:
        frames = self.model.forward(wav16k, fs=16000).cpu().numpy()  # (N, 128)
        if frames.ndim == 1:
            frames = frames[None, :]
        return np.concatenate([frames.mean(0), frames.std(0)]).astype(np.float32)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = np.arange(16000 * 5) / 16000
    wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    print("VGGish embedding boyutu:", VggishEmbedder(dev)(wav).shape)
