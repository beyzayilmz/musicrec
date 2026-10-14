"""PANNs (Cnn14) ile 2048 boyutlu embedding çıkarımı.

Giriş: 32 kHz mono, (batch, örnek) biçiminde. Checkpoint (Cnn14_mAP=0.431.pth)
ilk çalıştırmada ~/panns_data/ altına iner.
"""
import numpy as np
import torch


class PannsEmbedder:
    dim = 2048

    def __init__(self, device: str):
        from panns_inference import AudioTagging

        self.tagger = AudioTagging(checkpoint_path=None, device=device)

    @torch.no_grad()
    def __call__(self, wav32k: np.ndarray) -> np.ndarray:
        _, embedding = self.tagger.inference(wav32k)
        return np.asarray(embedding, dtype=np.float32)


if __name__ == "__main__":
    # Hızlı duman testi: 1 sn sessizlikten embedding al
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = PannsEmbedder(dev)(np.zeros((1, 32000), dtype=np.float32))
    print("PANNs embedding boyutu:", emb.shape)
