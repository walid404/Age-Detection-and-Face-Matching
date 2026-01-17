import torch
import torch.nn.functional as F

class FaceMatcher:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def match(self, emb1: torch.Tensor, emb2: torch.Tensor):
        """
        Supports:
          - (D,) vs (D,)
          - (B, D) vs (B, D)
        """

        if emb1.dim() == 1:
            similarity = F.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()
            return int(similarity >= self.threshold), similarity

        similarity = F.cosine_similarity(emb1, emb2, dim=1)
        matches = (similarity >= self.threshold).int()

        return matches.cpu().tolist(), similarity.cpu().tolist()
