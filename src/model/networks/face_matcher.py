import torch
import torch.nn.functional as F

class FaceMatcher:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def match(self, emb1, emb2):
        similarity = F.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()

        return 1 if similarity >= self.threshold else 0, similarity

    