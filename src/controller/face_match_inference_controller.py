import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def match_faces(img1, img2, extractor, matcher):

    emb1 = extractor.extract_embedding(img1)
    emb2 = extractor.extract_embedding(img2)

    match, similarity = matcher.match(emb1, emb2)

    return {
        "match": match,          # 1 = same person, 0 = different
        "similarity": similarity
    }
