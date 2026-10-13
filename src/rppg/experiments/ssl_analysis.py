"""Embedding visualization and positive/negative contrastive similarity diagnostics."""
import numpy as np


def contrastive_similarity(z1, z2):
    from sklearn.metrics.pairwise import cosine_similarity
    a, b = np.asarray(z1), np.asarray(z2)
    positive = np.diag(cosine_similarity(a, b))
    all_scores = cosine_similarity(a, b)
    negative = all_scores[~np.eye(len(a), dtype=bool)]
    return {"positive": positive, "negative": negative}


def embed_tsne(features, seed=42, kmeans_clusters=6):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    values = np.asarray(features)
    if values.ndim != 2 or len(values) < kmeans_clusters + 1:
        raise ValueError("Need a nontrivial 2D embedding matrix")
    perplexity = min(10, len(values) - 1)
    embedding = TSNE(n_components=2, perplexity=perplexity,
                     random_state=seed, init="pca", learning_rate="auto").fit_transform(values)
    labels = KMeans(n_clusters=kmeans_clusters, random_state=seed, n_init=10).fit_predict(values)
    return embedding, labels
