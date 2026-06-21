"""
Superpixel Clustering Script for Habitat Analysis

This script performs dimensionality reduction and clustering on superpixel features
to identify distinct habitat patterns within tumor regions.

Key features:
1. Uses t-SNE for dimensionality reduction
2. Implements KMeans clustering algorithm
3. Evaluates clustering quality using Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index
4. Automatically selects optimal number of clusters based on Silhouette Score
5. Generates visualizations of clustering results
6. Saves clustering labels, evaluation metrics, and configuration

Dependencies:
- numpy
- matplotlib
- sklearn
- umap-learn

Usage:
Import this module and call the cluster_and_evaluate function with appropriate parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def cluster_and_evaluate(X, idx_count, k_range=range(2, 11), output_dir=None):
    X_scaled = StandardScaler().fit_transform(X)
    
    X_embed = TSNE(n_components=2, perplexity=50, learning_rate='auto',
                   init='pca', random_state=42).fit_transform(X_scaled)

    sil_scores, ch_scores, db_scores = [], [], []
    all_labels = {}

    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_embed)
        sil = silhouette_score(X_embed, labels)
        ch = calinski_harabasz_score(X_embed, labels)
        db = davies_bouldin_score(X_embed, labels)

        sil_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        all_labels[k] = labels

    best_k = k_range[np.argmax(sil_scores)]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(k_range, sil_scores, label='Sil')
        plt.plot(k_range, ch_scores, label='CH')
        plt.plot(k_range, db_scores, label='DB')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "metrics.png"))
        plt.close()

        plt.figure(figsize=(20, 16))
        for i, k in enumerate(k_range):
            plt.subplot(3, 3, i + 1)
            plt.scatter(X_embed[:, 0], X_embed[:, 1], c=all_labels[k], s=3, cmap='tab20')
            plt.title(f"TSNE k={k}")
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "TSNE_allK.png"))
        plt.close()

        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        for k, label_array in all_labels.items():
            np.save(os.path.join(labels_dir, f"labels_k{k}.npy"), label_array)

        np.save(os.path.join(output_dir, f"labels_bestK_{best_k}.npy"), all_labels[best_k])
        np.save(os.path.join(output_dir, "sil_scores.npy"), np.array(sil_scores))
        np.save(os.path.join(output_dir, "ch_scores.npy"), np.array(ch_scores))
        np.save(os.path.join(output_dir, "db_scores.npy"), np.array(db_scores))
        np.save(os.path.join(output_dir, "k_range.npy"), np.array(list(k_range)))
        np.save(os.path.join(output_dir, "idx_count.npy"), idx_count)
        np.save(os.path.join(output_dir, "best_k.npy"), np.array(best_k))
        np.save(os.path.join(output_dir, "X_embed.npy"), X_embed)
        np.save(os.path.join(output_dir, "X_scaled.npy"), X_scaled)

        config_save = {
            "dim_reduction": "tsne",
            "cluster_method": "kmeans",
            "k_range": list(k_range),
            "best_k": int(best_k)
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_save, f, indent=2)

    return {
        'all_labels': all_labels,
        'best_k': best_k,
        'sil_scores': sil_scores,
        'ch_scores': ch_scores,
        'db_scores': db_scores,
        'X_embed': X_embed,
        'X_scaled': X_scaled
    }


if __name__ == "__main__":
    """
    Example Usage: Superpixel Clustering (t-SNE + KMeans)
    
    Input:
        input_dir: Directory with standardized features (all_superpixels.npy, idx_count.npy)
        output_dir: Directory to save clustering results
    
    Output:
        labels_bestK_{k}.npy, best_k.npy, metrics.png, TSNE_allK.png
    """
    input_dir = "./path/to/features"
    output_dir = "./output/clustering"
    