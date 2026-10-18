import os
import sys

# Prevent joblib/loky subprocess hang on Windows where wmic is deprecated/missing
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

from typing import Dict, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def perform_clustering(embeddings: np.ndarray, n_clusters: Optional[int] = None) -> Optional[Dict]:
    """
    Perform Gaussian Mixture Model (GMM) clustering on vector embeddings.
    
    Steps:
      1. Dimensionality reduction via PCA (if >100 dimensions and >50 samples)
      2. Feature standardization via StandardScaler
      3. GMM clustering with diagonal covariance
      4. Silhouette score evaluation
    """
    if embeddings is None or len(embeddings) < 3:
        print(f"Not enough embeddings for clustering: {len(embeddings) if embeddings is not None else 0}")
        return None
    
    print(f"Clustering {len(embeddings)} embeddings...")
    
    # 1. Reduce dimensions with PCA
    if embeddings.shape[1] > 100 and len(embeddings) > 50:
        print(f"Reducing dimensions from {embeddings.shape[1]}...")
        n_components = min(50, len(embeddings) // 2)
        pca = PCA(n_components=n_components)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"Reduced to {embeddings_reduced.shape[1]} dimensions")
    else:
        embeddings_reduced = embeddings
    
    # 2. Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_reduced)
    
    # 3. Determine number of clusters
    if n_clusters is None:
        n_clusters = max(2, min(8, len(embeddings) // 100))
    
    print(f"Running GMM with {n_clusters} clusters...")
    
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        random_state=42,
        n_init=5,
        max_iter=200
    )
    
    cluster_labels = gmm.fit_predict(embeddings_scaled)
    cluster_probs = gmm.predict_proba(embeddings_scaled)
    
    # 4. Calculate silhouette score
    if len(set(cluster_labels)) > 1:
        try:
            silhouette_avg = float(silhouette_score(embeddings_scaled, cluster_labels))
            print(f"Silhouette Score: {silhouette_avg:.3f}")
        except Exception:
            silhouette_avg = 0.0
    else:
        silhouette_avg = 0.0
    
    return {
        'labels': cluster_labels,
        'probabilities': cluster_probs,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'embeddings_scaled': embeddings_scaled
    }
