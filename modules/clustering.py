import os
import sys

# Prevent joblib/loky subprocess hang on Windows where wmic is deprecated/missing
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

from typing import Dict, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import config
    from .logger import get_logger
except ImportError:
    from modules.config import config
    from modules.logger import get_logger

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = get_logger("clustering")


def perform_clustering(embeddings: np.ndarray, n_clusters: Optional[int] = None) -> Optional[Dict]:
    """
    Perform Gaussian Mixture Model (GMM) clustering on vector embeddings.
    
    Steps:
      1. Dimensionality reduction via PCA (if >100 dimensions and >50 samples)
      2. Feature standardization via StandardScaler
      3. GMM clustering with diagonal covariance
      4. Silhouette score evaluation (dataset-wide and per-cluster)
    """
    if embeddings is None or len(embeddings) < 3:
        logger.warning(f"Not enough embeddings for clustering: {len(embeddings) if embeddings is not None else 0}")
        return None
    
    logger.info(f"Clustering {len(embeddings)} embeddings...")
    
    # 1. Reduce dimensions with PCA
    if embeddings.shape[1] > 100 and len(embeddings) > 50:
        max_pca = config.clustering.max_pca_components
        n_components = min(max_pca, len(embeddings) // 2)
        logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {n_components} via PCA...")
        pca = PCA(n_components=n_components)
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings
    
    # 2. Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_reduced)
    
    # 3. Determine number of clusters
    if n_clusters is None:
        n_clusters = max(2, min(8, len(embeddings) // 100))
    
    logger.info(f"Running GMM with {n_clusters} clusters...")
    
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        random_state=42,
        n_init=5,
        max_iter=200
    )
    
    cluster_labels = gmm.fit_predict(embeddings_scaled)
    cluster_probs = gmm.predict_proba(embeddings_scaled)
    
    # 4. Calculate silhouette score (overall and per-cluster)
    if len(set(cluster_labels)) > 1:
        try:
            silhouette_avg = float(silhouette_score(embeddings_scaled, cluster_labels))
            sample_silhouettes = silhouette_samples(embeddings_scaled, cluster_labels)
        except Exception as e:
            logger.warning(f"Failed to calculate silhouette score: {e}")
            silhouette_avg = 0.0
            sample_silhouettes = np.zeros(len(cluster_labels))
    else:
        silhouette_avg = 0.0
        sample_silhouettes = np.zeros(len(cluster_labels))
    
    cluster_silhouettes = {}
    for c_id in range(n_clusters):
        c_indices = np.where(cluster_labels == c_id)[0]
        if len(c_indices) > 0 and len(sample_silhouettes) == len(cluster_labels):
            cluster_silhouettes[c_id] = float(np.mean(sample_silhouettes[c_indices]))
        else:
            cluster_silhouettes[c_id] = 0.0

    logger.info(f"Overall Silhouette Score: {silhouette_avg:.3f}")
    for c_id, sil in cluster_silhouettes.items():
        logger.debug(f"  • Cluster {c_id}: silhouette = {sil:.3f}")
    
    return {
        'labels': cluster_labels,
        'probabilities': cluster_probs,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'sample_silhouettes': sample_silhouettes,
        'cluster_silhouettes': cluster_silhouettes,
        'embeddings_scaled': embeddings_scaled
    }
