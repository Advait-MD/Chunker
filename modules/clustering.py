import os
import sys

# Prevent joblib/loky subprocess hang on Windows where wmic is deprecated/missing
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

from typing import Dict, Optional, Tuple, List
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


def find_optimal_k_bic(embeddings: np.ndarray, min_k: int = 2, max_k: int = 8) -> Tuple[int, float]:
    """
    Dynamic Bayesian Information Criterion (BIC) optimal cluster selection (RAPTOR style).
    Sweeps candidate cluster counts k and selects the k that minimizes:
        BIC = ln(N) * k_params - 2 * ln(L)
    Balances model fit against parameter complexity without empirical heuristics.
    """
    n_samples = embeddings.shape[0]
    if n_samples < min_k:
        return max(1, n_samples), 0.0
    
    upper_bound = min(max_k, max(min_k, n_samples // 4))
    candidate_ks = list(range(min_k, upper_bound + 1))
    
    if len(candidate_ks) <= 1:
        return candidate_ks[0], 0.0
        
    bics = []
    fitted_models = []
    
    for k in candidate_ks:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='diag',
                random_state=42,
                n_init=2,
                max_iter=100
            )
            gmm.fit(embeddings)
            bic = float(gmm.bic(embeddings))
            bics.append(bic)
            fitted_models.append((k, bic))
        except Exception as e:
            logger.debug(f"BIC estimation failed for k={k}: {e}")
            bics.append(float('inf'))
            fitted_models.append((k, float('inf')))
            
    best_idx = int(np.argmin(bics))
    best_k, best_bic = fitted_models[best_idx]
    logger.info(f"Dynamic BIC Selection: Optimal k={best_k} (BIC: {best_bic:.2f}, candidates tested: {candidate_ks})")
    return best_k, best_bic


def perform_clustering(embeddings: np.ndarray, 
                       n_clusters: Optional[int] = None,
                       enable_soft: Optional[bool] = None,
                       soft_threshold: Optional[float] = None) -> Optional[Dict]:
    """
    Perform Gaussian Mixture Model (GMM) clustering on vector embeddings with:
      1. Dimensionality reduction via PCA (if >100 dimensions and >50 samples)
      2. Feature standardization via StandardScaler
      3. Dynamic Bayesian Information Criterion (BIC) optimal cluster count selection (RAPTOR upgrade)
      4. Soft GMM probabilistic multi-cluster assignment (RAPTOR upgrade)
      5. Silhouette score evaluation (dataset-wide and per-cluster)
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
    
    # 3. Determine number of clusters (Dynamic BIC or explicit override)
    bic_val = 0.0
    if n_clusters is None:
        if config.clustering.bic_selection:
            n_clusters, bic_val = find_optimal_k_bic(
                embeddings_scaled,
                min_k=config.clustering.min_k,
                max_k=config.clustering.max_k
            )
        else:
            n_clusters = max(2, min(8, len(embeddings) // 100))
    
    logger.info(f"Running GMM with {n_clusters} clusters (BIC: {bic_val:.2f})...")
    
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        random_state=42,
        n_init=2,
        max_iter=150
    )
    
    gmm.fit(embeddings_scaled)
    cluster_probs = gmm.predict_proba(embeddings_scaled)
    cluster_labels = np.argmax(cluster_probs, axis=1)
    
    # 4. Soft Clustering via Posterior Probabilities (RAPTOR upgrade)
    use_soft = enable_soft if enable_soft is not None else config.clustering.enable_soft_clustering
    threshold = soft_threshold if soft_threshold is not None else config.clustering.soft_threshold
    
    cluster_members: Dict[int, List[int]] = {c: [] for c in range(n_clusters)}
    soft_assignments: List[List[int]] = []
    
    for idx, row in enumerate(cluster_probs):
        if use_soft:
            assigned = [k for k, p in enumerate(row) if p >= threshold]
            if not assigned:
                assigned = [int(np.argmax(row))]
        else:
            assigned = [int(cluster_labels[idx])]
            
        soft_assignments.append(assigned)
        for c in assigned:
            cluster_members[c].append(idx)
            
    # Soft overlap diagnostic logging
    multi_cluster_count = sum(1 for a in soft_assignments if len(a) > 1)
    if multi_cluster_count > 0:
        logger.info(f"Soft Clustering: {multi_cluster_count}/{len(embeddings)} chunks assigned to multiple clusters (threshold={threshold:.2f})")
    
    # 5. Calculate silhouette score (overall and per-cluster using hard assignments)
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
        member_cnt = len(cluster_members.get(c_id, []))
        logger.debug(f"  • Cluster {c_id}: {member_cnt} members, silhouette = {sil:.3f}")
    
    return {
        'labels': cluster_labels,
        'probabilities': cluster_probs,
        'cluster_members': cluster_members,
        'soft_assignments': soft_assignments,
        'n_clusters': n_clusters,
        'bic_score': bic_val,
        'silhouette_score': silhouette_avg,
        'sample_silhouettes': sample_silhouettes,
        'cluster_silhouettes': cluster_silhouettes,
        'embeddings_scaled': embeddings_scaled
    }
