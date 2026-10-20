"""
raptor/cluster_utils.py — Dimensionality reduction, GMM clustering, BIC selection,
and recursive soft-clustering algorithms as described in Sarthi et al. (ICLR 2024).
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Set
import numpy as np
import umap
from sklearn.mixture import GaussianMixture

from .tree_structures import Node

logger = logging.getLogger("raptor.cluster")
RANDOM_SEED = 224
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Reduce embedding dimensionality for global structure preservation using UMAP.
    """
    n_samples = len(embeddings)
    if n_samples <= dim + 1:
        return embeddings

    if n_neighbors is None:
        n_neighbors = max(2, min(int((n_samples - 1) ** 0.5), n_samples - 1))
    else:
        n_neighbors = max(2, min(n_neighbors, n_samples - 1))

    target_dim = min(dim, n_samples - 2)
    if target_dim < 2:
        return embeddings

    try:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=target_dim,
            metric=metric,
            random_state=RANDOM_SEED,
            n_epochs=200,
        )
        return reducer.fit_transform(embeddings)
    except Exception as e:
        logger.warning(f"UMAP global reduction failed ({e}), using raw embeddings.")
        return embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    num_neighbors: int = 10,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Reduce embedding dimensionality for local cluster structure preservation using UMAP.
    """
    n_samples = len(embeddings)
    if n_samples <= dim + 1:
        return embeddings

    n_neighbors = max(2, min(num_neighbors, n_samples - 1))
    target_dim = min(dim, n_samples - 2)
    if target_dim < 2:
        return embeddings

    try:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=target_dim,
            metric=metric,
            random_state=RANDOM_SEED,
            n_epochs=200,
        )
        return reducer.fit_transform(embeddings)
    except Exception as e:
        logger.warning(f"UMAP local reduction failed ({e}), using raw embeddings.")
        return embeddings


def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using Bayesian Information Criterion (BIC).
    BIC = ln(N) * k - 2 * ln(L_hat)
    """
    n_samples = len(embeddings)
    if n_samples <= 2:
        return 1

    max_k = max(2, min(max_clusters, n_samples // 3 + 1))
    candidate_k = list(range(1, max_k))
    if not candidate_k:
        return 1

    bics = []
    for k in candidate_k:
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-2,
                random_state=random_state,
                max_iter=100,
            )
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        except Exception:
            bics.append(float("inf"))

    best_idx = int(np.argmin(bics))
    optimal_k = candidate_k[best_idx]
    return max(1, optimal_k)


def GMM_cluster(
    embeddings: np.ndarray,
    threshold: float = 0.1,
    random_state: int = RANDOM_SEED
) -> Tuple[List[np.ndarray], int]:
    """
    Perform Gaussian Mixture Model (GMM) clustering with soft assignment thresholding.
    Points are assigned to all clusters where posterior probability exceeds threshold.
    """
    n_samples = len(embeddings)
    if n_samples <= 1:
        return [[np.array([0])]], 1

    n_clusters = get_optimal_clusters(embeddings, random_state=random_state)
    if n_clusters == 1:
        return [np.array([0]) for _ in range(n_samples)], 1

    gm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        reg_covar=1e-2,
        random_state=random_state,
        max_iter=100,
    )
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)

    labels = []
    for i, prob in enumerate(probs):
        assigned = np.where(prob > threshold)[0]
        if len(assigned) == 0:
            # Fall back to argmax if no cluster passes threshold
            assigned = np.array([int(np.argmax(prob))])
        labels.append(assigned)

    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int = 10,
    threshold: float = 0.1,
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Execute RAPTOR's two-step hierarchical clustering (global UMAP -> global GMM -> local UMAP -> local GMM).
    Returns list of assigned cluster label arrays for each embedding index.
    """
    n_samples = len(embeddings)
    if n_samples <= 2:
        return [np.array([0]) for _ in range(n_samples)]

    # 1. Global reduction and clustering
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim=dim)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold=threshold
    )

    if verbose:
        logger.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([], dtype=int) for _ in range(n_samples)]
    total_clusters = 0

    # 2. Local clustering within each global cluster
    for i in range(n_global_clusters):
        # Find indices of points belonging to global cluster i
        member_indices = [idx for idx, gc in enumerate(global_clusters) if i in gc]
        if len(member_indices) == 0:
            continue

        global_cluster_embeddings_ = embeddings[member_indices]

        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim=dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold=threshold
            )

        # Map back to global embedding indices
        for local_idx, orig_idx in enumerate(member_indices):
            assigned_local = local_clusters[local_idx]
            global_assigned = assigned_local + total_clusters
            all_local_clusters[orig_idx] = np.append(
                all_local_clusters[orig_idx], global_assigned
            ).astype(int)

        total_clusters += n_local_clusters

    # Ensure every point has at least one cluster assigned
    for idx in range(n_samples):
        if len(all_local_clusters[idx]) == 0:
            all_local_clusters[idx] = np.array([0], dtype=int)

    if verbose:
        logger.info(f"Total Local Clusters: {total_clusters}")

    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, nodes: List[Node], **kwargs) -> List[List[Node]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    """
    RAPTOR Soft-Clustering with UMAP, GMM, BIC, and recursive token thresholding.
    """
    @classmethod
    def perform_clustering(
        cls,
        nodes: List[Node],
        embedding_model_name: Optional[str] = None,
        max_length_in_cluster: int = 3500,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        if not nodes:
            return []
        if len(nodes) <= 2:
            return [nodes]

        embeddings = np.array([node.get_embedding(embedding_model_name) for node in nodes])
        clusters = perform_clustering(
            embeddings,
            dim=reduction_dimension,
            threshold=threshold,
            verbose=verbose
        )

        node_clusters = []
        unique_labels = np.unique(np.concatenate(clusters))

        for label in unique_labels:
            indices = [i for i, c in enumerate(clusters) if label in c]
            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_tokens = sum(n.token_size for n in cluster_nodes)
            if total_tokens > max_length_in_cluster and len(cluster_nodes) > 2:
                if verbose:
                    logger.info(f"Reclustering cluster with {len(cluster_nodes)} nodes ({total_tokens} tokens)")
                # Recursive clustering on oversized cluster
                node_clusters.extend(
                    cls.perform_clustering(
                        cluster_nodes,
                        embedding_model_name=embedding_model_name,
                        max_length_in_cluster=max_length_in_cluster,
                        reduction_dimension=reduction_dimension,
                        threshold=threshold,
                        verbose=verbose,
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters


class Recency_Clustering(ClusteringAlgorithm):
    """
    Ablation baseline (Table 9 in Paper):
    Balanced tree-style encoding and summarization of contiguous chunks
    using a fixed window size (default 7 chunks, matching paper Appendix B.1).
    """
    @classmethod
    def perform_clustering(
        cls,
        nodes: List[Node],
        window_size: int = 7,
        **kwargs
    ) -> List[List[Node]]:
        if not nodes:
            return []
        node_clusters = []
        for i in range(0, len(nodes), window_size):
            chunk_group = nodes[i:i + window_size]
            if chunk_group:
                node_clusters.append(chunk_group)
        return node_clusters
