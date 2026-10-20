"""
raptor/tree_retriever.py — Tree Retrieval Algorithms (Tree Traversal & Collapsed Tree)
adhering strictly to Algorithms 1 & 2 from Sarthi et al. (ICLR 2024).
"""

import logging
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

try:
    from .tree_structures import Node, Tree
    from .models import BaseEmbeddingModel, OllamaEmbeddingModel
except (ImportError, ValueError):
    from tree_structures import Node, Tree
    from models import BaseEmbeddingModel, OllamaEmbeddingModel

logger = logging.getLogger("raptor.retriever")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class TreeRetriever:
    """
    Implements Tree Traversal (Algorithm 1) and Collapsed Tree (Algorithm 2) retrieval
    over the RAPTOR hierarchical tree structure.
    """
    def __init__(
        self,
        tree: Tree,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        embedding_model_name: str = "Ollama",
    ):
        self.tree = tree
        self.embedding_model = embedding_model or OllamaEmbeddingModel()
        self.embedding_model_name = embedding_model_name

    def get_query_embedding(self, query: str) -> np.ndarray:
        return self.embedding_model.create_embedding(query)

    def collapsed_tree_retrieval(
        self,
        query: str,
        max_tokens: int = 2000,
        allowed_layers: Optional[List[int]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Node], str]:
        """
        Algorithm 2: Collapsed Tree Algorithm (Sarthi et al. Appendix F).
        Flattens the entire tree into 1D, computes cosine similarity against all nodes,
        and retrieves top nodes until max_tokens budget is reached.
        
        Args:
            query: Query string.
            max_tokens: Maximum tokens in retrieved context (paper default: 2000).
            allowed_layers: Optional list of layers to restrict search (for layer ablations).
            top_k: Optional cap on the number of nodes.
        """
        query_emb = self.get_query_embedding(query)

        # 1. Gather all nodes (or filtered by allowed_layers)
        if allowed_layers is not None:
            candidate_nodes = []
            for l in allowed_layers:
                candidate_nodes.extend(self.tree.get_layer(l))
        else:
            candidate_nodes = self.tree.get_all_nodes()

        if not candidate_nodes:
            return [], ""

        # 2. Score each node by cosine similarity
        scored_nodes: List[Tuple[Node, float]] = []
        for node in candidate_nodes:
            node_emb = node.get_embedding(self.embedding_model_name)
            sim = cosine_similarity(query_emb, node_emb)
            scored_nodes.append((node, sim))

        # 3. Sort descending by similarity
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored_nodes = scored_nodes[:top_k]

        # 4. Greedily accumulate nodes until max_tokens
        result: List[Node] = []
        total_tokens = 0

        for node, score in scored_nodes:
            if total_tokens + node.token_size <= max_tokens:
                result.append(node)
                total_tokens += node.token_size
            elif not result:
                # Always ensure at least the top-1 node is included
                result.append(node)
                total_tokens += node.token_size
                break

        # Assemble retrieved context text
        context_parts = [node.text for node in result]
        context = "\n\n".join(context_parts)

        return result, context

    def tree_traversal_retrieval(
        self,
        query: str,
        k: int = 3,
        start_layer: Optional[int] = None,
    ) -> Tuple[List[Node], str]:
        """
        Algorithm 1: Tree Traversal Algorithm (Sarthi et al. Appendix F).
        Starts at the root layer, selects top-k nodes by cosine similarity,
        descends to the children of selected nodes at the next layer,
        and repeats down to the leaf layer.
        """
        query_emb = self.get_query_embedding(query)

        if start_layer is None:
            start_layer = self.tree.num_layers - 1

        start_layer = min(start_layer, self.tree.num_layers - 1)
        s_current: List[Node] = self.tree.get_layer(start_layer)
        if not s_current:
            s_current = list(self.tree.root_nodes.values())

        retrieved_layers: List[List[Node]] = []

        for current_layer_idx in range(start_layer, -1, -1):
            if not s_current:
                break

            # Score current candidates
            scored = [
                (node, cosine_similarity(query_emb, node.get_embedding(self.embedding_model_name)))
                for node in s_current
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_k_nodes = [node for node, score in scored[:k]]
            retrieved_layers.append(top_k_nodes)

            # Move to children for next layer down
            if current_layer_idx > 0:
                child_indices: Set[int] = set()
                for node in top_k_nodes:
                    child_indices.update(node.children)

                s_current = [
                    self.tree.all_nodes[idx]
                    for idx in child_indices
                    if idx in self.tree.all_nodes
                ]

        # Union of nodes across layers preserving traversal order
        seen_indices = set()
        final_nodes: List[Node] = []
        for layer_nodes in retrieved_layers:
            for node in layer_nodes:
                if node.index not in seen_indices:
                    seen_indices.add(node.index)
                    final_nodes.append(node)

        context = "\n\n".join([n.text for n in final_nodes])
        return final_nodes, context

    def retrieve(
        self,
        query: str,
        collapse_tree: bool = True,
        max_tokens: int = 2000,
        top_k: int = 5,
        allowed_layers: Optional[List[int]] = None,
    ) -> Tuple[List[Node], str, Dict[int, int]]:
        """
        Unified retrieval entry point. Returns (selected_nodes, context_text, layer_counts).
        """
        if collapse_tree:
            nodes, context = self.collapsed_tree_retrieval(
                query=query,
                max_tokens=max_tokens,
                allowed_layers=allowed_layers,
                top_k=top_k,
            )
        else:
            nodes, context = self.tree_traversal_retrieval(
                query=query,
                k=top_k,
            )

        # Count layers of retrieved nodes
        layer_counts: Dict[int, int] = {}
        for node in nodes:
            layer_counts[node.layer] = layer_counts.get(node.layer, 0) + 1

        return nodes, context, layer_counts
