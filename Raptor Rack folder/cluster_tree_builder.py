"""
raptor/cluster_tree_builder.py — Hierarchical Tree Construction via recursive
clustering and abstractive summarization (Sarthi et al., ICLR 2024).
"""

import copy
import logging
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

try:
    from .tree_structures import Node, Tree
    from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering, Recency_Clustering
    from .models import BaseEmbeddingModel, BaseSummarizationModel, OllamaEmbeddingModel, GemmaSummarizer
except (ImportError, ValueError):
    from tree_structures import Node, Tree
    from cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering, Recency_Clustering
    from models import BaseEmbeddingModel, BaseSummarizationModel, OllamaEmbeddingModel, GemmaSummarizer

logger = logging.getLogger("raptor.builder")


def count_tokens(text: str) -> int:
    """Accurately count tokens using word and punctuation tokens."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text))


def split_text_into_chunks(text: str, max_tokens: int = 100, overlap: int = 0) -> List[str]:
    """
    Segment retrieval corpus into short, contiguous texts of length 100 tokens.
    Sentence boundary preserving: if a sentence exceeds 100-token limit,
    move the entire sentence to the next chunk rather than cutting it mid-sentence.
    """
    if not text or not text.strip():
        return []

    # Split text into sentences using sentence-terminal punctuation
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for s in sentences:
        s_tokens = count_tokens(s)

        if s_tokens > max_tokens:
            # If a single sentence exceeds max_tokens, split on clause boundaries
            sub_clauses = re.split(r"(?<=[,;:])\s+", s)
            for sc in sub_clauses:
                sc_clean = sc.strip()
                if not sc_clean:
                    continue
                sc_tokens = count_tokens(sc_clean)
                if current_tokens + sc_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                current_chunk.append(sc_clean)
                current_tokens += sc_tokens
        elif current_tokens + s_tokens > max_tokens:
            # Move entire sentence to next chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_tokens = s_tokens
        else:
            current_chunk.append(s)
            current_tokens += s_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class ClusterTreeConfig:
    def __init__(
        self,
        max_tokens: int = 100,
        num_layers: int = 5,
        threshold: float = 0.1,
        reduction_dimension: int = 10,
        summarization_length: int = 150,
        clustering_algorithm: type = RAPTOR_Clustering,
        embedding_models: Optional[Dict[str, BaseEmbeddingModel]] = None,
        cluster_embedding_model: str = "Ollama",
        summarization_model: Optional[BaseSummarizationModel] = None,
    ):
        self.max_tokens = max_tokens
        self.num_layers = num_layers
        self.threshold = threshold
        self.reduction_dimension = reduction_dimension
        self.summarization_length = summarization_length
        self.clustering_algorithm = clustering_algorithm
        self.embedding_models = embedding_models or {"Ollama": OllamaEmbeddingModel()}
        self.cluster_embedding_model = cluster_embedding_model
        self.summarization_model = summarization_model or GemmaSummarizer()


class ClusterTreeBuilder:
    """
    Constructs the multi-layer RAPTOR tree from input text bottom-up
    through recursive clustering and summarization.
    """
    def __init__(self, config: Optional[ClusterTreeConfig] = None):
        self.config = config or ClusterTreeConfig()
        self.max_tokens = self.config.max_tokens
        self.num_layers = self.config.num_layers
        self.threshold = self.config.threshold
        self.reduction_dimension = self.config.reduction_dimension
        self.summarization_length = self.config.summarization_length
        self.clustering_algorithm = self.config.clustering_algorithm
        self.embedding_models = self.config.embedding_models
        self.cluster_embedding_model = self.config.cluster_embedding_model
        self.summarization_model = self.config.summarization_model

    def create_node(
        self,
        index: int,
        text: str,
        children: Optional[Set[int]] = None,
        layer: int = 0,
        metadata: Optional[Dict] = None,
    ) -> Node:
        """Create a node and compute its embeddings across configured embedding models."""
        embeddings = {}
        for name, emb_model in self.embedding_models.items():
            embeddings[name] = emb_model.create_embedding(text)

        node = Node(
            text=text,
            index=index,
            children=children or set(),
            parents=set(),
            layer=layer,
            embeddings=embeddings,
            token_size=count_tokens(text),
            metadata=metadata or {},
        )
        return node

    def build_from_text(self, text: str, verbose: bool = False) -> Tree:
        """
        Build full RAPTOR tree from raw text.
        """
        # 1. Segment text into 100-token sentence-aware leaf chunks
        chunks = split_text_into_chunks(text, max_tokens=self.max_tokens)
        if not chunks:
            raise ValueError("Input text produced zero chunks.")

        if verbose:
            logger.info(f"Segmented text into {len(chunks)} leaf chunks (max {self.max_tokens} tokens).")

        # 2. Batch embed leaf chunks in one shot
        emb_model = self.embedding_models[self.cluster_embedding_model]
        chunk_embeddings = emb_model.create_embeddings(chunks)

        leaf_nodes: Dict[int, Node] = {}
        for idx, chunk_text in enumerate(chunks):
            leaf_nodes[idx] = Node(
                text=chunk_text,
                index=idx,
                children=set(),
                parents=set(),
                layer=0,
                embeddings={self.cluster_embedding_model: chunk_embeddings[idx]},
                token_size=count_tokens(chunk_text),
                metadata={"source": "leaf_chunk", "chunk_id": idx}
            )

        all_tree_nodes: Dict[int, Node] = copy.copy(leaf_nodes)
        layer_to_nodes: Dict[int, List[Node]] = {0: list(leaf_nodes.values())}
        current_level_nodes: Dict[int, Node] = copy.copy(leaf_nodes)

        next_node_index = len(all_tree_nodes)

        # 3. Recursive bottom-up tree construction
        from concurrent.futures import ThreadPoolExecutor

        for layer in range(self.num_layers):
            current_nodes_list = list(current_level_nodes.values())
            n_current = len(current_nodes_list)

            if verbose:
                logger.info(f"--- Constructing Tree Layer {layer + 1} from {n_current} nodes ---")

            # Stopping condition: cannot reduce further
            if n_current <= max(2, self.reduction_dimension + 1):
                if verbose:
                    logger.info(f"Stopping tree construction: current nodes ({n_current}) <= threshold ({self.reduction_dimension + 1}).")
                break

            # Perform soft clustering
            clusters = self.clustering_algorithm.perform_clustering(
                nodes=current_nodes_list,
                embedding_model_name=self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                threshold=self.threshold,
                verbose=verbose,
            )

            if not clusters or len(clusters) == 0:
                if verbose:
                    logger.info(f"Clustering returned 0 clusters at Layer {layer}. Stopping.")
                break

            # If all nodes fell into a single cluster and we already built at least 1 layer
            if len(clusters) == 1 and len(clusters[0]) == n_current and layer >= 1:
                cluster = clusters[0]
                context = "\n\n".join([n.text for n in cluster])
                summary_text = self.summarization_model.summarize(
                    context, max_tokens=self.summarization_length
                )
                child_indices = {n.index for n in cluster}
                parent_emb = emb_model.create_embedding(summary_text)
                parent_node = Node(
                    text=summary_text,
                    index=next_node_index,
                    children=child_indices,
                    parents=set(),
                    layer=layer + 1,
                    embeddings={self.cluster_embedding_model: parent_emb},
                    token_size=count_tokens(summary_text),
                    metadata={"cluster_id": 0, "parent_layer": layer + 1}
                )
                for child_idx in child_indices:
                    all_tree_nodes[child_idx].parents.add(next_node_index)

                all_tree_nodes[next_node_index] = parent_node
                layer_to_nodes[layer + 1] = [parent_node]
                next_node_index += 1
                break

            # Sequentially summarize each cluster to match Ollama's single-stream local queue
            summarized_clusters = []
            for cid, cl in enumerate(clusters):
                if not cl:
                    continue
                ctx = "\n\n".join([n.text for n in cl])
                summary = self.summarization_model.summarize(ctx, max_tokens=self.summarization_length)
                summarized_clusters.append((cid, cl, summary))

            # Batch embed parent summaries
            parent_texts = [s for _, _, s in summarized_clusters]
            parent_embs = emb_model.create_embeddings(parent_texts)

            new_level_nodes: Dict[int, Node] = {}
            for idx, (cid, cl, summary_text) in enumerate(summarized_clusters):
                child_indices = {n.index for n in cl}
                parent_node = Node(
                    text=summary_text,
                    index=next_node_index,
                    children=child_indices,
                    parents=set(),
                    layer=layer + 1,
                    embeddings={self.cluster_embedding_model: parent_embs[idx]},
                    token_size=count_tokens(summary_text),
                    metadata={"cluster_id": cid, "parent_layer": layer + 1}
                )

                for child_idx in child_indices:
                    if child_idx in all_tree_nodes:
                        all_tree_nodes[child_idx].parents.add(next_node_index)

                new_level_nodes[next_node_index] = parent_node
                all_tree_nodes[next_node_index] = parent_node
                next_node_index += 1

            if not new_level_nodes:
                break

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes

        # Identify roots (highest layer nodes)
        max_layer = max(layer_to_nodes.keys())
        root_nodes = {n.index: n for n in layer_to_nodes[max_layer]}
        total_layers = max_layer + 1

        tree = Tree(
            all_nodes=all_tree_nodes,
            root_nodes=root_nodes,
            leaf_nodes=leaf_nodes,
            num_layers=total_layers,
            layer_to_nodes=layer_to_nodes,
        )

        if verbose:
            stats = tree.get_statistics()
            logger.info(f"Tree built: {stats['num_layers']} layers, {stats['total_nodes']} total nodes ({stats['leaf_nodes']} leaves, {stats['non_leaf_nodes']} summaries).")

        return tree
