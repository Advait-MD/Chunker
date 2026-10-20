"""
raptor/raptor_system.py — Unified RAPTOR Architecture System Interface.
Integrates Tree Building, Recursive Summarization, Multi-layer Retrieval,
and Reader/QA generation with local Gemma 3:1b.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

try:
    from .tree_structures import Node, Tree
    from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig, split_text_into_chunks
    from .cluster_utils import RAPTOR_Clustering, Recency_Clustering
    from .tree_retriever import TreeRetriever, cosine_similarity
    from .models import (
        BaseEmbeddingModel,
        OllamaEmbeddingModel,
        SBertEmbeddingModel,
        BM25Retriever,
        GemmaSummarizer,
        GemmaQAModel,
    )
except (ImportError, ValueError):
    from tree_structures import Node, Tree
    from cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig, split_text_into_chunks
    from cluster_utils import RAPTOR_Clustering, Recency_Clustering
    from tree_retriever import TreeRetriever, cosine_similarity
    from models import (
        BaseEmbeddingModel,
        OllamaEmbeddingModel,
        SBertEmbeddingModel,
        BM25Retriever,
        GemmaSummarizer,
        GemmaQAModel,
    )

logger = logging.getLogger("raptor.system")


class RAPTOR:
    """
    Main RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) system.
    """
    def __init__(
        self,
        tree: Optional[Tree] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        summarization_model: Optional[GemmaSummarizer] = None,
        qa_model: Optional[GemmaQAModel] = None,
        embedding_model_name: Optional[str] = None,
        summarization_model_name: Optional[str] = None,
        cluster_tree_config: Optional[ClusterTreeConfig] = None,
        clustering_algorithm: type = RAPTOR_Clustering,
        max_tokens_per_chunk: int = 100,
        num_layers: int = 5,
        threshold: float = 0.1,
        reduction_dimension: int = 10,
    ):
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif embedding_model_name:
            self.embedding_model = OllamaEmbeddingModel(model_name=embedding_model_name)
        else:
            self.embedding_model = OllamaEmbeddingModel()

        if summarization_model is not None:
            self.summarization_model = summarization_model
        elif summarization_model_name:
            self.summarization_model = GemmaSummarizer(model_name=summarization_model_name)
        else:
            self.summarization_model = GemmaSummarizer()

        if qa_model is not None:
            self.qa_model = qa_model
        elif summarization_model_name:
            self.qa_model = GemmaQAModel(model_name=summarization_model_name)
        else:
            self.qa_model = GemmaQAModel()

        self.cluster_tree_config = cluster_tree_config or ClusterTreeConfig(
            max_tokens=max_tokens_per_chunk,
            num_layers=num_layers,
            threshold=threshold,
            reduction_dimension=reduction_dimension,
            clustering_algorithm=clustering_algorithm,
            embedding_models={"EMB": self.embedding_model},
            cluster_embedding_model="EMB",
            summarization_model=self.summarization_model,
        )

        self.tree_builder = ClusterTreeBuilder(self.cluster_tree_config)
        self.tree: Optional[Tree] = tree
        self.retriever: Optional[TreeRetriever] = None

        if self.tree is not None:
            self._init_retriever()

    def _init_retriever(self):
        if self.tree is not None:
            self.retriever = TreeRetriever(
                tree=self.tree,
                embedding_model=self.embedding_model,
                embedding_model_name="EMB",
            )

    def add_documents(self, text: str, verbose: bool = False) -> Tree:
        """
        Build RAPTOR hierarchical tree from input document text.
        """
        self.tree = self.tree_builder.build_from_text(text, verbose=verbose)
        self._init_retriever()
        return self.tree

    def retrieve(
        self,
        query: str,
        collapse_tree: bool = True,
        max_tokens: int = 2000,
        top_k: int = 5,
        allowed_layers: Optional[List[int]] = None,
    ) -> Tuple[List[Node], str, Dict[int, int]]:
        """
        Retrieve context nodes from RAPTOR tree.
        """
        if self.retriever is None:
            raise ValueError("RAPTOR tree has not been built yet. Call add_documents() first.")

        return self.retriever.retrieve(
            query=query,
            collapse_tree=collapse_tree,
            max_tokens=max_tokens,
            top_k=top_k,
            allowed_layers=allowed_layers,
        )

    def answer_question(
        self,
        question: str,
        options: Optional[List[str]] = None,
        collapse_tree: bool = True,
        max_tokens: int = 2000,
        allowed_layers: Optional[List[int]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        End-to-end question answering: retrieves relevant context from RAPTOR tree
        and generates response with reader model (Gemma 3:1b).
        """
        nodes, context, layer_counts = self.retrieve(
            query=question,
            collapse_tree=collapse_tree,
            max_tokens=max_tokens,
            allowed_layers=allowed_layers,
        )

        answer = self.qa_model.answer_question(
            context=context,
            question=question,
            options=options,
            temperature=temperature,
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_nodes_count": len(nodes),
            "retrieved_tokens": sum(n.token_size for n in nodes),
            "layer_distribution": layer_counts,
            "context": context,
            "nodes": nodes,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get tree structure and compression metrics."""
        if self.tree is None:
            return {}
        return self.tree.get_statistics()

    def save(self, file_path: str):
        """Persist tree to disk."""
        if self.tree is None:
            raise ValueError("No tree to save.")
        self.tree.save(file_path)

    @classmethod
    def load(
        cls,
        file_path: str,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        qa_model: Optional[GemmaQAModel] = None,
    ) -> "RAPTOR":
        """Load persisted tree from disk."""
        tree = Tree.load(file_path)
        system = cls(tree=tree, embedding_model=embedding_model, qa_model=qa_model)
        return system
