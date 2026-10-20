"""
raptor/tree_structures.py — Core Node and Tree data structures for RAPTOR.

Represents hierarchical tree layers, parent-child links, embeddings,
and text payloads according to Sarthi et al. (ICLR 2024).
"""

import json
import pickle
import os
from typing import Dict, List, Set, Any, Optional
import numpy as np


class Node:
    """
    Represents an individual node in the RAPTOR hierarchical tree structure.
    
    Attributes:
        index: Unique integer index within the tree.
        text: Text content of the chunk (leaf) or abstractive summary (non-leaf).
        children: Set of child node indices (empty for leaf nodes).
        parents: Set of parent node indices.
        layer: Hierarchy level (0 = leaf chunks, 1 = layer 1 summaries, etc.).
        embeddings: Dict of embeddings keyed by model name (e.g. 'Ollama', 'SBERT', 'DPR').
        token_size: Token length of the node's text.
        metadata: Additional metadata (source file, cluster_id, etc.).
    """
    def __init__(
        self,
        text: str,
        index: int,
        children: Optional[Set[int]] = None,
        parents: Optional[Set[int]] = None,
        layer: int = 0,
        embeddings: Optional[Dict[str, Any]] = None,
        token_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.index = index
        self.children = set(children) if children is not None else set()
        self.parents = set(parents) if parents is not None else set()
        self.layer = layer
        self.embeddings = embeddings if embeddings is not None else {}
        self.token_size = token_size if token_size is not None else len(text.split())
        self.metadata = metadata if metadata is not None else {}

    def get_embedding(self, model_name: Optional[str] = None) -> np.ndarray:
        """Retrieve embedding for the given model or the first available embedding."""
        if not self.embeddings:
            raise ValueError(f"Node {self.index} has no embeddings.")
        if model_name and model_name in self.embeddings:
            return np.array(self.embeddings[model_name], dtype=np.float32)
        # Return first available embedding
        first_key = next(iter(self.embeddings))
        return np.array(self.embeddings[first_key], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Node to serializable dictionary."""
        emb_serializable = {}
        for k, v in self.embeddings.items():
            if isinstance(v, np.ndarray):
                emb_serializable[k] = v.tolist()
            else:
                emb_serializable[k] = v

        return {
            "index": self.index,
            "text": self.text,
            "children": list(self.children),
            "parents": list(self.parents),
            "layer": self.layer,
            "token_size": self.token_size,
            "embeddings": emb_serializable,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Reconstruct Node from dictionary."""
        embeddings = {}
        for k, v in data.get("embeddings", {}).items():
            embeddings[k] = np.array(v, dtype=np.float32)
        return cls(
            text=data["text"],
            index=data["index"],
            children=set(data.get("children", [])),
            parents=set(data.get("parents", [])),
            layer=data.get("layer", 0),
            embeddings=embeddings,
            token_size=data.get("token_size"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"Node(index={self.index}, layer={self.layer}, tokens={self.token_size}, children={len(self.children)})"


class Tree:
    """
    Represents the full hierarchical tree structure constructed by RAPTOR.
    
    Attributes:
        all_nodes: Dict mapping node index -> Node.
        root_nodes: Dict mapping node index -> root Node (highest layer nodes).
        leaf_nodes: Dict mapping node index -> leaf Node (layer 0).
        num_layers: Total number of layers in the tree.
        layer_to_nodes: Dict mapping layer index -> List[Node].
    """
    def __init__(
        self,
        all_nodes: Dict[int, Node],
        root_nodes: Dict[int, Node],
        leaf_nodes: Dict[int, Node],
        num_layers: int,
        layer_to_nodes: Dict[int, List[Node]],
    ):
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes

    def get_layer(self, layer_idx: int) -> List[Node]:
        """Get all nodes at a given layer (0 = leaves, num_layers-1 = roots)."""
        return self.layer_to_nodes.get(layer_idx, [])

    def get_all_nodes(self) -> List[Node]:
        """Get flattened list of all nodes across all layers."""
        return list(self.all_nodes.values())

    def get_children(self, node: Node) -> List[Node]:
        """Get children nodes for a given node."""
        return [self.all_nodes[idx] for idx in node.children if idx in self.all_nodes]

    def get_statistics(self) -> Dict[str, Any]:
        """Compute tree construction and compression statistics (Table 10)."""
        total_nodes = len(self.all_nodes)
        leaf_count = len(self.leaf_nodes)
        non_leaf_count = total_nodes - leaf_count

        layer_counts = {l: len(nodes) for l, nodes in self.layer_to_nodes.items()}
        
        leaf_tokens = [n.token_size for n in self.leaf_nodes.values()]
        avg_child_tokens = float(np.mean(leaf_tokens)) if leaf_tokens else 0.0

        summary_nodes = [n for n in self.all_nodes.values() if n.layer > 0]
        summary_tokens = [n.token_size for n in summary_nodes]
        avg_summary_tokens = float(np.mean(summary_tokens)) if summary_tokens else 0.0

        child_counts = [len(n.children) for n in summary_nodes]
        avg_child_per_parent = float(np.mean(child_counts)) if child_counts else 0.0

        # Compression ratio: avg summary tokens / (avg child tokens * avg child per parent)
        if avg_child_tokens > 0 and avg_child_per_parent > 0:
            compression_ratio = avg_summary_tokens / (avg_child_tokens * avg_child_per_parent)
        else:
            compression_ratio = 0.0

        return {
            "total_nodes": total_nodes,
            "leaf_nodes": leaf_count,
            "non_leaf_nodes": non_leaf_count,
            "num_layers": self.num_layers,
            "layer_counts": layer_counts,
            "avg_summary_length_tokens": round(avg_summary_tokens, 2),
            "avg_child_node_length_tokens": round(avg_child_tokens, 2),
            "avg_children_per_parent": round(avg_child_per_parent, 2),
            "avg_compression_ratio": round(compression_ratio, 3),
        }

    def save(self, file_path: str):
        """Save tree to a file (pickle or JSON based on extension)."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        if file_path.endswith(".json"):
            data = {
                "num_layers": self.num_layers,
                "all_nodes": {str(k): v.to_dict() for k, v in self.all_nodes.items()},
                "root_indices": list(self.root_nodes.keys()),
                "leaf_indices": list(self.leaf_nodes.keys()),
                "layer_to_indices": {
                    str(k): [n.index for n in v] for k, v in self.layer_to_nodes.items()
                },
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "Tree":
        """Load tree from a file."""
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_nodes = {
                int(k): Node.from_dict(v) for k, v in data["all_nodes"].items()
            }
            root_nodes = {idx: all_nodes[idx] for idx in data["root_indices"] if idx in all_nodes}
            leaf_nodes = {idx: all_nodes[idx] for idx in data["leaf_indices"] if idx in all_nodes}
            num_layers = data["num_layers"]
            layer_to_nodes = {
                int(k): [all_nodes[idx] for idx in v if idx in all_nodes]
                for k, v in data["layer_to_indices"].items()
            }
            return cls(all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes)
        else:
            with open(file_path, "rb") as f:
                return pickle.load(f)
