"""Core type definitions and dataclasses for microbiome simulation.

This module defines the fundamental data structures used throughout the system,
including phylogenetic trees, microbiome samples, and configuration objects.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PhylogeneticTree:
    """Phylogenetic tree structure.
    
    Attributes:
        num_taxa: Number of OTUs/taxa in the tree
        edges: List of (parent, child) pairs defining tree structure
        edge_lengths: Patristic distances for each edge
        taxa_names: OTU identifiers for each taxon
    """
    num_taxa: int
    edges: List[Tuple[int, int]]
    edge_lengths: List[float]
    taxa_names: List[str]
    
    def get_patristic_distance(self, i: int, j: int) -> float:
        """Compute phylogenetic distance between taxa.
        
        Args:
            i: Index of first taxon
            j: Index of second taxon
            
        Returns:
            Patristic distance between taxa i and j
        """
        if i == j:
            return 0.0
        
        # Build adjacency with distances
        adj: Dict[int, List[Tuple[int, float]]] = {k: [] for k in range(self.num_taxa)}
        for (parent, child), length in zip(self.edges, self.edge_lengths):
            adj[parent].append((child, length))
            adj[child].append((parent, length))
        
        # BFS to find path distance
        visited = {i: 0.0}
        queue = [(i, 0.0)]
        while queue:
            node, dist = queue.pop(0)
            if node == j:
                return dist
            for neighbor, edge_len in adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = dist + edge_len
                    queue.append((neighbor, dist + edge_len))
        
        return float('inf')  # Not connected


@dataclass
class MicrobiomeSample:
    """Single microbiome sample.
    
    Attributes:
        sample_id: Unique identifier for the sample
        composition: Relative abundance vector (sums to 1)
        metadata: Host characteristics (age, BMI, diet, etc.)
        timestamp: Time point for longitudinal samples
    """
    sample_id: str
    composition: np.ndarray
    metadata: Dict[str, Any]
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Validate composition is on simplex."""
        if self.composition is not None:
            if np.any(self.composition < 0):
                raise ValueError("Composition must be non-negative")
            total = np.sum(self.composition)
            if not np.isclose(total, 1.0, rtol=1e-5):
                raise ValueError(f"Composition must sum to 1, got {total}")


@dataclass
class ProcessedDataset:
    """Preprocessed dataset ready for training.
    
    Attributes:
        compositions: Array of shape (num_samples, num_taxa)
        metadata: Array of shape (num_samples, metadata_dim)
        tree: Phylogenetic tree structure
        sample_ids: List of sample identifiers
        taxa_mask: Boolean array indicating which taxa were retained
    """
    compositions: np.ndarray
    metadata: np.ndarray
    tree: PhylogeneticTree
    sample_ids: List[str]
    taxa_mask: np.ndarray


@dataclass
class LongitudinalSubject:
    """Longitudinal samples from single individual.
    
    Attributes:
        subject_id: Unique identifier for the subject
        samples: List of samples ordered by timestamp
    """
    subject_id: str
    samples: List[MicrobiomeSample] = field(default_factory=list)
    
    def __post_init__(self):
        """Sort samples by timestamp."""
        self.samples = sorted(
            self.samples, 
            key=lambda s: s.timestamp if s.timestamp is not None else 0.0
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters.
    
    Attributes:
        batch_size: Number of samples per batch
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization weight
        num_iterations: Total training iterations
        ema_decay: Exponential moving average decay
        lambda_comp: Compositional constraint loss weight
        lambda_phylo: Phylogenetic coherence loss weight
    """
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_iterations: int = 500000
    ema_decay: float = 0.9999
    lambda_comp: float = 0.1
    lambda_phylo: float = 0.05


@dataclass
class DiffusionConfig:
    """Diffusion model configuration.
    
    Attributes:
        num_timesteps: Number of diffusion steps
        beta_start: Starting noise schedule value
        beta_end: Ending noise schedule value
        image_size: Spatial resolution of rasterized images
        num_channels: Number of functional channels
    """
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    image_size: int = 256
    num_channels: int = 16
