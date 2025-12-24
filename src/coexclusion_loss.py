"""Co-exclusion Loss for realistic microbiome generation.

This module implements a loss function that penalizes co-occurrence of
taxa that are known to be mutually exclusive in real microbiome data.
This enforces biological constraints like competitive exclusion.

Co-exclusion patterns are common in microbiome data where certain taxa
compete for the same ecological niche and rarely co-occur in the same sample.

References:
    Faust, K., et al. (2012). Microbial Co-occurrence Relationships in the
    Human Microbiome.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class CoexclusionLoss(nn.Module):
    """Loss for enforcing co-exclusion patterns in generated microbiome data.
    
    This loss penalizes the co-occurrence of taxa pairs that are known to
    be mutually exclusive based on biological knowledge (competitive exclusion).
    
    The loss supports two penalty modes:
    - Soft penalty: Continuous penalty based on product of abundances
    - Hard penalty: Binary penalty when both taxa exceed a threshold
    
    Attributes:
        coexclusion_pairs: List of (taxon_i, taxon_j) index pairs
        penalty_weight: Weight for the co-exclusion penalty
        penalty_mode: 'soft' or 'hard' penalty mode
        threshold: Abundance threshold for hard penalty mode
    """
    
    def __init__(
        self,
        coexclusion_pairs: List[Tuple[int, int]],
        penalty_weight: float = 10.0,
        penalty_mode: str = 'soft',
        threshold: float = 0.001
    ):
        """Initialize co-exclusion loss.
        
        Args:
            coexclusion_pairs: List of (taxon_i, taxon_j) index pairs that
                              should not co-occur.
            penalty_weight: Weight for the co-exclusion penalty (default 10.0).
                           Higher values enforce stricter exclusion.
            penalty_mode: 'soft' for continuous penalty based on abundance product,
                         'hard' for binary penalty when both exceed threshold.
            threshold: Abundance threshold for considering a taxon "present"
                      in hard penalty mode (default 0.001 = 0.1%).
        
        Raises:
            ValueError: If coexclusion_pairs is empty or penalty_mode is invalid.
        """
        super().__init__()
        
        if not coexclusion_pairs:
            raise ValueError("coexclusion_pairs cannot be empty")
        
        if penalty_mode not in ('soft', 'hard'):
            raise ValueError(f"penalty_mode must be 'soft' or 'hard', got {penalty_mode}")
        
        self.penalty_weight = penalty_weight
        self.penalty_mode = penalty_mode
        self.threshold = threshold
        
        # Store pairs as tensor indices for efficient computation
        pairs_tensor = torch.tensor(coexclusion_pairs, dtype=torch.long)
        self.register_buffer('pair_indices_i', pairs_tensor[:, 0])
        self.register_buffer('pair_indices_j', pairs_tensor[:, 1])
        
        self.num_pairs = len(coexclusion_pairs)
    
    def forward(self, compositions: Tensor) -> Tensor:
        """Compute co-exclusion penalty for compositions.
        
        Args:
            compositions: Relative abundance tensor of shape (batch, num_taxa).
                         Values should be non-negative and typically sum to 1.
        
        Returns:
            Co-exclusion loss (scalar tensor). Higher values indicate more
            violations of co-exclusion constraints.
        """
        if self.penalty_mode == 'soft':
            return self._soft_penalty(compositions)
        else:
            return self._hard_penalty(compositions)
    
    def _soft_penalty(self, compositions: Tensor) -> Tensor:
        """Compute soft co-exclusion penalty.
        
        The soft penalty is the sum of products of abundances for each
        co-exclusion pair. This provides a smooth, differentiable penalty
        that increases as both taxa become more abundant.
        
        Penalty = sum_pairs(abundance_i * abundance_j)
        
        Args:
            compositions: Relative abundance tensor (batch, num_taxa)
        
        Returns:
            Soft penalty loss (scalar tensor)
        """
        # Get abundances for each pair
        # Shape: (batch, num_pairs)
        abundances_i = compositions[:, self.pair_indices_i]
        abundances_j = compositions[:, self.pair_indices_j]
        
        # Compute product of abundances for each pair
        # This is high when both taxa are present
        cooccurrence = abundances_i * abundances_j
        
        # Sum over pairs and average over batch
        penalty = cooccurrence.sum(dim=-1).mean()
        
        return self.penalty_weight * penalty
    
    def _hard_penalty(self, compositions: Tensor) -> Tensor:
        """Compute hard co-exclusion penalty.
        
        The hard penalty counts the number of pairs where both taxa
        exceed the presence threshold. This is less smooth but more
        directly enforces the constraint.
        
        Penalty = count(both_present) / num_pairs
        
        Args:
            compositions: Relative abundance tensor (batch, num_taxa)
        
        Returns:
            Hard penalty loss (scalar tensor)
        """
        # Get abundances for each pair
        abundances_i = compositions[:, self.pair_indices_i]
        abundances_j = compositions[:, self.pair_indices_j]
        
        # Check if each taxon exceeds threshold
        present_i = (abundances_i > self.threshold).float()
        present_j = (abundances_j > self.threshold).float()
        
        # Both present = violation
        # Use soft approximation for differentiability during training
        if self.training:
            # Sigmoid approximation of threshold
            temp = 100.0  # Temperature for sigmoid sharpness
            present_i = torch.sigmoid(temp * (abundances_i - self.threshold))
            present_j = torch.sigmoid(temp * (abundances_j - self.threshold))
        
        violations = present_i * present_j
        
        # Fraction of violations
        penalty = violations.sum(dim=-1).mean() / self.num_pairs
        
        return self.penalty_weight * penalty
    
    def compute_compliance(
        self,
        compositions: Tensor,
        threshold: Optional[float] = None
    ) -> float:
        """Compute co-exclusion compliance score.
        
        Compliance is the fraction of co-exclusion pairs that are properly
        separated (at most one taxon present per pair).
        
        Args:
            compositions: Relative abundance tensor (batch, num_taxa)
            threshold: Abundance threshold for presence (default: self.threshold)
        
        Returns:
            Compliance score in [0, 1]. Higher is better.
        """
        if threshold is None:
            threshold = self.threshold
        
        with torch.no_grad():
            # Get abundances for each pair
            abundances_i = compositions[:, self.pair_indices_i]
            abundances_j = compositions[:, self.pair_indices_j]
            
            # Check if each taxon exceeds threshold
            present_i = abundances_i > threshold
            present_j = abundances_j > threshold
            
            # Violation = both present
            violations = (present_i & present_j).float()
            
            # Compliance = 1 - violation rate
            violation_rate = violations.mean().item()
            compliance = 1.0 - violation_rate
        
        return compliance
    
    def compute_cooccurrence_rate(
        self,
        compositions: Tensor,
        threshold: Optional[float] = None
    ) -> Tensor:
        """Compute co-occurrence rate for each pair.
        
        Args:
            compositions: Relative abundance tensor (batch, num_taxa)
            threshold: Abundance threshold for presence (default: self.threshold)
        
        Returns:
            Co-occurrence rate for each pair, shape (num_pairs,)
        """
        if threshold is None:
            threshold = self.threshold
        
        with torch.no_grad():
            # Get abundances for each pair
            abundances_i = compositions[:, self.pair_indices_i]
            abundances_j = compositions[:, self.pair_indices_j]
            
            # Check if each taxon exceeds threshold
            present_i = abundances_i > threshold
            present_j = abundances_j > threshold
            
            # Co-occurrence = both present
            cooccurrence = (present_i & present_j).float()
            
            # Rate per pair (average over batch)
            rates = cooccurrence.mean(dim=0)
        
        return rates
    
    def get_pair_indices(self) -> List[Tuple[int, int]]:
        """Get the list of co-exclusion pair indices.
        
        Returns:
            List of (taxon_i, taxon_j) index pairs
        """
        return list(zip(
            self.pair_indices_i.tolist(),
            self.pair_indices_j.tolist()
        ))


def load_coexclusion_pairs(
    filepath: Union[str, Path],
    taxa_mapping: Optional[Dict[str, int]] = None
) -> List[Tuple[int, int]]:
    """Load co-exclusion pairs from a JSON file.
    
    The JSON file should have the format:
    {
        "pairs": [
            {"taxon_i": "Bacteroides", "taxon_j": "Prevotella", "source": "..."},
            ...
        ]
    }
    
    Or with numeric indices:
    {
        "pairs": [
            {"index_i": 0, "index_j": 5},
            ...
        ]
    }
    
    Args:
        filepath: Path to JSON file containing co-exclusion pairs
        taxa_mapping: Optional mapping from taxon names to indices.
                     Required if pairs are specified by name.
    
    Returns:
        List of (taxon_i, taxon_j) index pairs
    
    Raises:
        FileNotFoundError: If filepath does not exist
        ValueError: If taxa names are used but taxa_mapping is not provided
        KeyError: If a taxon name is not found in taxa_mapping
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    pairs = []
    for pair_data in data.get('pairs', []):
        # Check if indices are provided directly
        if 'index_i' in pair_data and 'index_j' in pair_data:
            pairs.append((pair_data['index_i'], pair_data['index_j']))
        # Otherwise use taxon names
        elif 'taxon_i' in pair_data and 'taxon_j' in pair_data:
            if taxa_mapping is None:
                raise ValueError(
                    "taxa_mapping required when pairs are specified by name"
                )
            taxon_i = pair_data['taxon_i']
            taxon_j = pair_data['taxon_j']
            
            if taxon_i not in taxa_mapping:
                raise KeyError(f"Taxon '{taxon_i}' not found in taxa_mapping")
            if taxon_j not in taxa_mapping:
                raise KeyError(f"Taxon '{taxon_j}' not found in taxa_mapping")
            
            pairs.append((taxa_mapping[taxon_i], taxa_mapping[taxon_j]))
    
    return pairs


def create_coexclusion_loss_from_file(
    filepath: Union[str, Path],
    taxa_mapping: Optional[Dict[str, int]] = None,
    penalty_weight: float = 10.0,
    penalty_mode: str = 'soft',
    threshold: float = 0.001
) -> CoexclusionLoss:
    """Create a CoexclusionLoss from a JSON configuration file.
    
    Convenience function that loads pairs and creates the loss in one step.
    
    Args:
        filepath: Path to JSON file containing co-exclusion pairs
        taxa_mapping: Optional mapping from taxon names to indices
        penalty_weight: Weight for the co-exclusion penalty
        penalty_mode: 'soft' or 'hard' penalty mode
        threshold: Abundance threshold for hard penalty mode
    
    Returns:
        Configured CoexclusionLoss instance
    """
    pairs = load_coexclusion_pairs(filepath, taxa_mapping)
    return CoexclusionLoss(
        coexclusion_pairs=pairs,
        penalty_weight=penalty_weight,
        penalty_mode=penalty_mode,
        threshold=threshold
    )


def get_default_coexclusion_pairs() -> List[Tuple[int, int]]:
    """Get the default co-exclusion pairs from the bundled configuration.
    
    These pairs are based on well-documented competitive exclusion patterns
    in human gut microbiome literature, particularly the enterotype signatures.
    
    Returns:
        List of (taxon_i, taxon_j) index pairs
    
    Note:
        The indices correspond to a standard taxa ordering. When using with
        a specific dataset, you may need to remap indices based on your
        taxa ordering.
    """
    # Default pairs based on enterotype signatures
    # These are the most well-documented co-exclusion patterns
    default_pairs = [
        (0, 1),   # Bacteroides vs Prevotella (primary enterotype)
        (0, 2),   # Bacteroides vs Ruminococcus
        (1, 2),   # Prevotella vs Ruminococcus
        (3, 4),   # Species-level exclusion 1
        (5, 4),   # Species-level exclusion 2
    ]
    return default_pairs


def get_coexclusion_config_path() -> Path:
    """Get the path to the default co-exclusion configuration file.
    
    Returns:
        Path to configs/coexclusion_pairs.json
    """
    # Navigate from src/ to configs/
    module_dir = Path(__file__).parent
    config_path = module_dir.parent / 'configs' / 'coexclusion_pairs.json'
    return config_path


def load_default_coexclusion_pairs(
    taxa_mapping: Optional[Dict[str, int]] = None,
    use_index_pairs: bool = True
) -> List[Tuple[int, int]]:
    """Load co-exclusion pairs from the default configuration file.
    
    Args:
        taxa_mapping: Optional mapping from taxon names to indices.
                     If provided and use_index_pairs=False, pairs will be 
                     loaded by taxon name.
        use_index_pairs: If True, load pairs from 'pairs_by_index' section
                        which uses numeric indices. If False, load from
                        'pairs' section which uses taxon names.
    
    Returns:
        List of (taxon_i, taxon_j) index pairs
    
    Raises:
        FileNotFoundError: If the default config file is not found
    """
    config_path = get_coexclusion_config_path()
    
    if not config_path.exists():
        # Fall back to default pairs if config not found
        return get_default_coexclusion_pairs()
    
    # Load the config file
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    pairs = []
    
    if use_index_pairs and 'pairs_by_index' in data:
        # Load from pairs_by_index section (numeric indices)
        for pair_data in data['pairs_by_index']:
            pairs.append((pair_data['index_i'], pair_data['index_j']))
    elif taxa_mapping is not None:
        # Load from pairs section using taxa names
        for pair_data in data.get('pairs', []):
            if 'taxon_i' in pair_data and 'taxon_j' in pair_data:
                taxon_i = pair_data['taxon_i']
                taxon_j = pair_data['taxon_j']
                if taxon_i in taxa_mapping and taxon_j in taxa_mapping:
                    pairs.append((taxa_mapping[taxon_i], taxa_mapping[taxon_j]))
    else:
        # Fall back to hardcoded defaults
        return get_default_coexclusion_pairs()
    
    return pairs if pairs else get_default_coexclusion_pairs()
