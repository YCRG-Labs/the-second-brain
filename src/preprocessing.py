"""Preprocessing pipeline for microbiome data.

This module implements data preprocessing including sample filtering,
rarefaction, and taxa filtering to prepare raw microbiome data for training.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.types import MicrobiomeSample, ProcessedDataset, PhylogeneticTree
from src.exceptions import InsufficientReadsError


class PreprocessingPipeline:
    """Data preprocessing for microbiome samples.
    
    This class implements the full preprocessing pipeline including:
    - Sample filtering by read depth
    - Rarefaction to fixed depth
    - Taxa filtering by prevalence
    
    Attributes:
        min_reads: Minimum read count threshold for sample filtering
        rarefaction_depth: Target depth for rarefaction
        min_prevalence: Minimum prevalence threshold for taxa filtering
        reference_db: Reference database for taxonomy (not used in current implementation)
    """
    
    def __init__(
        self,
        min_reads: int = 10000,
        rarefaction_depth: int = 10000,
        min_prevalence: float = 0.05,
        reference_db: str = "silva_v138"
    ):
        """Initialize preprocessing pipeline.
        
        Args:
            min_reads: Minimum read count for sample filtering
            rarefaction_depth: Target depth for rarefaction
            min_prevalence: Minimum prevalence (fraction of samples) for taxa
            reference_db: Reference database name (for future use)
        """
        self.min_reads = min_reads
        self.rarefaction_depth = rarefaction_depth
        self.min_prevalence = min_prevalence
        self.reference_db = reference_db
    
    def filter_samples(
        self, 
        samples: List[MicrobiomeSample],
        counts: np.ndarray
    ) -> tuple[List[MicrobiomeSample], np.ndarray]:
        """Remove samples below read threshold.
        
        Filters out samples that have fewer than min_reads total reads.
        
        Args:
            samples: List of microbiome samples
            counts: Raw count matrix of shape (num_samples, num_taxa)
            
        Returns:
            Tuple of (filtered_samples, filtered_counts)
            
        Raises:
            InsufficientReadsError: If a sample has insufficient reads (logged as warning)
        """
        filtered_samples = []
        filtered_counts = []
        
        for i, sample in enumerate(samples):
            total_reads = np.sum(counts[i])
            
            if total_reads >= self.min_reads:
                filtered_samples.append(sample)
                filtered_counts.append(counts[i])
            else:
                # Log warning but don't raise - just skip the sample
                import warnings
                warnings.warn(
                    f"Sample {sample.sample_id} has {total_reads} reads "
                    f"(< {self.min_reads}), skipping",
                    UserWarning
                )
        
        if len(filtered_samples) == 0:
            raise ValueError("No samples passed filtering threshold")
        
        return filtered_samples, np.array(filtered_counts)
    
    def rarefy(self, counts: np.ndarray) -> np.ndarray:
        """Subsample to fixed depth.
        
        Performs rarefaction by randomly subsampling reads to exactly
        rarefaction_depth reads per sample.
        
        Args:
            counts: Count matrix of shape (num_samples, num_taxa)
            
        Returns:
            Rarefied count matrix of same shape
            
        Raises:
            ValueError: If any sample has fewer reads than rarefaction_depth
        """
        num_samples, num_taxa = counts.shape
        rarefied = np.zeros_like(counts)
        
        for i in range(num_samples):
            total_reads = np.sum(counts[i])
            
            if total_reads < self.rarefaction_depth:
                raise ValueError(
                    f"Sample {i} has {total_reads} reads, "
                    f"cannot rarefy to {self.rarefaction_depth}"
                )
            
            # Create pool of reads (each taxon appears counts[i, j] times)
            read_pool = []
            for j in range(num_taxa):
                read_pool.extend([j] * int(counts[i, j]))
            
            # Randomly subsample
            np.random.shuffle(read_pool)
            subsampled = read_pool[:self.rarefaction_depth]
            
            # Count occurrences
            for taxon_idx in subsampled:
                rarefied[i, taxon_idx] += 1
        
        return rarefied
    
    def filter_taxa(
        self, 
        counts: np.ndarray, 
        prevalence_threshold: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove rare taxa below prevalence threshold.
        
        Filters out taxa that are present in fewer than prevalence_threshold
        fraction of samples.
        
        Args:
            counts: Count matrix of shape (num_samples, num_taxa)
            prevalence_threshold: Minimum prevalence (uses self.min_prevalence if None)
            
        Returns:
            Tuple of (filtered_counts, taxa_mask)
            - filtered_counts: Count matrix with rare taxa removed
            - taxa_mask: Boolean array indicating which taxa were retained
        """
        if prevalence_threshold is None:
            prevalence_threshold = self.min_prevalence
        
        num_samples, num_taxa = counts.shape
        
        # Count how many samples each taxon appears in (presence/absence)
        presence = (counts > 0).astype(int)
        prevalence = np.sum(presence, axis=0) / num_samples
        
        # Create mask for taxa that meet threshold
        taxa_mask = prevalence >= prevalence_threshold
        
        # Filter counts
        filtered_counts = counts[:, taxa_mask]
        
        return filtered_counts, taxa_mask
    
    def process(
        self,
        samples: List[MicrobiomeSample],
        counts: np.ndarray,
        tree: PhylogeneticTree,
        metadata_matrix: Optional[np.ndarray] = None
    ) -> ProcessedDataset:
        """Full preprocessing pipeline.
        
        Applies the complete preprocessing workflow:
        1. Filter samples by read depth
        2. Rarefy to fixed depth
        3. Filter rare taxa
        4. Convert to compositions
        5. Update phylogenetic tree
        
        Args:
            samples: List of microbiome samples
            counts: Raw count matrix of shape (num_samples, num_taxa)
            tree: Phylogenetic tree for all taxa
            metadata_matrix: Optional metadata matrix (num_samples, metadata_dim)
                           If None, extracted from samples
            
        Returns:
            ProcessedDataset with compositions, metadata, and filtered tree
        """
        # Step 1: Filter samples
        filtered_samples, filtered_counts = self.filter_samples(samples, counts)
        
        # Step 2: Rarefy
        rarefied_counts = self.rarefy(filtered_counts)
        
        # Step 3: Filter taxa
        final_counts, taxa_mask = self.filter_taxa(rarefied_counts)
        
        # Step 4: Convert to compositions (normalize to sum to 1)
        compositions = final_counts / np.sum(final_counts, axis=1, keepdims=True)
        
        # Step 5: Extract metadata
        if metadata_matrix is None:
            # Extract metadata from samples
            # For now, we'll create a simple numeric representation
            # In practice, this would involve proper encoding of categorical variables
            metadata_list = []
            for sample in filtered_samples:
                # Convert metadata dict to array (simplified)
                # This is a placeholder - real implementation would need proper encoding
                meta_values = []
                for key in sorted(sample.metadata.keys()):
                    val = sample.metadata[key]
                    if isinstance(val, (int, float)):
                        meta_values.append(float(val))
                    else:
                        # For non-numeric, use hash as placeholder
                        meta_values.append(float(hash(str(val)) % 1000))
                metadata_list.append(meta_values)
            
            # Pad to same length
            max_len = max(len(m) for m in metadata_list) if metadata_list else 0
            metadata_matrix = np.zeros((len(metadata_list), max_len))
            for i, meta in enumerate(metadata_list):
                metadata_matrix[i, :len(meta)] = meta
        else:
            # Use provided metadata, but filter to match samples
            sample_indices = [samples.index(s) for s in filtered_samples]
            metadata_matrix = metadata_matrix[sample_indices]
        
        # Step 6: Update phylogenetic tree
        # Filter tree to only include retained taxa
        retained_indices = np.where(taxa_mask)[0]
        
        # Create mapping from old to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(retained_indices)}
        
        # Filter edges and edge lengths
        new_edges = []
        new_edge_lengths = []
        for (parent, child), length in zip(tree.edges, tree.edge_lengths):
            if parent in old_to_new and child in old_to_new:
                new_edges.append((old_to_new[parent], old_to_new[child]))
                new_edge_lengths.append(length)
        
        # Filter taxa names
        new_taxa_names = [tree.taxa_names[i] for i in retained_indices]
        
        # Create new tree
        filtered_tree = PhylogeneticTree(
            num_taxa=len(retained_indices),
            edges=new_edges,
            edge_lengths=new_edge_lengths,
            taxa_names=new_taxa_names
        )
        
        # Extract sample IDs
        sample_ids = [s.sample_id for s in filtered_samples]
        
        return ProcessedDataset(
            compositions=compositions,
            metadata=metadata_matrix,
            tree=filtered_tree,
            sample_ids=sample_ids,
            taxa_mask=taxa_mask
        )
