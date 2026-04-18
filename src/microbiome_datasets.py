"""Data preprocessing for real microbiome datasets.

This module implements data loading and preprocessing for:
- American Gut Project (AGP)
- Human Microbiome Project (HMP)

Provides standardized OTU tables with computed statistics for training
realistic microbiome generation models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
import warnings
import urllib.request
import zipfile
import gzip
import shutil
import os


# URLs for public microbiome datasets
DATASET_URLS = {
    'american_gut': {
        # GitHub mirror of American Gut data
        'github': 'https://github.com/biocore/American-Gut/raw/master/data/AG/AG_100nt.biom',
        # Qiita public download (may require authentication)
        'qiita': 'https://qiita.ucsd.edu/public_artifact_download/?artifact_id=77054',
    },
    'hmp': {
        # HMP Data Portal - 16S rRNA data (alternative URLs)
        'v35_otu': 'https://portal.hmpdacc.org/files/otu_table_psn_v35.txt.gz',
        'v13_otu': 'https://portal.hmpdacc.org/files/otu_table_psn_v13.txt.gz',
        # NCBI mirror
        'ncbi': 'https://ftp.ncbi.nlm.nih.gov/genomes/HUMAN_MICROBIOM/HMP_REFERENCE_GENOMES/',
    }
}


def _download_with_ssl_context(url: str, output_path: str, verbose: bool = True) -> None:
    """Download file with SSL context handling for problematic certificates."""
    import ssl
    
    def _progress_hook(count, block_size, total_size):
        if verbose and total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}%", end='', flush=True)
    
    try:
        # First try normal download
        urllib.request.urlretrieve(url, output_path, _progress_hook)
    except urllib.error.URLError as e:
        if 'CERTIFICATE_VERIFY_FAILED' in str(e) or 'SSL' in str(e):
            # Try with unverified SSL context
            if verbose:
                print("\nSSL verification failed, trying with unverified context...")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, output_path, _progress_hook)
        else:
            raise


@dataclass
class MicrobiomeDatasetStats:
    """Statistics computed from a microbiome dataset.
    
    Attributes:
        mean_sparsity: Mean fraction of zeros per sample
        std_sparsity: Standard deviation of sparsity
        taxon_prevalences: Prevalence (fraction of samples present) per taxon
        alpha_diversity_mean: Mean Shannon entropy
        alpha_diversity_std: Std of Shannon entropy
        beta_diversity_mean: Mean pairwise Bray-Curtis
        beta_diversity_std: Std of pairwise Bray-Curtis
        cooccurrence_matrix: Pairwise co-occurrence rates
        num_samples: Number of samples
        num_taxa: Number of taxa after filtering
    """
    mean_sparsity: float
    std_sparsity: float
    taxon_prevalences: np.ndarray
    alpha_diversity_mean: float
    alpha_diversity_std: float
    beta_diversity_mean: float
    beta_diversity_std: float
    cooccurrence_matrix: Optional[np.ndarray] = None
    num_samples: int = 0
    num_taxa: int = 0


@dataclass
class ProcessedMicrobiomeDataset:
    """Processed microbiome dataset ready for training.
    
    Attributes:
        compositions: Relative abundance matrix (num_samples, num_taxa)
        counts: Raw count matrix (num_samples, num_taxa) if available
        sample_ids: List of sample identifiers
        taxa_names: List of taxon names
        stats: Computed dataset statistics
        metadata: Optional sample metadata
    """
    compositions: np.ndarray
    counts: Optional[np.ndarray]
    sample_ids: List[str]
    taxa_names: List[str]
    stats: MicrobiomeDatasetStats
    metadata: Optional[Dict[str, np.ndarray]] = None


def compute_shannon_entropy(compositions: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute Shannon entropy (alpha diversity) for each sample.
    
    Args:
        compositions: Relative abundance matrix (num_samples, num_taxa)
        eps: Small constant to avoid log(0)
        
    Returns:
        Array of Shannon entropy values (num_samples,)
    """
    # Mask zeros to avoid log(0)
    p = np.clip(compositions, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def compute_bray_curtis(compositions: np.ndarray) -> np.ndarray:
    """Compute pairwise Bray-Curtis dissimilarity.
    
    Args:
        compositions: Relative abundance matrix (num_samples, num_taxa)
        
    Returns:
        Pairwise distance matrix (num_samples, num_samples)
    """
    n = compositions.shape[0]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            numerator = np.sum(np.abs(compositions[i] - compositions[j]))
            denominator = np.sum(compositions[i]) + np.sum(compositions[j])
            if denominator > 0:
                bc = numerator / denominator
            else:
                bc = 0.0
            distances[i, j] = bc
            distances[j, i] = bc
    
    return distances


def compute_cooccurrence_matrix(
    compositions: np.ndarray, 
    threshold: float = 0.0
) -> np.ndarray:
    """Compute pairwise co-occurrence rates between taxa.
    
    Args:
        compositions: Relative abundance matrix (num_samples, num_taxa)
        threshold: Minimum abundance to consider taxon present
        
    Returns:
        Co-occurrence matrix (num_taxa, num_taxa) with rates in [0, 1]
    """
    presence = (compositions > threshold).astype(float)
    num_samples = compositions.shape[0]
    
    # Co-occurrence: both taxa present in same sample
    cooccurrence = presence.T @ presence / num_samples
    
    return cooccurrence


def compute_dataset_stats(
    compositions: np.ndarray,
    compute_cooccurrence: bool = True
) -> MicrobiomeDatasetStats:
    """Compute comprehensive statistics from a microbiome dataset.
    
    Args:
        compositions: Relative abundance matrix (num_samples, num_taxa)
        compute_cooccurrence: Whether to compute co-occurrence matrix
        
    Returns:
        MicrobiomeDatasetStats with all computed statistics
    """
    num_samples, num_taxa = compositions.shape
    
    # Sparsity (fraction of zeros per sample)
    sparsity_per_sample = np.mean(compositions == 0, axis=1)
    mean_sparsity = float(np.mean(sparsity_per_sample))
    std_sparsity = float(np.std(sparsity_per_sample))
    
    # Taxon prevalences (fraction of samples where taxon is present)
    taxon_prevalences = np.mean(compositions > 0, axis=0)
    
    # Alpha diversity (Shannon entropy)
    alpha_div = compute_shannon_entropy(compositions)
    alpha_diversity_mean = float(np.mean(alpha_div))
    alpha_diversity_std = float(np.std(alpha_div))
    
    # Beta diversity (Bray-Curtis)
    # For large datasets, sample pairs to avoid O(n^2) computation
    if num_samples > 500:
        # Sample 500 random pairs
        idx1 = np.random.choice(num_samples, 500, replace=True)
        idx2 = np.random.choice(num_samples, 500, replace=True)
        beta_values = []
        for i, j in zip(idx1, idx2):
            if i != j:
                num = np.sum(np.abs(compositions[i] - compositions[j]))
                denom = np.sum(compositions[i]) + np.sum(compositions[j])
                if denom > 0:
                    beta_values.append(num / denom)
        beta_values = np.array(beta_values)
    else:
        bc_matrix = compute_bray_curtis(compositions)
        # Get upper triangle (excluding diagonal)
        beta_values = bc_matrix[np.triu_indices(num_samples, k=1)]
    
    beta_diversity_mean = float(np.mean(beta_values))
    beta_diversity_std = float(np.std(beta_values))
    
    # Co-occurrence matrix
    cooccurrence_matrix = None
    if compute_cooccurrence:
        cooccurrence_matrix = compute_cooccurrence_matrix(compositions)
    
    return MicrobiomeDatasetStats(
        mean_sparsity=mean_sparsity,
        std_sparsity=std_sparsity,
        taxon_prevalences=taxon_prevalences,
        alpha_diversity_mean=alpha_diversity_mean,
        alpha_diversity_std=alpha_diversity_std,
        beta_diversity_mean=beta_diversity_mean,
        beta_diversity_std=beta_diversity_std,
        cooccurrence_matrix=cooccurrence_matrix,
        num_samples=num_samples,
        num_taxa=num_taxa
    )



class MicrobiomeDatasetPreprocessor:
    """Preprocessor for microbiome OTU tables.
    
    Handles filtering by prevalence and abundance, normalization,
    and statistics computation.
    """
    
    def __init__(
        self,
        min_prevalence: float = 0.01,
        min_abundance: float = 1e-5,
        max_taxa: Optional[int] = None,
        rarefaction_depth: Optional[int] = None
    ):
        """Initialize preprocessor.
        
        Args:
            min_prevalence: Minimum fraction of samples a taxon must appear in
            min_abundance: Minimum mean relative abundance for a taxon
            max_taxa: Maximum number of taxa to retain (by prevalence)
            rarefaction_depth: If set, rarefy counts to this depth
        """
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.max_taxa = max_taxa
        self.rarefaction_depth = rarefaction_depth
    
    def filter_taxa(
        self,
        counts: np.ndarray,
        taxa_names: List[str]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Filter taxa by prevalence and abundance.
        
        Args:
            counts: Raw count matrix (num_samples, num_taxa)
            taxa_names: List of taxon names
            
        Returns:
            Tuple of (filtered_counts, filtered_names, taxa_mask)
        """
        num_samples, num_taxa = counts.shape
        
        # Compute relative abundances for filtering
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        rel_abundance = counts / row_sums
        
        # Prevalence filter
        prevalence = np.mean(counts > 0, axis=0)
        prevalence_mask = prevalence >= self.min_prevalence
        
        # Abundance filter
        mean_abundance = np.mean(rel_abundance, axis=0)
        abundance_mask = mean_abundance >= self.min_abundance
        
        # Combined mask
        taxa_mask = prevalence_mask & abundance_mask
        
        # If max_taxa specified, keep top by prevalence
        if self.max_taxa is not None and np.sum(taxa_mask) > self.max_taxa:
            # Get indices of taxa passing filter, sorted by prevalence
            passing_indices = np.where(taxa_mask)[0]
            sorted_by_prev = passing_indices[np.argsort(-prevalence[passing_indices])]
            top_indices = sorted_by_prev[:self.max_taxa]
            
            # Create new mask with only top taxa
            taxa_mask = np.zeros(num_taxa, dtype=bool)
            taxa_mask[top_indices] = True
        
        # Apply filter
        filtered_counts = counts[:, taxa_mask]
        filtered_names = [taxa_names[i] for i in range(num_taxa) if taxa_mask[i]]
        
        return filtered_counts, filtered_names, taxa_mask
    
    def rarefy(self, counts: np.ndarray, depth: int) -> np.ndarray:
        """Rarefy counts to fixed depth.
        
        Args:
            counts: Count matrix (num_samples, num_taxa)
            depth: Target rarefaction depth
            
        Returns:
            Rarefied count matrix
        """
        num_samples, num_taxa = counts.shape
        rarefied = np.zeros_like(counts)
        
        for i in range(num_samples):
            total = int(counts[i].sum())
            if total < depth:
                # Skip samples with insufficient reads
                continue
            
            # Create pool of taxon indices
            pool = np.repeat(np.arange(num_taxa), counts[i].astype(int))
            
            # Random subsample
            subsampled = np.random.choice(pool, size=depth, replace=False)
            
            # Count occurrences
            for taxon_idx in subsampled:
                rarefied[i, taxon_idx] += 1
        
        return rarefied
    
    def normalize_to_compositions(self, counts: np.ndarray) -> np.ndarray:
        """Convert counts to relative abundances (compositions).
        
        Args:
            counts: Count matrix (num_samples, num_taxa)
            
        Returns:
            Composition matrix where each row sums to 1
        """
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        return counts / row_sums
    
    def process(
        self,
        counts: np.ndarray,
        taxa_names: List[str],
        sample_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None
    ) -> ProcessedMicrobiomeDataset:
        """Full preprocessing pipeline.
        
        Args:
            counts: Raw count matrix (num_samples, num_taxa)
            taxa_names: List of taxon names
            sample_ids: Optional list of sample IDs
            metadata: Optional metadata dictionary
            
        Returns:
            ProcessedMicrobiomeDataset with compositions and statistics
        """
        num_samples = counts.shape[0]
        
        # Generate sample IDs if not provided
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(num_samples)]
        
        # Rarefaction if specified
        if self.rarefaction_depth is not None:
            counts = self.rarefy(counts, self.rarefaction_depth)
            # Remove samples with zero counts after rarefaction
            valid_samples = counts.sum(axis=1) > 0
            counts = counts[valid_samples]
            sample_ids = [sid for sid, valid in zip(sample_ids, valid_samples) if valid]
            if metadata is not None:
                metadata = {k: v[valid_samples] for k, v in metadata.items()}
        
        # Filter taxa
        filtered_counts, filtered_names, taxa_mask = self.filter_taxa(counts, taxa_names)
        
        # Normalize to compositions
        compositions = self.normalize_to_compositions(filtered_counts)
        
        # Compute statistics
        stats = compute_dataset_stats(compositions)
        
        return ProcessedMicrobiomeDataset(
            compositions=compositions,
            counts=filtered_counts,
            sample_ids=sample_ids,
            taxa_names=filtered_names,
            stats=stats,
            metadata=metadata
        )



class AmericanGutDataset:
    """Loader and preprocessor for American Gut Project data.
    
    The American Gut Project is one of the largest crowd-sourced
    microbiome studies with thousands of fecal samples.
    
    Example:
        >>> dataset = AmericanGutDataset()
        >>> # Download real data
        >>> biom_path = dataset.download()
        >>> processed = dataset.load_and_preprocess(biom_path)
        >>> # Or use synthetic data for testing
        >>> processed = dataset.generate_synthetic(num_samples=1000)
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        min_prevalence: float = 0.01,
        min_abundance: float = 1e-5,
        max_taxa: int = 500
    ):
        """Initialize American Gut dataset loader.
        
        Args:
            data_dir: Directory containing AGP data files
            min_prevalence: Minimum taxon prevalence threshold
            min_abundance: Minimum mean abundance threshold
            max_taxa: Maximum number of taxa to retain
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.preprocessor = MicrobiomeDatasetPreprocessor(
            min_prevalence=min_prevalence,
            min_abundance=min_abundance,
            max_taxa=max_taxa
        )
        self._processed_data: Optional[ProcessedMicrobiomeDataset] = None
    
    def download(
        self,
        output_dir: Optional[str] = None,
        source: str = 'qiita',
        verbose: bool = True
    ) -> str:
        """Download American Gut Project data from public sources.
        
        Args:
            output_dir: Directory to save downloaded files (default: ./data/american_gut)
            source: Data source - 'qiita' or 'ftp'
            verbose: Print progress messages
            
        Returns:
            Path to downloaded BIOM file
            
        Note:
            Requires biom-format package to load the downloaded file.
            Install with: pip install biom-format
        """
        if output_dir is None:
            output_dir = Path('./data/american_gut')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'ag_otu_table.biom'
        
        if output_file.exists():
            if verbose:
                print(f"File already exists: {output_file}")
            return str(output_file)
        
        # Try multiple sources in order (github is most reliable)
        sources_to_try = ['github', 'qiita'] if source == 'qiita' else [source]
        
        def _progress_hook(count, block_size, total_size):
            if verbose and total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownloading: {percent}%", end='', flush=True)
        
        last_error = None
        for src in sources_to_try:
            url = DATASET_URLS['american_gut'].get(src)
            if not url:
                continue
                
            if verbose:
                print(f"Trying to download American Gut data from {src}...")
                print(f"URL: {url}")
            
            try:
                _download_with_ssl_context(url, str(output_file), verbose=verbose)
                
                if verbose:
                    print(f"\nDownloaded to: {output_file}")
                
                self.data_dir = output_dir
                return str(output_file)
                
            except Exception as e:
                last_error = e
                if verbose:
                    print(f"\nFailed from {src}: {e}")
                # Clean up partial download
                if output_file.exists():
                    output_file.unlink()
                continue
        
        # If all sources failed, raise error
        raise RuntimeError(
            f"Failed to download American Gut data from all sources. "
            f"Last error: {last_error}\n"
            f"You can manually download from: https://qiita.ucsd.edu/study/description/10317"
        )
    
    def load_and_preprocess(
        self,
        biom_path: Optional[str] = None,
        download_if_missing: bool = True
    ) -> ProcessedMicrobiomeDataset:
        """Load and preprocess American Gut data.
        
        Args:
            biom_path: Path to BIOM file (downloads if None and download_if_missing=True)
            download_if_missing: Download data if not found locally
            
        Returns:
            ProcessedMicrobiomeDataset ready for training
        """
        if biom_path is None:
            if self.data_dir:
                # Look for existing files
                for pattern in ['*.biom', 'ag*.biom']:
                    files = list(self.data_dir.glob(pattern))
                    if files:
                        biom_path = str(files[0])
                        break
            
            if biom_path is None and download_if_missing:
                biom_path = self.download()
            elif biom_path is None:
                raise ValueError("No BIOM file found. Set download_if_missing=True or provide biom_path.")
        
        # Load data
        counts, sample_ids, taxa_names = self.load_from_biom(biom_path)
        
        # Preprocess
        self._processed_data = self.preprocessor.process(
            counts=counts,
            taxa_names=taxa_names,
            sample_ids=sample_ids
        )
        
        return self._processed_data
    
    def load_from_biom(self, biom_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load data from BIOM format file.
        
        Args:
            biom_path: Path to BIOM file
            
        Returns:
            Tuple of (counts, sample_ids, taxa_names)
        """
        try:
            import biom
            table = biom.load_table(biom_path)
            counts = table.matrix_data.toarray().T  # samples x taxa
            sample_ids = list(table.ids(axis='sample'))
            taxa_names = list(table.ids(axis='observation'))
            return counts, sample_ids, taxa_names
        except ImportError:
            raise ImportError(
                "biom-format package required. Install with: pip install biom-format"
            )
    
    def load_from_tsv(
        self, 
        otu_path: str,
        transpose: bool = True
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load data from TSV OTU table.
        
        Args:
            otu_path: Path to TSV file
            transpose: If True, transpose so rows are samples
            
        Returns:
            Tuple of (counts, sample_ids, taxa_names)
        """
        import csv
        
        with open(otu_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            data = []
            row_ids = []
            for row in reader:
                row_ids.append(row[0])
                data.append([float(x) if x else 0.0 for x in row[1:]])
        
        counts = np.array(data)
        col_ids = header[1:]
        
        if transpose:
            counts = counts.T
            sample_ids = col_ids
            taxa_names = row_ids
        else:
            sample_ids = row_ids
            taxa_names = col_ids
        
        return counts, sample_ids, taxa_names
    
    def generate_synthetic(
        self,
        num_samples: int = 1000,
        num_taxa: int = 500,
        sparsity: float = 0.7,
        seed: Optional[int] = None
    ) -> ProcessedMicrobiomeDataset:
        """Generate synthetic data mimicking American Gut characteristics.
        
        Useful for testing when real data is not available.
        
        Args:
            num_samples: Number of samples to generate
            num_taxa: Number of taxa
            sparsity: Target sparsity level
            seed: Random seed for reproducibility
            
        Returns:
            ProcessedMicrobiomeDataset with synthetic data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate taxa prevalences (power law distribution)
        prevalences = np.random.power(0.3, num_taxa)
        prevalences = np.sort(prevalences)[::-1]  # Sort descending
        
        # Generate compositions
        compositions = np.zeros((num_samples, num_taxa))
        
        for i in range(num_samples):
            # Determine which taxa are present based on prevalence
            presence = np.random.random(num_taxa) < prevalences
            
            # Adjust to achieve target sparsity
            current_sparsity = 1 - np.mean(presence)
            if current_sparsity < sparsity:
                # Need more zeros - randomly zero out some present taxa
                present_idx = np.where(presence)[0]
                n_to_zero = int((sparsity - current_sparsity) * num_taxa)
                if n_to_zero > 0 and len(present_idx) > n_to_zero:
                    zero_idx = np.random.choice(present_idx, n_to_zero, replace=False)
                    presence[zero_idx] = False
            
            # Generate abundances for present taxa (log-normal)
            abundances = np.zeros(num_taxa)
            present_idx = np.where(presence)[0]
            if len(present_idx) > 0:
                log_abundances = np.random.normal(-3, 1.5, len(present_idx))
                abundances[present_idx] = np.exp(log_abundances)
            
            # Normalize to composition
            if abundances.sum() > 0:
                compositions[i] = abundances / abundances.sum()
        
        # Generate sample IDs and taxa names
        sample_ids = [f"AGP_{i:06d}" for i in range(num_samples)]
        taxa_names = [f"OTU_{i:05d}" for i in range(num_taxa)]
        
        # Compute statistics
        stats = compute_dataset_stats(compositions)
        
        self._processed_data = ProcessedMicrobiomeDataset(
            compositions=compositions,
            counts=None,
            sample_ids=sample_ids,
            taxa_names=taxa_names,
            stats=stats,
            metadata=None
        )
        
        return self._processed_data
    
    def load(
        self,
        path: Optional[str] = None,
        format: str = 'auto'
    ) -> ProcessedMicrobiomeDataset:
        """Load and preprocess American Gut data.
        
        Args:
            path: Path to data file (BIOM or TSV)
            format: 'biom', 'tsv', or 'auto' to detect from extension
            
        Returns:
            ProcessedMicrobiomeDataset
        """
        if path is None:
            if self.data_dir is None:
                # Generate synthetic data as fallback
                warnings.warn(
                    "No data path provided, generating synthetic American Gut data"
                )
                return self.generate_synthetic()
            path = str(self.data_dir / "otu_table.biom")
        
        # Detect format
        if format == 'auto':
            if path.endswith('.biom'):
                format = 'biom'
            else:
                format = 'tsv'
        
        # Load raw data
        if format == 'biom':
            counts, sample_ids, taxa_names = self.load_from_biom(path)
        else:
            counts, sample_ids, taxa_names = self.load_from_tsv(path)
        
        # Preprocess
        self._processed_data = self.preprocessor.process(
            counts=counts,
            taxa_names=taxa_names,
            sample_ids=sample_ids
        )
        
        return self._processed_data
    
    @property
    def data(self) -> Optional[ProcessedMicrobiomeDataset]:
        """Get processed data if loaded."""
        return self._processed_data



class HMPDataset:
    """Loader and preprocessor for Human Microbiome Project data.
    
    The Human Microbiome Project provides reference microbiome data
    from multiple body sites of healthy adults.
    
    Example:
        >>> dataset = HMPDataset()
        >>> # Download real data
        >>> tsv_path = dataset.download()
        >>> processed = dataset.load_and_preprocess(tsv_path)
        >>> # Or use synthetic data for testing
        >>> processed = dataset.generate_synthetic(num_samples=500)
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        body_site: str = 'stool',
        min_prevalence: float = 0.01,
        min_abundance: float = 1e-5,
        max_taxa: int = 500
    ):
        """Initialize HMP dataset loader.
        
        Args:
            data_dir: Directory containing HMP data files
            body_site: Body site to load ('stool', 'oral', 'skin', etc.)
            min_prevalence: Minimum taxon prevalence threshold
            min_abundance: Minimum mean abundance threshold
            max_taxa: Maximum number of taxa to retain
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.body_site = body_site
        self.preprocessor = MicrobiomeDatasetPreprocessor(
            min_prevalence=min_prevalence,
            min_abundance=min_abundance,
            max_taxa=max_taxa
        )
        self._processed_data: Optional[ProcessedMicrobiomeDataset] = None
    
    def download(
        self,
        output_dir: Optional[str] = None,
        region: str = 'v35',
        verbose: bool = True
    ) -> str:
        """Download Human Microbiome Project data from public sources.
        
        Args:
            output_dir: Directory to save downloaded files (default: ./data/hmp)
            region: 16S region - 'v35' or 'v13'
            verbose: Print progress messages
            
        Returns:
            Path to downloaded OTU table file
        """
        if output_dir is None:
            output_dir = Path('./data/hmp')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        url = DATASET_URLS['hmp'][f'{region}_otu']
        gz_file = output_dir / f'otu_table_psn_{region}.txt.gz'
        output_file = output_dir / f'otu_table_psn_{region}.txt'
        
        if output_file.exists():
            if verbose:
                print(f"File already exists: {output_file}")
            return str(output_file)
        
        if verbose:
            print(f"Downloading HMP {region.upper()} data...")
            print(f"URL: {url}")
        
        try:
            _download_with_ssl_context(url, str(gz_file), verbose=verbose)
            
            if verbose:
                print(f"\nExtracting...")
            
            # Decompress gzip file
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove compressed file
            os.remove(gz_file)
            
            if verbose:
                print(f"Downloaded to: {output_file}")
            
            self.data_dir = output_dir
            return str(output_file)
            
        except Exception as e:
            raise RuntimeError(f"Failed to download HMP data: {e}")
    
    def load_and_preprocess(
        self,
        data_path: Optional[str] = None,
        download_if_missing: bool = True
    ) -> ProcessedMicrobiomeDataset:
        """Load and preprocess HMP data.
        
        Args:
            data_path: Path to OTU table file (downloads if None and download_if_missing=True)
            download_if_missing: Download data if not found locally
            
        Returns:
            ProcessedMicrobiomeDataset ready for training
        """
        if data_path is None:
            if self.data_dir:
                # Look for existing files
                for pattern in ['*.txt', 'otu_table*.txt']:
                    files = list(self.data_dir.glob(pattern))
                    if files:
                        data_path = str(files[0])
                        break
            
            if data_path is None and download_if_missing:
                data_path = self.download()
            elif data_path is None:
                raise ValueError("No data file found. Set download_if_missing=True or provide data_path.")
        
        # Load data
        counts, sample_ids, taxa_names = self.load_from_tsv(data_path)
        
        # Preprocess
        self._processed_data = self.preprocessor.process(
            counts=counts,
            taxa_names=taxa_names,
            sample_ids=sample_ids
        )
        
        return self._processed_data
    
    def load_from_biom(self, biom_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load data from BIOM format file (HMP).
        
        Args:
            biom_path: Path to BIOM file
            
        Returns:
            Tuple of (counts, sample_ids, taxa_names)
        """
        try:
            import biom
            table = biom.load_table(biom_path)
            counts = table.matrix_data.toarray().T  # samples x taxa
            sample_ids = list(table.ids(axis='sample'))
            taxa_names = list(table.ids(axis='observation'))
            return counts, sample_ids, taxa_names
        except ImportError:
            raise ImportError(
                "biom-format package required. Install with: pip install biom-format"
            )
    
    def load_from_tsv(
        self, 
        otu_path: str,
        transpose: bool = True
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load data from TSV OTU table.
        
        Args:
            otu_path: Path to TSV file
            transpose: If True, transpose so rows are samples
            
        Returns:
            Tuple of (counts, sample_ids, taxa_names)
        """
        import csv
        
        with open(otu_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            data = []
            row_ids = []
            for row in reader:
                row_ids.append(row[0])
                data.append([float(x) if x else 0.0 for x in row[1:]])
        
        counts = np.array(data)
        col_ids = header[1:]
        
        if transpose:
            counts = counts.T
            sample_ids = col_ids
            taxa_names = row_ids
        else:
            sample_ids = row_ids
            taxa_names = col_ids
        
        return counts, sample_ids, taxa_names
    
    def generate_synthetic(
        self,
        num_samples: int = 500,
        num_taxa: int = 500,
        sparsity: float = 0.65,
        seed: Optional[int] = None
    ) -> ProcessedMicrobiomeDataset:
        """Generate synthetic data mimicking HMP characteristics.
        
        HMP stool samples tend to be slightly less sparse than AGP
        and have different diversity patterns.
        
        Args:
            num_samples: Number of samples to generate
            num_taxa: Number of taxa
            sparsity: Target sparsity level
            seed: Random seed for reproducibility
            
        Returns:
            ProcessedMicrobiomeDataset with synthetic data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate taxa prevalences (slightly different distribution than AGP)
        prevalences = np.random.power(0.35, num_taxa)
        prevalences = np.sort(prevalences)[::-1]
        
        # Generate compositions
        compositions = np.zeros((num_samples, num_taxa))
        
        for i in range(num_samples):
            # Determine which taxa are present
            presence = np.random.random(num_taxa) < prevalences
            
            # Adjust sparsity
            current_sparsity = 1 - np.mean(presence)
            if current_sparsity < sparsity:
                present_idx = np.where(presence)[0]
                n_to_zero = int((sparsity - current_sparsity) * num_taxa)
                if n_to_zero > 0 and len(present_idx) > n_to_zero:
                    zero_idx = np.random.choice(present_idx, n_to_zero, replace=False)
                    presence[zero_idx] = False
            
            # Generate abundances (log-normal with different parameters)
            abundances = np.zeros(num_taxa)
            present_idx = np.where(presence)[0]
            if len(present_idx) > 0:
                log_abundances = np.random.normal(-2.5, 1.3, len(present_idx))
                abundances[present_idx] = np.exp(log_abundances)
            
            # Normalize
            if abundances.sum() > 0:
                compositions[i] = abundances / abundances.sum()
        
        # Generate sample IDs and taxa names
        sample_ids = [f"HMP_{self.body_site}_{i:05d}" for i in range(num_samples)]
        taxa_names = [f"OTU_{i:05d}" for i in range(num_taxa)]
        
        # Compute statistics
        stats = compute_dataset_stats(compositions)
        
        self._processed_data = ProcessedMicrobiomeDataset(
            compositions=compositions,
            counts=None,
            sample_ids=sample_ids,
            taxa_names=taxa_names,
            stats=stats,
            metadata=None
        )
        
        return self._processed_data
    
    def load(
        self,
        path: Optional[str] = None,
        format: str = 'auto'
    ) -> ProcessedMicrobiomeDataset:
        """Load and preprocess HMP data.
        
        Args:
            path: Path to data file (BIOM or TSV)
            format: 'biom', 'tsv', or 'auto' to detect from extension
            
        Returns:
            ProcessedMicrobiomeDataset
        """
        if path is None:
            if self.data_dir is None:
                warnings.warn(
                    "No data path provided, generating synthetic HMP data"
                )
                return self.generate_synthetic()
            path = str(self.data_dir / f"{self.body_site}_otu_table.biom")
        
        # Detect format
        if format == 'auto':
            if path.endswith('.biom'):
                format = 'biom'
            else:
                format = 'tsv'
        
        # Load raw data
        if format == 'biom':
            counts, sample_ids, taxa_names = self.load_from_biom(path)
        else:
            counts, sample_ids, taxa_names = self.load_from_tsv(path)
        
        # Preprocess
        self._processed_data = self.preprocessor.process(
            counts=counts,
            taxa_names=taxa_names,
            sample_ids=sample_ids
        )
        
        return self._processed_data
    
    @property
    def data(self) -> Optional[ProcessedMicrobiomeDataset]:
        """Get processed data if loaded."""
        return self._processed_data




def load_npz_dataset(npz_path: str):
    """Load a preprocessed dataset from .npz file."""
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    compositions = data['compositions']
    taxa_names = list(data['taxa_names'])
    sample_ids = list(data['sample_ids'])
    stats = MicrobiomeDatasetStats(
        mean_sparsity=float(data['mean_sparsity']),
        std_sparsity=0.0,
        taxon_prevalences=data['taxon_prevalences'],
        alpha_diversity_mean=float(data['alpha_diversity_mean']),
        alpha_diversity_std=0.0,
        beta_diversity_mean=float(data['beta_diversity_mean']),
        beta_diversity_std=0.0,
        num_samples=compositions.shape[0],
        num_taxa=compositions.shape[1]
    )
    return ProcessedMicrobiomeDataset(
        compositions=compositions,
        counts=None,
        sample_ids=sample_ids,
        taxa_names=taxa_names,
        stats=stats,
        metadata=None
    )

def load_dataset(
    name: str,
    data_dir: Optional[str] = None,
    use_real_data: bool = False,
    download_if_missing: bool = True,
    **kwargs
) -> ProcessedMicrobiomeDataset:
    """Convenience function to load a microbiome dataset by name.
    
    Args:
        name: Dataset name ('american_gut', 'hmp', 'agp')
        data_dir: Optional data directory
        use_real_data: If True, download and use real data; if False, use synthetic
        download_if_missing: Download real data if not found locally
        **kwargs: Additional arguments passed to dataset loader
        
    Returns:
        ProcessedMicrobiomeDataset
        
    Example:
        >>> # Use synthetic data (fast, for testing)
        >>> dataset = load_dataset('american_gut')
        >>> 
        >>> # Use real data (downloads ~100MB)
        >>> dataset = load_dataset('american_gut', use_real_data=True)
    """
    name_lower = name.lower()
    
    if name_lower in ('american_gut', 'agp', 'americangut'):
        dataset = AmericanGutDataset(data_dir=data_dir, **kwargs)
        if use_real_data:
            return dataset.load_and_preprocess(download_if_missing=download_if_missing)
        else:
            return dataset.generate_synthetic()
    elif name_lower in ('hmp', 'human_microbiome_project'):
        dataset = HMPDataset(data_dir=data_dir, **kwargs)
        if use_real_data:
            return dataset.load_and_preprocess(download_if_missing=download_if_missing)
        else:
            return dataset.generate_synthetic()
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: american_gut, hmp")


def create_train_val_split(
    dataset: ProcessedMicrobiomeDataset,
    val_fraction: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[ProcessedMicrobiomeDataset, ProcessedMicrobiomeDataset]:
    """Split dataset into training and validation sets.
    
    Args:
        dataset: Dataset to split
        val_fraction: Fraction of samples for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = dataset.compositions.shape[0]
    num_val = int(num_samples * val_fraction)
    
    # Random permutation
    indices = np.random.permutation(num_samples)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    # Split compositions
    train_comp = dataset.compositions[train_indices]
    val_comp = dataset.compositions[val_indices]
    
    # Split counts if available
    train_counts = dataset.counts[train_indices] if dataset.counts is not None else None
    val_counts = dataset.counts[val_indices] if dataset.counts is not None else None
    
    # Split sample IDs
    train_ids = [dataset.sample_ids[i] for i in train_indices]
    val_ids = [dataset.sample_ids[i] for i in val_indices]
    
    # Split metadata if available
    train_meta = None
    val_meta = None
    if dataset.metadata is not None:
        train_meta = {k: v[train_indices] for k, v in dataset.metadata.items()}
        val_meta = {k: v[val_indices] for k, v in dataset.metadata.items()}
    
    # Compute stats for each split
    train_stats = compute_dataset_stats(train_comp)
    val_stats = compute_dataset_stats(val_comp)
    
    train_dataset = ProcessedMicrobiomeDataset(
        compositions=train_comp,
        counts=train_counts,
        sample_ids=train_ids,
        taxa_names=dataset.taxa_names,
        stats=train_stats,
        metadata=train_meta
    )
    
    val_dataset = ProcessedMicrobiomeDataset(
        compositions=val_comp,
        counts=val_counts,
        sample_ids=val_ids,
        taxa_names=dataset.taxa_names,
        stats=val_stats,
        metadata=val_meta
    )
    
    return train_dataset, val_dataset
