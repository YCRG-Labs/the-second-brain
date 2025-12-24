"""Data management system for downloading, caching, and preprocessing datasets.

This module provides infrastructure for managing microbiome datasets including:
- Automatic download from public repositories
- Checksum verification
- Local caching
- Dataset registry
- Train/val/test splitting
"""

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .types import ProcessedDataset, PhylogeneticTree
from .exceptions import MicrobiomeSimulationError


class DataDownloadError(MicrobiomeSimulationError):
    """Raised when dataset download fails."""
    pass


class DataIntegrityError(MicrobiomeSimulationError):
    """Raised when checksum verification fails."""
    pass


class PreprocessingError(MicrobiomeSimulationError):
    """Raised when data preprocessing fails."""
    pass


@dataclass
class DatasetMetadata:
    """Metadata for a registered dataset.
    
    Attributes:
        name: Dataset identifier
        description: Human-readable description
        url: Download URL
        checksum: SHA256 checksum for verification
        num_samples: Number of samples in dataset
        num_taxa: Number of taxa/OTUs
        data_type: Type of data ('otu_table', 'fastq', 'biom')
        citation: Citation information
    """
    name: str
    description: str
    url: str
    checksum: str
    num_samples: int
    num_taxa: int
    data_type: str
    citation: str


class DatasetRegistry:
    """Registry of available datasets with metadata."""
    
    # Registry of known datasets
    DATASETS: Dict[str, DatasetMetadata] = {
        'american_gut': DatasetMetadata(
            name='american_gut',
            description='American Gut Project - 16S rRNA sequencing data',
            url='https://qiita.ucsd.edu/public_artifact_download/?artifact_id=67716',
            checksum='',  # To be filled with actual checksum
            num_samples=10000,
            num_taxa=5000,
            data_type='biom',
            citation='McDonald et al. 2018, mSystems'
        ),
        'hmp': DatasetMetadata(
            name='hmp',
            description='Human Microbiome Project - 16S rRNA sequencing data',
            url='https://portal.hmpdacc.org/files/',
            checksum='',  # To be filled with actual checksum
            num_samples=5000,
            num_taxa=4000,
            data_type='biom',
            citation='Human Microbiome Project Consortium 2012, Nature'
        ),
        'moving_pictures': DatasetMetadata(
            name='moving_pictures',
            description='Moving Pictures of the Human Microbiome - longitudinal study',
            url='https://qiita.ucsd.edu/public_artifact_download/?artifact_id=1629',
            checksum='',  # To be filled with actual checksum
            num_samples=1967,
            num_taxa=3118,
            data_type='biom',
            citation='Caporaso et al. 2011, Genome Biology'
        )
    }
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return list(cls.DATASETS.keys())
    
    @classmethod
    def get_metadata(cls, dataset_name: str) -> DatasetMetadata:
        """Get metadata for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset metadata
            
        Raises:
            ValueError: If dataset not found in registry
        """
        if dataset_name not in cls.DATASETS:
            available = ', '.join(cls.list_datasets())
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {available}"
            )
        return cls.DATASETS[dataset_name]
    
    @classmethod
    def register_dataset(cls, metadata: DatasetMetadata) -> None:
        """Register a new dataset.
        
        Args:
            metadata: Dataset metadata to register
        """
        cls.DATASETS[metadata.name] = metadata


class AmericanGutDownloader:
    """Specialized downloader for American Gut Project data.
    
    The American Gut Project is a citizen science project that collected
    microbiome samples from thousands of participants. Data is available
    through Qiita (https://qiita.ucsd.edu/) and EBI.
    """
    
    # Known artifact IDs for different data types
    ARTIFACTS = {
        'otu_table': {
            'id': '67716',
            'url': 'https://qiita.ucsd.edu/public_artifact_download/?artifact_id=67716',
            'checksum': '',  # To be filled with actual checksum
            'description': 'Closed-reference OTU table at 97% similarity'
        },
        'metadata': {
            'id': '10317',
            'url': 'https://qiita.ucsd.edu/public_download/?data=sample_information&study_id=10317',
            'checksum': '',
            'description': 'Sample metadata including host characteristics'
        }
    }
    
    @staticmethod
    def download_otu_table(output_dir: Path, verify_checksum: bool = True) -> Path:
        """Download American Gut OTU table.
        
        Args:
            output_dir: Directory to save downloaded file
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Path to downloaded OTU table
            
        Raises:
            DataDownloadError: If download fails
            DataIntegrityError: If checksum verification fails
        """
        artifact = AmericanGutDownloader.ARTIFACTS['otu_table']
        output_path = output_dir / "otu_table.biom"
        
        print(f"Downloading American Gut OTU table (artifact {artifact['id']})...")
        print(f"This may take several minutes...")
        
        try:
            # Download with progress reporting
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(
                artifact['url'],
                output_path,
                reporthook=report_progress
            )
            print()  # New line after progress
            
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise DataDownloadError(
                f"Failed to download American Gut OTU table: {str(e)}"
            ) from e
        
        # Verify checksum if provided and requested
        if verify_checksum and artifact['checksum']:
            print("Verifying checksum...")
            sha256 = hashlib.sha256()
            with open(output_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            
            actual_checksum = sha256.hexdigest()
            if actual_checksum != artifact['checksum']:
                output_path.unlink()
                raise DataIntegrityError(
                    "Checksum verification failed for American Gut OTU table"
                )
        
        print(f"Successfully downloaded to {output_path}")
        return output_path
    
    @staticmethod
    def download_metadata(output_dir: Path, verify_checksum: bool = True) -> Path:
        """Download American Gut metadata.
        
        Args:
            output_dir: Directory to save downloaded file
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Path to downloaded metadata file
            
        Raises:
            DataDownloadError: If download fails
        """
        artifact = AmericanGutDownloader.ARTIFACTS['metadata']
        output_path = output_dir / "metadata.txt"
        
        print(f"Downloading American Gut metadata...")
        
        try:
            urllib.request.urlretrieve(artifact['url'], output_path)
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise DataDownloadError(
                f"Failed to download American Gut metadata: {str(e)}"
            ) from e
        
        print(f"Successfully downloaded to {output_path}")
        return output_path
    
    @staticmethod
    def download_all(output_dir: Path, verify_checksum: bool = True) -> Dict[str, Path]:
        """Download all American Gut Project files.
        
        Args:
            output_dir: Directory to save downloaded files
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Dictionary mapping file type to path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'otu_table': AmericanGutDownloader.download_otu_table(
                output_dir, verify_checksum
            ),
            'metadata': AmericanGutDownloader.download_metadata(
                output_dir, verify_checksum
            )
        }


class HMPDownloader:
    """Specialized downloader for Human Microbiome Project data.
    
    The Human Microbiome Project (HMP) characterized the microbiome of
    healthy adults across multiple body sites. Data is available through
    the HMP Data Analysis and Coordination Center (DACC).
    """
    
    # Known data files from HMP DACC
    DATA_FILES = {
        'otu_table': {
            'url': 'https://portal.hmpdacc.org/files/HMQCP.tar.bz2',
            'checksum': '',  # To be filled with actual checksum
            'description': '16S rRNA OTU tables from multiple body sites'
        },
        'metadata': {
            'url': 'https://portal.hmpdacc.org/files/ppAll_V35_map.txt',
            'checksum': '',
            'description': 'Sample metadata including body site and subject info'
        },
        'taxonomy': {
            'url': 'https://portal.hmpdacc.org/files/gg_13_5_taxonomy.txt.gz',
            'checksum': '',
            'description': 'Greengenes taxonomy mapping'
        }
    }
    
    @staticmethod
    def download_otu_table(output_dir: Path, verify_checksum: bool = True) -> Path:
        """Download HMP OTU table.
        
        Args:
            output_dir: Directory to save downloaded file
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Path to downloaded OTU table archive
            
        Raises:
            DataDownloadError: If download fails
            DataIntegrityError: If checksum verification fails
        """
        data_file = HMPDownloader.DATA_FILES['otu_table']
        output_path = output_dir / "otu_table.tar.bz2"
        
        print(f"Downloading HMP OTU table...")
        print(f"This may take several minutes...")
        
        try:
            # Download with progress reporting
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(
                data_file['url'],
                output_path,
                reporthook=report_progress
            )
            print()  # New line after progress
            
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise DataDownloadError(
                f"Failed to download HMP OTU table: {str(e)}"
            ) from e
        
        # Verify checksum if provided and requested
        if verify_checksum and data_file['checksum']:
            print("Verifying checksum...")
            sha256 = hashlib.sha256()
            with open(output_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            
            actual_checksum = sha256.hexdigest()
            if actual_checksum != data_file['checksum']:
                output_path.unlink()
                raise DataIntegrityError(
                    "Checksum verification failed for HMP OTU table"
                )
        
        print(f"Successfully downloaded to {output_path}")
        return output_path
    
    @staticmethod
    def download_metadata(output_dir: Path, verify_checksum: bool = True) -> Path:
        """Download HMP metadata.
        
        Args:
            output_dir: Directory to save downloaded file
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Path to downloaded metadata file
            
        Raises:
            DataDownloadError: If download fails
        """
        data_file = HMPDownloader.DATA_FILES['metadata']
        output_path = output_dir / "metadata.txt"
        
        print(f"Downloading HMP metadata...")
        
        try:
            urllib.request.urlretrieve(data_file['url'], output_path)
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise DataDownloadError(
                f"Failed to download HMP metadata: {str(e)}"
            ) from e
        
        print(f"Successfully downloaded to {output_path}")
        return output_path
    
    @staticmethod
    def download_taxonomy(output_dir: Path, verify_checksum: bool = True) -> Path:
        """Download HMP taxonomy mapping.
        
        Args:
            output_dir: Directory to save downloaded file
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Path to downloaded taxonomy file
            
        Raises:
            DataDownloadError: If download fails
        """
        data_file = HMPDownloader.DATA_FILES['taxonomy']
        output_path = output_dir / "taxonomy.txt.gz"
        
        print(f"Downloading HMP taxonomy...")
        
        try:
            urllib.request.urlretrieve(data_file['url'], output_path)
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise DataDownloadError(
                f"Failed to download HMP taxonomy: {str(e)}"
            ) from e
        
        print(f"Successfully downloaded to {output_path}")
        return output_path
    
    @staticmethod
    def download_all(output_dir: Path, verify_checksum: bool = True) -> Dict[str, Path]:
        """Download all HMP files.
        
        Args:
            output_dir: Directory to save downloaded files
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Dictionary mapping file type to path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'otu_table': HMPDownloader.download_otu_table(
                output_dir, verify_checksum
            ),
            'metadata': HMPDownloader.download_metadata(
                output_dir, verify_checksum
            ),
            'taxonomy': HMPDownloader.download_taxonomy(
                output_dir, verify_checksum
            )
        }


class DataManager:
    """Manages dataset download, preprocessing, and caching.
    
    This class provides a unified interface for:
    - Downloading datasets from public repositories
    - Verifying data integrity with checksums
    - Caching downloaded and preprocessed data
    - Creating reproducible train/val/test splits
    
    Example:
        >>> manager = DataManager(cache_dir="~/.microbiome_data")
        >>> dataset = manager.get_dataset('american_gut', split='train')
        >>> print(dataset.compositions.shape)
        (7000, 5000)
    """
    
    def __init__(self, cache_dir: str = "~/.microbiome_data"):
        """Initialize data manager with cache directory.
        
        Args:
            cache_dir: Directory for caching downloaded and processed data
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.raw_dir = self.cache_dir / "raw"
        self.processed_dir = self.cache_dir / "processed"
        self.splits_dir = self.cache_dir / "splits"
        
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.splits_dir.mkdir(exist_ok=True)
        
        # Load or create registry index
        self.registry_file = self.cache_dir / "registry.json"
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load local registry of downloaded datasets."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.local_registry = json.load(f)
        else:
            self.local_registry = {}
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save local registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.local_registry, f, indent=2)
    
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal checksum string
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum matches expected value.
        
        Args:
            file_path: Path to file
            expected_checksum: Expected SHA256 checksum
            
        Returns:
            True if checksum matches, False otherwise
        """
        if not expected_checksum:
            # Skip verification if no checksum provided
            return True
        
        actual_checksum = self._compute_checksum(file_path)
        return actual_checksum == expected_checksum
    
    def download_dataset(
        self, 
        dataset_name: str,
        force: bool = False
    ) -> Path:
        """Download dataset from public repository.
        
        Downloads the dataset if not already cached, verifies checksum,
        and returns path to downloaded data.
        
        Args:
            dataset_name: Name of dataset (e.g., 'american_gut', 'hmp')
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded data directory
            
        Raises:
            DataDownloadError: If download fails
            DataIntegrityError: If checksum verification fails
        """
        metadata = DatasetRegistry.get_metadata(dataset_name)
        dataset_dir = self.raw_dir / dataset_name
        
        # Check if already downloaded
        if dataset_dir.exists() and not force:
            if dataset_name in self.local_registry:
                print(f"Dataset '{dataset_name}' already cached at {dataset_dir}")
                return dataset_dir
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Use specialized downloader if available
        if dataset_name == 'american_gut':
            try:
                files = AmericanGutDownloader.download_all(
                    dataset_dir,
                    verify_checksum=bool(metadata.checksum)
                )
            except Exception as e:
                raise DataDownloadError(
                    f"Failed to download {dataset_name}: {str(e)}"
                ) from e
        elif dataset_name == 'hmp':
            try:
                files = HMPDownloader.download_all(
                    dataset_dir,
                    verify_checksum=bool(metadata.checksum)
                )
            except Exception as e:
                raise DataDownloadError(
                    f"Failed to download {dataset_name}: {str(e)}"
                ) from e
        else:
            # Generic download for other datasets
            print(f"Downloading {dataset_name} from {metadata.url}...")
            download_path = dataset_dir / "data.raw"
            
            try:
                urllib.request.urlretrieve(metadata.url, download_path)
            except Exception as e:
                raise DataDownloadError(
                    f"Failed to download {dataset_name}: {str(e)}"
                ) from e
            
            # Verify checksum if provided
            if metadata.checksum:
                print("Verifying checksum...")
                if not self._verify_checksum(download_path, metadata.checksum):
                    download_path.unlink()  # Remove corrupted file
                    raise DataIntegrityError(
                        f"Checksum verification failed for {dataset_name}. "
                        "File may be corrupted."
                    )
        
        # Update local registry
        self.local_registry[dataset_name] = {
            'downloaded_at': str(Path.cwd()),
            'metadata': asdict(metadata)
        }
        self._save_registry()
        
        print(f"Successfully downloaded {dataset_name}")
        return dataset_dir
    
    def preprocess_dataset(
        self,
        raw_data_path: Path,
        output_path: Path,
        min_reads: int = 10000,
        rarefaction_depth: int = 10000,
        min_prevalence: float = 0.05
    ) -> ProcessedDataset:
        """Run complete preprocessing pipeline.
        
        Processes raw microbiome data through the full pipeline:
        1. Load raw data (FASTQ, BIOM, or OTU table)
        2. Quality control and filtering
        3. Rarefaction to fixed depth
        4. Taxa filtering by prevalence
        5. Taxonomy assignment (if needed)
        
        Args:
            raw_data_path: Path to raw data directory
            output_path: Path to save processed data
            min_reads: Minimum reads per sample
            rarefaction_depth: Rarefaction depth for normalization
            min_prevalence: Minimum prevalence for taxa filtering
            
        Returns:
            Processed dataset
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        from .preprocessing import PreprocessingPipeline
        
        try:
            # Initialize preprocessing pipeline
            pipeline = PreprocessingPipeline(
                min_reads=min_reads,
                rarefaction_depth=rarefaction_depth,
                min_prevalence=min_prevalence
            )
            
            # Load raw data
            # This is a simplified version - in practice would handle BIOM, FASTQ, etc.
            samples, counts, tree = self._load_raw_data(raw_data_path)
            
            # Run preprocessing
            processed = pipeline.process(samples, counts, tree)
            
            # Save processed data
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                output_path,
                compositions=processed.compositions,
                metadata=processed.metadata,
                tree=processed.tree,
                sample_ids=processed.sample_ids,
                taxa_mask=processed.taxa_mask
            )
            
            print(f"Preprocessed data saved to {output_path}")
            return processed
            
        except Exception as e:
            raise PreprocessingError(
                f"Preprocessing failed: {str(e)}"
            ) from e
    
    def _load_raw_data(
        self, 
        raw_data_path: Path
    ) -> Tuple[List, np.ndarray, PhylogeneticTree]:
        """Load raw data from various formats.
        
        This is a placeholder for loading different data formats.
        In practice, would handle BIOM, FASTQ, OTU tables, etc.
        
        Args:
            raw_data_path: Path to raw data directory
            
        Returns:
            Tuple of (samples, counts, tree)
            
        Raises:
            PreprocessingError: If data loading fails
        """
        # This is a simplified placeholder
        # Real implementation would:
        # 1. Detect file format (BIOM, FASTQ, etc.)
        # 2. Parse appropriately
        # 3. Load or construct phylogenetic tree
        # 4. Create MicrobiomeSample objects
        
        raise NotImplementedError(
            "Raw data loading not yet implemented. "
            "This requires format-specific parsers for BIOM, FASTQ, etc."
        )
    
    def create_splits(
        self,
        dataset: ProcessedDataset,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        stratify_by: Optional[str] = None,
        save_to: Optional[Path] = None
    ) -> Dict[str, ProcessedDataset]:
        """Create train/val/test splits with stratification.
        
        Creates reproducible train/validation/test splits with optional
        stratification by metadata fields. Saves split indices for
        reproducibility.
        
        Args:
            dataset: Dataset to split
            train_frac: Fraction for training set
            val_frac: Fraction for validation set
            test_frac: Fraction for test set
            seed: Random seed for reproducibility
            stratify_by: Metadata field index to stratify by (not yet implemented)
            save_to: Optional path to save split indices
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
            
        Raises:
            ValueError: If fractions don't sum to 1.0
        """
        # Validate fractions
        if not np.isclose(train_frac + val_frac + test_frac, 1.0):
            raise ValueError(
                f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
            )
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        num_samples = len(dataset.sample_ids)
        
        # Create shuffled indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(num_samples * train_frac)
        val_end = train_end + int(num_samples * val_frac)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create split datasets
        splits = {
            'train': self._create_subset(dataset, train_indices),
            'val': self._create_subset(dataset, val_indices),
            'test': self._create_subset(dataset, test_indices)
        }
        
        # Save split indices if requested
        if save_to:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_to,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac
            )
            print(f"Split indices saved to {save_to}")
        
        print(f"Created splits: train={len(train_indices)}, "
              f"val={len(val_indices)}, test={len(test_indices)}")
        
        return splits
    
    def _create_subset(
        self, 
        dataset: ProcessedDataset, 
        indices: np.ndarray
    ) -> ProcessedDataset:
        """Create a subset of a dataset.
        
        Args:
            dataset: Original dataset
            indices: Indices to include in subset
            
        Returns:
            New ProcessedDataset with selected samples
        """
        return ProcessedDataset(
            compositions=dataset.compositions[indices],
            metadata=dataset.metadata[indices],
            tree=dataset.tree,  # Tree is shared across splits
            sample_ids=[dataset.sample_ids[i] for i in indices],
            taxa_mask=dataset.taxa_mask  # Mask is shared across splits
        )
    
    def load_splits(
        self,
        dataset: ProcessedDataset,
        split_file: Path
    ) -> Dict[str, ProcessedDataset]:
        """Load previously saved splits.
        
        Args:
            dataset: Full dataset to split
            split_file: Path to saved split indices
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
            
        Raises:
            FileNotFoundError: If split file doesn't exist
        """
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Load split indices
        split_data = np.load(split_file)
        
        train_indices = split_data['train_indices']
        val_indices = split_data['val_indices']
        test_indices = split_data['test_indices']
        
        # Create split datasets
        splits = {
            'train': self._create_subset(dataset, train_indices),
            'val': self._create_subset(dataset, val_indices),
            'test': self._create_subset(dataset, test_indices)
        }
        
        print(f"Loaded splits from {split_file}")
        print(f"  train={len(train_indices)}, "
              f"val={len(val_indices)}, test={len(test_indices)}")
        
        return splits
    
    def get_dataset(
        self, 
        dataset_name: str,
        split: str = 'train',
        download_if_missing: bool = True
    ) -> ProcessedDataset:
        """Get preprocessed dataset split.
        
        Downloads and preprocesses if needed, then returns the requested split.
        
        Args:
            dataset_name: Name of dataset
            split: One of 'train', 'val', 'test', or 'all'
            download_if_missing: Download if not cached
            
        Returns:
            Processed dataset for the requested split
            
        Raises:
            ValueError: If split is invalid
            FileNotFoundError: If dataset not found and download disabled
        """
        if split not in ['train', 'val', 'test', 'all']:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'val', 'test', or 'all'"
            )
        
        # Check if processed data exists
        processed_path = self.processed_dir / dataset_name / f"{split}.npz"
        
        if not processed_path.exists():
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Dataset '{dataset_name}' split '{split}' not found. "
                    "Set download_if_missing=True to download."
                )
            
            # Download raw data
            raw_path = self.download_dataset(dataset_name)
            
            # Preprocess (will be implemented in 3.4)
            # For now, raise NotImplementedError
            raise NotImplementedError(
                "Automatic preprocessing not yet implemented. "
                "Will be available in subtask 3.4"
            )
        
        # Load processed data
        data = np.load(processed_path, allow_pickle=True)
        
        # Reconstruct ProcessedDataset
        # This is a simplified version - full implementation in 3.4
        return ProcessedDataset(
            compositions=data['compositions'],
            metadata=data['metadata'],
            tree=data['tree'].item(),
            sample_ids=data['sample_ids'].tolist(),
            taxa_mask=data['taxa_mask']
        )
    
    def list_cached_datasets(self) -> List[str]:
        """List all datasets currently in cache.
        
        Returns:
            List of cached dataset names
        """
        return list(self.local_registry.keys())
    
    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear cached data.
        
        Args:
            dataset_name: Specific dataset to clear, or None to clear all
        """
        if dataset_name:
            # Clear specific dataset
            dataset_dir = self.raw_dir / dataset_name
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            
            processed_dir = self.processed_dir / dataset_name
            if processed_dir.exists():
                shutil.rmtree(processed_dir)
            
            if dataset_name in self.local_registry:
                del self.local_registry[dataset_name]
                self._save_registry()
            
            print(f"Cleared cache for {dataset_name}")
        else:
            # Clear all
            shutil.rmtree(self.raw_dir)
            shutil.rmtree(self.processed_dir)
            shutil.rmtree(self.splits_dir)
            
            self.raw_dir.mkdir(exist_ok=True)
            self.processed_dir.mkdir(exist_ok=True)
            self.splits_dir.mkdir(exist_ok=True)
            
            self.local_registry = {}
            self._save_registry()
            
            print("Cleared all cached data")
