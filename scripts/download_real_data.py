#!/usr/bin/env python3
"""Download and prepare real microbiome data for publication-ready experiments.

This script integrates with the American-Gut repository to access real
microbiome datasets and prepares them for use in the simulation system.

Usage:
    python scripts/download_real_data.py --dataset american_gut --output data/
    python scripts/download_real_data.py --dataset hmp --output data/
    python scripts/download_real_data.py --all
"""

import argparse
import sys
import shutil
from pathlib import Path
import json
import urllib.request
import ftplib
from typing import Optional, Dict, List
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RealDataDownloader:
    """Downloads and prepares real microbiome datasets."""
    
    def __init__(self, output_dir: str = "data", american_gut_repo: str = "American-Gut"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.ag_repo = Path(american_gut_repo)
        if not self.ag_repo.exists():
            raise FileNotFoundError(
                f"American-Gut repository not found at {american_gut_repo}. "
                "Please clone it first: git clone https://github.com/biocore/American-Gut.git"
            )
        
        self.ag_data_dir = self.ag_repo / "data" / "AG"
        self.hmp_data_dir = self.ag_repo / "data" / "HMP"
        
        print(f"Output directory: {self.output_dir}")
        print(f"American-Gut repo: {self.ag_repo}")
    
    def download_american_gut_local(self, rarefaction: str = "even10k") -> Dict[str, Path]:
        """Copy American Gut data from local repository.
        
        Args:
            rarefaction: Which rarefaction level to use:
                - "full" - Full unrarefied data (AG_100nt.biom)
                - "even1k" - Rarefied to 1000 sequences per sample
                - "even10k" - Rarefied to 10000 sequences per sample (recommended)
        
        Returns:
            Dictionary with paths to copied files
        """
        print("\n" + "="*60)
        print("DOWNLOADING AMERICAN GUT DATA (LOCAL COPY)")
        print("="*60)
        
        # Determine source file
        if rarefaction == "full":
            source_biom = self.ag_data_dir / "AG_100nt.biom"
            source_txt = self.ag_data_dir / "AG_100nt.txt"
        elif rarefaction == "even1k":
            source_biom = self.ag_data_dir / "AG_100nt_even1k.biom"
            source_txt = self.ag_data_dir / "AG_100nt_even1k.txt"
        elif rarefaction == "even10k":
            source_biom = self.ag_data_dir / "AG_100nt_even10k.biom"
            source_txt = self.ag_data_dir / "AG_100nt_even10k.txt"
        else:
            raise ValueError(f"Unknown rarefaction level: {rarefaction}")
        
        if not source_biom.exists():
            raise FileNotFoundError(
                f"American Gut data file not found: {source_biom}\n"
                f"Please ensure the American-Gut repository is properly cloned."
            )
        
        # Create output directory
        ag_output = self.output_dir / "american_gut"
        ag_output.mkdir(exist_ok=True, parents=True)
        
        # Copy files
        dest_biom = ag_output / "ag_otu_table.biom"
        dest_txt = ag_output / "ag_otu_table.txt"
        
        print(f"Copying {source_biom.name}...")
        shutil.copy2(source_biom, dest_biom)
        print(f"  → {dest_biom}")
        
        if source_txt.exists():
            print(f"Copying {source_txt.name}...")
            shutil.copy2(source_txt, dest_txt)
            print(f"  → {dest_txt}")
        
        # Copy metadata files if available
        metadata_files = [
            "accession_to_sample.json",
            "sample_to_accession.json"
        ]
        
        for metadata_file in metadata_files:
            source_meta = self.ag_data_dir / metadata_file
            if source_meta.exists():
                dest_meta = ag_output / metadata_file
                shutil.copy2(source_meta, dest_meta)
                print(f"Copied metadata: {metadata_file}")
        
        # Get file sizes
        biom_size_mb = dest_biom.stat().st_size / (1024 * 1024)
        
        print(f"\n[OK] American Gut data copied successfully!")
        print(f"  BIOM file size: {biom_size_mb:.1f} MB")
        print(f"  Rarefaction level: {rarefaction}")
        
        return {
            "biom": dest_biom,
            "txt": dest_txt if source_txt.exists() else None,
            "output_dir": ag_output
        }
    
    def download_american_gut_latest(self) -> Dict[str, Path]:
        """Download latest American Gut data from FTP server.
        
        Note: This downloads the most recent data, which may be larger
        than the repository version.
        """
        print("\n" + "="*60)
        print("DOWNLOADING LATEST AMERICAN GUT DATA (FTP)")
        print("="*60)
        print("Note: This may take several minutes for large files...")
        
        ag_output = self.output_dir / "american_gut"
        ag_output.mkdir(exist_ok=True, parents=True)
        
        ftp_host = "ftp.microbio.me"
        ftp_path = "/AmericanGut/latest/"
        
        try:
            print(f"Connecting to {ftp_host}...")
            ftp = ftplib.FTP(ftp_host)
            ftp.login()
            ftp.cwd(ftp_path)
            
            # List available files
            files = ftp.nlst()
            print(f"Found {len(files)} files on FTP server")
            
            # Download BIOM table (look for the main OTU table)
            biom_files = [f for f in files if f.endswith('.biom') and 'otu' in f.lower()]
            
            if not biom_files:
                print("Warning: No BIOM files found on FTP server")
                print("Falling back to local repository data...")
                ftp.quit()
                return self.download_american_gut_local()
            
            # Download the first BIOM file found
            biom_file = biom_files[0]
            dest_biom = ag_output / "ag_otu_table.biom"
            
            print(f"Downloading {biom_file}...")
            with open(dest_biom, 'wb') as f:
                ftp.retrbinary(f'RETR {biom_file}', f.write)
            
            biom_size_mb = dest_biom.stat().st_size / (1024 * 1024)
            print(f"  → {dest_biom} ({biom_size_mb:.1f} MB)")
            
            ftp.quit()
            
            print(f"\n[OK] Latest American Gut data downloaded successfully!")
            
            return {
                "biom": dest_biom,
                "output_dir": ag_output
            }
            
        except Exception as e:
            print(f"FTP download failed: {e}")
            print("Falling back to local repository data...")
            return self.download_american_gut_local()
    
    def download_hmp_local(self) -> Dict[str, Path]:
        """Copy HMP data from local American-Gut repository.
        
        Returns:
            Dictionary with paths to copied files
        """
        print("\n" + "="*60)
        print("DOWNLOADING HMP DATA (LOCAL COPY)")
        print("="*60)
        
        if not self.hmp_data_dir.exists():
            raise FileNotFoundError(
                f"HMP data directory not found: {self.hmp_data_dir}\n"
                f"The American-Gut repository may not include HMP data."
            )
        
        # Create output directory
        hmp_output = self.output_dir / "hmp"
        hmp_output.mkdir(exist_ok=True, parents=True)
        
        # Look for HMP BIOM files
        hmp_biom_files = list(self.hmp_data_dir.glob("*.biom"))
        
        if not hmp_biom_files:
            print("Warning: No HMP BIOM files found in American-Gut repository")
            print("HMP data may need to be downloaded separately")
            return {"output_dir": hmp_output, "files": []}
        
        copied_files = []
        for source_biom in hmp_biom_files:
            dest_biom = hmp_output / source_biom.name
            print(f"Copying {source_biom.name}...")
            shutil.copy2(source_biom, dest_biom)
            print(f"  → {dest_biom}")
            copied_files.append(dest_biom)
        
        print(f"\n[OK] HMP data copied successfully!")
        print(f"  Files copied: {len(copied_files)}")
        
        return {
            "output_dir": hmp_output,
            "files": copied_files
        }
    
    def create_data_manifest(self, datasets: Dict[str, Dict]) -> Path:
        """Create a manifest file documenting downloaded datasets.
        
        Args:
            datasets: Dictionary of dataset information
        
        Returns:
            Path to manifest file
        """
        manifest_path = self.output_dir / "data_manifest.json"
        
        manifest = {
            "version": "1.0",
            "datasets": datasets,
            "source": "American-Gut repository + FTP downloads",
            "notes": "Real microbiome data for publication-ready experiments"
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        print(f"\n[OK] Data manifest created: {manifest_path}")
        
        return manifest_path
    
    def download_all(self, use_latest: bool = False) -> Dict[str, Dict]:
        """Download all available datasets.
        
        Args:
            use_latest: If True, download latest from FTP; otherwise use local repo
        
        Returns:
            Dictionary with information about downloaded datasets
        """
        print("\n" + "="*60)
        print("DOWNLOADING ALL REAL MICROBIOME DATA")
        print("="*60)
        
        datasets = {}
        
        # American Gut
        try:
            if use_latest:
                ag_result = self.download_american_gut_latest()
            else:
                ag_result = self.download_american_gut_local(rarefaction="even10k")
            
            datasets["american_gut"] = {
                "status": "success",
                "files": ag_result,
                "description": "American Gut Project - 100nt reads, rarefied to 10k"
            }
        except Exception as e:
            print(f"Error downloading American Gut data: {e}")
            datasets["american_gut"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # HMP
        try:
            hmp_result = self.download_hmp_local()
            datasets["hmp"] = {
                "status": "success",
                "files": hmp_result,
                "description": "Human Microbiome Project data"
            }
        except Exception as e:
            print(f"Error downloading HMP data: {e}")
            datasets["hmp"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Create manifest
        self.create_data_manifest(datasets)
        
        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        
        for dataset_name, info in datasets.items():
            status_symbol = "[OK]" if info["status"] == "success" else "[FAIL]"
            print(f"{status_symbol} {dataset_name}: {info['status']}")
        
        print("\nData is ready for use with:")
        print("  from src.microbiome_datasets import load_dataset")
        print("  dataset = load_dataset('american_gut', use_real_data=True)")
        
        return datasets


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download real microbiome data for publication experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download American Gut data from local repository
    python scripts/download_real_data.py --dataset american_gut
    
    # Download latest American Gut data from FTP
    python scripts/download_real_data.py --dataset american_gut --latest
    
    # Download all available datasets
    python scripts/download_real_data.py --all
    
    # Specify custom output directory
    python scripts/download_real_data.py --all --output ./my_data
        """
    )
    
    parser.add_argument(
        '--dataset',
        choices=['american_gut', 'hmp'],
        help='Specific dataset to download'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available datasets'
    )
    
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Download latest data from FTP (American Gut only)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for downloaded data (default: data/)'
    )
    
    parser.add_argument(
        '--american-gut-repo',
        type=str,
        default='American-Gut',
        help='Path to American-Gut repository (default: American-Gut/)'
    )
    
    parser.add_argument(
        '--rarefaction',
        choices=['full', 'even1k', 'even10k'],
        default='even10k',
        help='Rarefaction level for American Gut data (default: even10k)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.dataset:
        parser.error("Must specify either --dataset or --all")
    
    try:
        downloader = RealDataDownloader(
            output_dir=args.output,
            american_gut_repo=args.american_gut_repo
        )
        
        if args.all:
            downloader.download_all(use_latest=args.latest)
        elif args.dataset == 'american_gut':
            if args.latest:
                downloader.download_american_gut_latest()
            else:
                downloader.download_american_gut_local(rarefaction=args.rarefaction)
        elif args.dataset == 'hmp':
            downloader.download_hmp_local()
        
        print("\n" + "="*60)
        print("[OK] DATA DOWNLOAD COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext steps:")
        print("1. Verify data with: python scripts/verify_data.py")
        print("2. Train models with: python scripts/retrain_optimal.py --real-data")
        print("3. Generate publication outputs with real data")
        
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
