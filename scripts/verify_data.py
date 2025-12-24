#!/usr/bin/env python3
"""Verify downloaded real microbiome data integrity and format.

This script validates that downloaded real data is properly formatted
and compatible with the microbiome simulation system.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --dataset american_gut
    python scripts/verify_data.py --verbose
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    from src.microbiome_datasets import load_dataset, AmericanGutDataset, HMPDataset
    from src.evaluation import shannon_entropy, alpha_diversity, beta_diversity
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


class DataVerifier:
    """Verifies integrity and format of downloaded microbiome data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.results = {
            "verification_status": "unknown",
            "datasets": {},
            "summary": {}
        }
    
    def check_file_exists(self, file_path: Path) -> bool:
        """Check if a file exists and is readable."""
        if not file_path.exists():
            return False
        
        try:
            # Try to read first few bytes
            with open(file_path, 'rb') as f:
                f.read(1024)
            return True
        except Exception:
            return False
    
    def verify_american_gut(self, verbose: bool = False) -> Dict:
        """Verify American Gut dataset."""
        print("\n" + "="*50)
        print("VERIFYING AMERICAN GUT DATA")
        print("="*50)
        
        ag_dir = self.data_dir / "american_gut"
        result = {
            "status": "unknown",
            "files_found": {},
            "data_stats": {},
            "issues": []
        }
        
        # Check for required files
        required_files = {
            "biom": ag_dir / "ag_otu_table.biom",
            "txt": ag_dir / "ag_otu_table.txt"
        }
        
        optional_files = {
            "accession_metadata": ag_dir / "accession_to_sample.json",
            "sample_metadata": ag_dir / "sample_to_accession.json"
        }
        
        # Check file existence
        for file_type, file_path in required_files.items():
            exists = self.check_file_exists(file_path)
            result["files_found"][file_type] = {
                "path": str(file_path),
                "exists": exists,
                "size_mb": file_path.stat().st_size / (1024*1024) if exists else 0
            }
            
            if exists:
                print(f"[OK] Found {file_type}: {file_path.name} ({result['files_found'][file_type]['size_mb']:.1f} MB)")
            else:
                print(f"[FAIL] Missing {file_type}: {file_path}")
                result["issues"].append(f"Missing required file: {file_path}")
        
        for file_type, file_path in optional_files.items():
            exists = self.check_file_exists(file_path)
            result["files_found"][file_type] = {
                "path": str(file_path),
                "exists": exists,
                "size_mb": file_path.stat().st_size / (1024*1024) if exists else 0
            }
            
            if exists:
                print(f"[OK] Found {file_type}: {file_path.name}")
            else:
                print(f"- Optional file not found: {file_path.name}")
        
        # Try to load data if dependencies available
        if DEPENDENCIES_AVAILABLE and result["files_found"]["biom"]["exists"]:
            try:
                print("\nLoading data for validation...")
                
                # Load using our dataset class
                dataset = AmericanGutDataset(data_dir=str(ag_dir))
                processed = dataset.load_and_preprocess()
                
                # Get basic statistics
                compositions = processed.compositions
                result["data_stats"] = {
                    "num_samples": compositions.shape[0],
                    "num_taxa": compositions.shape[1],
                    "sparsity": float(np.mean(compositions == 0)),
                    "alpha_diversity_mean": float(np.mean(alpha_diversity(compositions))),
                    "alpha_diversity_std": float(np.std(alpha_diversity(compositions))),
                    "min_abundance": float(np.min(compositions[compositions > 0])),
                    "max_abundance": float(np.max(compositions))
                }
                
                print(f"  Samples: {result['data_stats']['num_samples']:,}")
                print(f"  Taxa: {result['data_stats']['num_taxa']:,}")
                print(f"  Sparsity: {result['data_stats']['sparsity']:.3f}")
                print(f"  Alpha diversity: {result['data_stats']['alpha_diversity_mean']:.3f} ± {result['data_stats']['alpha_diversity_std']:.3f}")
                
                # Validate data properties
                if result["data_stats"]["num_samples"] < 100:
                    result["issues"].append("Very few samples (< 100)")
                
                if result["data_stats"]["sparsity"] < 0.5:
                    result["issues"].append("Unusually low sparsity for microbiome data")
                
                if result["data_stats"]["alpha_diversity_mean"] < 1.0:
                    result["issues"].append("Unusually low alpha diversity")
                
                print("[OK] Data loaded and validated successfully")
                
            except Exception as e:
                error_msg = f"Failed to load data: {e}"
                print(f"[FAIL] {error_msg}")
                result["issues"].append(error_msg)
        
        # Determine overall status
        if not result["issues"]:
            result["status"] = "passed"
            print("\n[OK] American Gut data verification PASSED")
        else:
            result["status"] = "failed"
            print(f"\n[FAIL] American Gut data verification FAILED ({len(result['issues'])} issues)")
            if verbose:
                for issue in result["issues"]:
                    print(f"  - {issue}")
        
        return result
    
    def verify_hmp(self, verbose: bool = False) -> Dict:
        """Verify HMP dataset."""
        print("\n" + "="*50)
        print("VERIFYING HMP DATA")
        print("="*50)
        
        hmp_dir = self.data_dir / "hmp"
        result = {
            "status": "unknown",
            "files_found": {},
            "data_stats": {},
            "issues": []
        }
        
        if not hmp_dir.exists():
            result["status"] = "not_found"
            result["issues"].append("HMP data directory not found")
            print("[FAIL] HMP data directory not found")
            return result
        
        # Look for BIOM files
        biom_files = list(hmp_dir.glob("*.biom"))
        
        if not biom_files:
            result["status"] = "no_data"
            result["issues"].append("No BIOM files found in HMP directory")
            print("[FAIL] No HMP BIOM files found")
            return result
        
        print(f"Found {len(biom_files)} BIOM files:")
        for biom_file in biom_files:
            size_mb = biom_file.stat().st_size / (1024*1024)
            print(f"  - {biom_file.name} ({size_mb:.1f} MB)")
            
            result["files_found"][biom_file.name] = {
                "path": str(biom_file),
                "exists": True,
                "size_mb": size_mb
            }
        
        # Try to load first file if dependencies available
        if DEPENDENCIES_AVAILABLE and biom_files:
            try:
                print(f"\nLoading {biom_files[0].name} for validation...")
                
                dataset = HMPDataset(data_dir=str(hmp_dir))
                processed = dataset.load_and_preprocess()
                
                compositions = processed.compositions
                result["data_stats"] = {
                    "num_samples": compositions.shape[0],
                    "num_taxa": compositions.shape[1],
                    "sparsity": float(np.mean(compositions == 0)),
                    "alpha_diversity_mean": float(np.mean(alpha_diversity(compositions))),
                    "alpha_diversity_std": float(np.std(alpha_diversity(compositions)))
                }
                
                print(f"  Samples: {result['data_stats']['num_samples']:,}")
                print(f"  Taxa: {result['data_stats']['num_taxa']:,}")
                print(f"  Sparsity: {result['data_stats']['sparsity']:.3f}")
                print(f"  Alpha diversity: {result['data_stats']['alpha_diversity_mean']:.3f} ± {result['data_stats']['alpha_diversity_std']:.3f}")
                
                print("[OK] HMP data loaded successfully")
                
            except Exception as e:
                error_msg = f"Failed to load HMP data: {e}"
                print(f"[FAIL] {error_msg}")
                result["issues"].append(error_msg)
        
        # Determine status
        if not result["issues"]:
            result["status"] = "passed"
            print("\n[OK] HMP data verification PASSED")
        else:
            result["status"] = "failed"
            print(f"\n[FAIL] HMP data verification FAILED ({len(result['issues'])} issues)")
        
        return result
    
    def check_manifest(self) -> Dict:
        """Check data manifest file."""
        manifest_path = self.data_dir / "data_manifest.json"
        
        if not manifest_path.exists():
            return {
                "exists": False,
                "error": "No data manifest found"
            }
        
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            return {
                "exists": True,
                "manifest": manifest,
                "datasets_listed": list(manifest.get("datasets", {}).keys())
            }
        except Exception as e:
            return {
                "exists": True,
                "error": f"Failed to read manifest: {e}"
            }
    
    def verify_all(self, verbose: bool = False) -> Dict:
        """Verify all available datasets."""
        print("MICROBIOME DATA VERIFICATION")
        print("="*60)
        
        if not DEPENDENCIES_AVAILABLE:
            print("Warning: Limited verification due to missing dependencies")
            print("Install required packages for full validation")
        
        # Check manifest
        manifest_result = self.check_manifest()
        self.results["manifest"] = manifest_result
        
        if manifest_result["exists"] and "manifest" in manifest_result:
            print(f"\n[OK] Found data manifest with {len(manifest_result['datasets_listed'])} datasets")
        
        # Verify individual datasets
        datasets_to_check = ["american_gut", "hmp"]
        
        for dataset_name in datasets_to_check:
            if dataset_name == "american_gut":
                result = self.verify_american_gut(verbose=verbose)
            elif dataset_name == "hmp":
                result = self.verify_hmp(verbose=verbose)
            
            self.results["datasets"][dataset_name] = result
        
        # Generate summary
        passed_datasets = [
            name for name, result in self.results["datasets"].items()
            if result["status"] == "passed"
        ]
        
        failed_datasets = [
            name for name, result in self.results["datasets"].items()
            if result["status"] == "failed"
        ]
        
        self.results["summary"] = {
            "total_datasets": len(datasets_to_check),
            "passed": len(passed_datasets),
            "failed": len(failed_datasets),
            "passed_datasets": passed_datasets,
            "failed_datasets": failed_datasets
        }
        
        if len(passed_datasets) == len(datasets_to_check):
            self.results["verification_status"] = "passed"
        elif len(passed_datasets) > 0:
            self.results["verification_status"] = "partial"
        else:
            self.results["verification_status"] = "failed"
        
        # Print summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        
        for dataset_name, result in self.results["datasets"].items():
            status_symbol = {
                "passed": "[OK]",
                "failed": "[FAIL]",
                "not_found": "[-]",
                "no_data": "[-]"
            }.get(result["status"], "[?]")
            
            print(f"{status_symbol} {dataset_name}: {result['status']}")
            
            if verbose and result["issues"]:
                for issue in result["issues"]:
                    print(f"    - {issue}")
        
        print(f"\nOverall status: {self.results['verification_status']}")
        print(f"Datasets ready: {len(passed_datasets)}/{len(datasets_to_check)}")
        
        if passed_datasets:
            print(f"\nReady for use:")
            for dataset in passed_datasets:
                print(f"  from src.microbiome_datasets import load_dataset")
                print(f"  dataset = load_dataset('{dataset}', use_real_data=True)")
        
        return self.results
    
    def save_report(self, output_path: Optional[str] = None) -> Path:
        """Save verification report to JSON file."""
        if output_path is None:
            output_path = self.data_dir / "verification_report.json"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n[OK] Verification report saved: {output_path}")
        return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify downloaded microbiome data integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        choices=['american_gut', 'hmp'],
        help='Verify specific dataset only'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory to verify (default: data/)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information and issues'
    )
    
    parser.add_argument(
        '--save-report',
        type=str,
        help='Save verification report to specified file'
    )
    
    args = parser.parse_args()
    
    try:
        verifier = DataVerifier(data_dir=args.data_dir)
        
        if args.dataset:
            if args.dataset == "american_gut":
                result = verifier.verify_american_gut(verbose=args.verbose)
            elif args.dataset == "hmp":
                result = verifier.verify_hmp(verbose=args.verbose)
            
            success = result["status"] == "passed"
        else:
            results = verifier.verify_all(verbose=args.verbose)
            success = results["verification_status"] in ["passed", "partial"]
        
        # Save report if requested
        if args.save_report:
            verifier.save_report(args.save_report)
        
        if success:
            print("\n[OK] Data verification completed successfully!")
            return 0
        else:
            print("\n[FAIL] Data verification found issues. See output for details.")
            return 1
            
    except Exception as e:
        print(f"\n[FAIL] Verification error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())