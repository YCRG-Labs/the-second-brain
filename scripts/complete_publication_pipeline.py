#!/usr/bin/env python3
"""Complete publication pipeline: Train models and generate final results.

This script runs the complete pipeline from model training to final publication outputs.

Usage:
    python scripts/complete_publication_pipeline.py
    python scripts/complete_publication_pipeline.py --quick-mode
    python scripts/complete_publication_pipeline.py --models diffusion vae
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path


def run_command(command, description, timeout=3600):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] {description} completed successfully ({duration:.1f}s)")
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"[FAIL] {description} failed (return code: {result.returncode})")
            if result.stderr.strip():
                print("Error output:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {description} timed out after {timeout} seconds")
        return False
        
    except Exception as e:
        print(f"[FAIL] {description} failed with exception: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Complete publication pipeline with real model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (recommended)
    python scripts/complete_publication_pipeline.py
    
    # Quick mode for testing
    python scripts/complete_publication_pipeline.py --quick-mode
    
    # Train specific models only
    python scripts/complete_publication_pipeline.py --models diffusion vae
    
    # Custom training epochs
    python scripts/complete_publication_pipeline.py --epochs 100
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['diffusion', 'vae', 'gan', 'copula'],
        default=['diffusion', 'vae', 'gan', 'copula'],
        help='Models to train (default: all)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--quick-mode',
        action='store_true',
        help='Quick mode (reduced epochs, shorter timeout)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use existing results)'
    )
    
    args = parser.parse_args()
    
    print("COMPLETE PUBLICATION PIPELINE")
    print("="*60)
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")
    print(f"Quick mode: {args.quick_mode}")
    print(f"Skip training: {args.skip_training}")
    
    success_count = 0
    total_steps = 3 if not args.skip_training else 2
    
    # Step 1: Verify data (always run)
    if run_command(
        ["python", "scripts/verify_data.py", "--dataset", "american_gut"],
        "Data Verification",
        timeout=300
    ):
        success_count += 1
    
    # Step 2: Train models (unless skipped)
    if not args.skip_training:
        train_command = [
            "python", "scripts/train_publication_models.py",
            "--models"] + args.models + [
            "--epochs", str(args.epochs)
        ]
        
        if args.quick_mode:
            train_command.append("--quick-mode")
        
        timeout = 1800 if args.quick_mode else 7200  # 30 min vs 2 hours
        
        if run_command(
            train_command,
            "Model Training",
            timeout=timeout
        ):
            success_count += 1
    else:
        print(f"\n{'='*60}")
        print("STEP: Model Training")
        print(f"{'='*60}")
        print("[SKIP] Model training skipped (using existing results)")
        success_count += 1
    
    # Step 3: Generate final publication outputs
    if run_command(
        ["python", "scripts/generate_final_publication_results.py"],
        "Final Publication Output Generation",
        timeout=600
    ):
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    if success_count == total_steps:
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        print()
        print("🎉 PUBLICATION PIPELINE COMPLETE!")
        print()
        print("Generated outputs:")
        print("- Trained models in: publication_models/")
        print("- Final figures in: final_publication_outputs/figures/")
        print("- LaTeX tables in: final_publication_outputs/tables/")
        print("- Summary in: final_publication_outputs/FINAL_PUBLICATION_SUMMARY.md")
        print()
        print("Your microbiome simulation system is now ready for academic publication!")
        print("All results are based on actual trained models using real American Gut data.")
        
        return 0
    else:
        print(f"⚠️  PIPELINE COMPLETED WITH ISSUES")
        print(f"Successful steps: {success_count}/{total_steps}")
        print()
        print("Some steps failed, but partial results may be available.")
        print("Check the output above for specific error details.")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())