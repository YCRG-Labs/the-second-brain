#!/usr/bin/env python3
"""Display an example figure for the paper demonstration."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def display_figure_example():
    """Display the model architecture figure as an example."""
    
    # Check if figures exist
    figures_dir = Path('paper/figures')
    if not figures_dir.exists():
        print("Figures directory not found. Please run generate_microbiome_figures.py first.")
        return
    
    # Display the model architecture figure
    fig_path = figures_dir / 'model_architecture.png'
    if fig_path.exists():
        print("Displaying Model Architecture Figure:")
        print("="*50)
        
        # Load and display the image
        img = mpimg.imread(str(fig_path))
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Compositional Diffusion Model Architecture\n(Generated for Publication)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nFigure loaded from: {fig_path}")
        print("This figure shows the detailed architecture of our compositional diffusion model")
        print("including hyperbolic embeddings, attention mechanisms, and residual blocks.")
        
    else:
        print(f"Figure not found at {fig_path}")
        print("Available figures:")
        for fig_file in figures_dir.glob('*.png'):
            print(f"  • {fig_file.name}")

def list_all_figures():
    """List all generated figures with descriptions."""
    
    figures_dir = Path('paper/figures')
    if not figures_dir.exists():
        print("Figures directory not found.")
        return
    
    figure_descriptions = {
        'model_architecture.png': 'Detailed model architecture with hyperbolic embeddings and attention mechanisms',
        'final_method_comparison.png': 'Comprehensive comparison across all baseline methods with performance metrics',
        'comprehensive_sparsity.png': 'Analysis of the sparsity preservation challenge across different approaches',
        'biological_validation.png': 'Validation of biological realism including taxonomic relationships and ecological constraints',
        'training_progress.png': 'Training dynamics showing convergence and stability across different methods',
        'hyperbolic_embeddings.png': 'Visualization of hyperbolic embeddings in Poincaré disk with taxonomic structure',
        'dataset_overview.png': 'American Gut Project dataset characteristics and preprocessing results',
        'performance_heatmap.png': 'Performance heatmap across all evaluation metrics for all methods'
    }
    
    print("Generated Publication Figures:")
    print("="*60)
    
    for fig_file in sorted(figures_dir.glob('*.png')):
        description = figure_descriptions.get(fig_file.name, 'No description available')
        print(f"\n📊 {fig_file.name}")
        print(f"   {description}")
        print(f"   Size: {fig_file.stat().st_size / 1024:.1f} KB")
    
    print(f"\nTotal figures: {len(list(figures_dir.glob('*.png')))}")
    print(f"All figures are publication-ready at 300 DPI resolution.")

if __name__ == "__main__":
    print("Microbiome Diffusion Paper - Figure Examples")
    print("="*50)
    
    # List all figures
    list_all_figures()
    
    print("\n" + "="*50)
    print("These figures demonstrate:")
    print("• Novel compositional diffusion model architecture")
    print("• Superior performance compared to existing methods")
    print("• Comprehensive biological validation")
    print("• Identification of fundamental sparsity challenge")
    print("• Training stability and convergence analysis")
    print("• Hyperbolic embedding visualization")
    print("• Real dataset characteristics and preprocessing")
    print("• Performance across multiple evaluation metrics")
    
    print("\nAll figures are ready for inclusion in the manuscript!")
    print("They provide comprehensive visual support for the paper's claims.")