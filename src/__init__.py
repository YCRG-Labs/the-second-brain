"""Microbiome Simulation System.

A spatial-temporal microbiome dynamics simulator using computer vision approaches.
"""

__version__ = "0.1.0"

from src.clr_transform import CLRTransform
from src.hyperbolic import (
    poincare_distance,
    exponential_map,
    logarithmic_map,
    mobius_add,
    HyperbolicEmbedder,
)
from src.rasterization import (
    stereographic_project,
    DifferentiableRasterizer,
)
from src.diffusion import (
    NoiseSchedule,
    SinusoidalTimeEmbedding,
    UNetDenoiser,
    CompositionalDiffusion,
)
from src.zero_inflated import (
    ZeroInflatedDecoder,
    RealisticMicrobiomeEncoder,
    RealisticMicrobiomeVAE,
    ZeroInflatedDiffusionDecoder,
)
from src.diversity_loss import (
    DiversityMatchingLoss,
    RBFKernel,
    MultiScaleRBFKernel,
    differentiable_shannon_entropy,
    differentiable_bray_curtis,
    differentiable_beta_diversity,
    compute_mmd,
)
from src.sparsity_loss import (
    SparsityLoss,
    RareTaxaLoss,
    compute_sparsity,
    compute_prevalence,
    compute_target_sparsity_from_data,
    compute_target_prevalence_from_data,
)
from src.spatiotemporal import (
    PatchEmbedding,
    SpatialAttention,
    TemporalAttention,
    FactorizedTransformerBlock,
    SpatiotemporalTransformer,
    construct_video_sequence,
    decompose_frame_to_patches,
    reconstruct_frame_from_patches,
)
from src.neural_field import (
    PositionalEncoding,
    NeuralMicrobiomeField,
)
from src.preprocessing import (
    PreprocessingPipeline,
)
from src.serialization import (
    ModelSerializer,
)
from src.evaluation import (
    shannon_entropy,
    alpha_diversity,
    bray_curtis_dissimilarity,
    beta_diversity,
    microbiome_frechet_distance,
    abundance_mae,
    top_k_accuracy,
    MicrobiomeEvaluator,
    ComprehensiveEvaluator,
    BiologicalValidator,
    MethodComparator,
    ComparisonVisualizer,
)
from src.figures import (
    FigureGenerator,
    LaTeXTableGenerator,
    PUBLICATION_COLORS,
    METHOD_COLORS,
)
from src.reproducibility import (
    ReproducibilityManager,
    EnvironmentInfo,
    set_global_seeds,
)
from src.model_zoo import (
    ModelZoo,
    ModelCard,
    load_pretrained,
    list_pretrained_models,
    get_model_zoo,
)
from src.realistic_model import (
    RealisticMicrobiomeModel,
    RealisticTrainingConfig,
)

__all__ = [
    "CLRTransform",
    "poincare_distance",
    "exponential_map",
    "logarithmic_map",
    "mobius_add",
    "HyperbolicEmbedder",
    "stereographic_project",
    "DifferentiableRasterizer",
    "NoiseSchedule",
    "SinusoidalTimeEmbedding",
    "UNetDenoiser",
    "CompositionalDiffusion",
    "ZeroInflatedDecoder",
    "RealisticMicrobiomeEncoder",
    "RealisticMicrobiomeVAE",
    "ZeroInflatedDiffusionDecoder",
    # Diversity Loss
    "DiversityMatchingLoss",
    "RBFKernel",
    "MultiScaleRBFKernel",
    "differentiable_shannon_entropy",
    "differentiable_bray_curtis",
    "differentiable_beta_diversity",
    "compute_mmd",
    # Sparsity Loss
    "SparsityLoss",
    "RareTaxaLoss",
    "compute_sparsity",
    "compute_prevalence",
    "compute_target_sparsity_from_data",
    "compute_target_prevalence_from_data",
    "PatchEmbedding",
    "SpatialAttention",
    "TemporalAttention",
    "FactorizedTransformerBlock",
    "SpatiotemporalTransformer",
    "construct_video_sequence",
    "decompose_frame_to_patches",
    "reconstruct_frame_from_patches",
    "PositionalEncoding",
    "NeuralMicrobiomeField",
    "PreprocessingPipeline",
    "ModelSerializer",
    # Evaluation
    "shannon_entropy",
    "alpha_diversity",
    "bray_curtis_dissimilarity",
    "beta_diversity",
    "microbiome_frechet_distance",
    "abundance_mae",
    "top_k_accuracy",
    "MicrobiomeEvaluator",
    "ComprehensiveEvaluator",
    "BiologicalValidator",
    "MethodComparator",
    "ComparisonVisualizer",
    # Figure generation
    "FigureGenerator",
    "LaTeXTableGenerator",
    "PUBLICATION_COLORS",
    "METHOD_COLORS",
    # Reproducibility
    "ReproducibilityManager",
    "EnvironmentInfo",
    "set_global_seeds",
    # Model Zoo
    "ModelZoo",
    "ModelCard",
    "load_pretrained",
    "list_pretrained_models",
    "get_model_zoo",
    # Realistic Microbiome Model
    "RealisticMicrobiomeModel",
    "RealisticTrainingConfig",
]
