"""Custom exceptions for microbiome simulation system.

This module defines domain-specific exceptions for input validation,
numerical stability, and runtime errors.
"""


class MicrobiomeSimulationError(Exception):
    """Base exception for microbiome simulation errors."""
    pass


# Input Validation Errors

class InvalidCompositionError(MicrobiomeSimulationError):
    """Raised when composition doesn't sum to 1 or has negative values."""
    pass


class EmptyTreeError(MicrobiomeSimulationError):
    """Raised when phylogenetic tree has no taxa."""
    pass


class InsufficientReadsError(MicrobiomeSimulationError):
    """Raised when sample has fewer than minimum reads."""
    pass


class DimensionMismatchError(MicrobiomeSimulationError):
    """Raised when embedding/abundance dimensions don't match."""
    pass


# Runtime Errors

class CheckpointIncompatibleError(MicrobiomeSimulationError):
    """Raised when checkpoint architecture doesn't match model."""
    pass


class ConvergenceError(MicrobiomeSimulationError):
    """Raised when optimization fails to converge."""
    pass
