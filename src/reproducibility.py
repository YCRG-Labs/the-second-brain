"""Reproducibility infrastructure for experiments.

This module provides the ReproducibilityManager class for ensuring complete
reproducibility of experiments, including seed setting, environment logging,
and configuration saving.
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml


@dataclass
class EnvironmentInfo:
    """Complete environment information for reproducibility.
    
    Attributes:
        python_version: Python version string
        torch_version: PyTorch version
        numpy_version: NumPy version
        cuda_available: Whether CUDA is available
        cuda_version: CUDA version if available
        cudnn_version: cuDNN version if available
        gpu_name: GPU device name if available
        gpu_count: Number of GPU devices
        cpu_count: Number of CPU cores
        platform_system: Operating system name
        platform_release: OS release version
        platform_machine: Machine architecture
        hostname: Machine hostname (anonymized)
        packages: Dictionary of installed package versions
        git_commit: Current git commit hash if in repo
        git_branch: Current git branch if in repo
        git_dirty: Whether there are uncommitted changes
        timestamp: When environment was captured
    """
    python_version: str
    torch_version: str
    numpy_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    gpu_name: Optional[str]
    gpu_count: int
    cpu_count: int
    platform_system: str
    platform_release: str
    platform_machine: str
    hostname: str
    packages: Dict[str, str]
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: bool
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> 'EnvironmentInfo':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ReproducibilityManager:
    """Ensure complete reproducibility of experiments.
    
    This class provides utilities for:
    - Setting all random seeds deterministically
    - Logging complete environment information
    - Saving and loading experiment configurations
    - Generating reproduction scripts
    - Verifying environment compatibility
    
    Example:
        >>> manager = ReproducibilityManager(Path('./experiment'))
        >>> manager.set_seeds(42)
        >>> env_info = manager.log_environment()
        >>> manager.save_config({'learning_rate': 0.001})
    """
    
    def __init__(self, experiment_dir: Path):
        """Initialize with experiment directory.
        
        Args:
            experiment_dir: Directory for experiment artifacts
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.config_dir = self.experiment_dir / 'configs'
        self.env_dir = self.experiment_dir / 'environment'
        self.scripts_dir = self.experiment_dir / 'scripts'
        
        self.config_dir.mkdir(exist_ok=True)
        self.env_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
        
        self._seed: Optional[int] = None
        self._env_info: Optional[EnvironmentInfo] = None
    
    def set_seeds(self, seed: int = 42) -> None:
        """Set all random seeds for reproducibility.
        
        Sets seeds for:
        - Python's random module
        - NumPy's random generator
        - PyTorch CPU and CUDA generators
        - Enables deterministic CUDA operations
        
        Args:
            seed: Random seed value
        """
        self._seed = seed
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        
        # CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Deterministic operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Save seed to file
        seed_file = self.config_dir / 'seed.txt'
        with open(seed_file, 'w') as f:
            f.write(str(seed))
    
    def get_seed(self) -> Optional[int]:
        """Get the current seed value.
        
        Returns:
            Current seed or None if not set
        """
        return self._seed
    
    def log_environment(self) -> EnvironmentInfo:
        """Log complete environment information.
        
        Captures all relevant environment details including:
        - Python and package versions
        - Hardware information
        - Git repository state
        
        Returns:
            EnvironmentInfo object with all captured information
        """
        # Get installed packages
        packages = self._get_installed_packages()
        
        # Get git information
        git_commit, git_branch, git_dirty = self._get_git_info()
        
        # Get GPU information
        gpu_name = None
        gpu_count = 0
        cuda_version = None
        cudnn_version = None
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        
        # Anonymize hostname (hash it)
        hostname_hash = hashlib.sha256(platform.node().encode()).hexdigest()[:12]
        
        self._env_info = EnvironmentInfo(
            python_version=sys.version,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            cpu_count=os.cpu_count() or 0,
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            hostname=hostname_hash,
            packages=packages,
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty,
            timestamp=datetime.now().isoformat()
        )
        
        # Save to file
        self._env_info.to_json(self.env_dir / 'environment.json')
        
        return self._env_info
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get dictionary of installed package versions.
        
        Returns:
            Dictionary mapping package names to versions
        """
        packages = {}
        
        try:
            import pkg_resources
            for pkg in pkg_resources.working_set:
                packages[pkg.key] = pkg.version
        except ImportError:
            # Fallback: try importlib.metadata
            try:
                from importlib.metadata import distributions
                for dist in distributions():
                    packages[dist.metadata['Name'].lower()] = dist.version
            except ImportError:
                pass
        
        return packages
    
    def _get_git_info(self) -> tuple:
        """Get git repository information.
        
        Returns:
            Tuple of (commit_hash, branch_name, is_dirty)
        """
        git_commit = None
        git_branch = None
        git_dirty = False
        
        try:
            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.experiment_dir.parent
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.experiment_dir.parent
            )
            if result.returncode == 0:
                git_branch = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=self.experiment_dir.parent
            )
            if result.returncode == 0:
                git_dirty = len(result.stdout.strip()) > 0
                
        except (FileNotFoundError, subprocess.SubprocessError):
            # Git not available or not in a git repo
            pass
        
        return git_commit, git_branch, git_dirty
    
    def save_config(self, config: Dict[str, Any], name: str = 'config') -> Path:
        """Save experiment configuration.
        
        Saves configuration in both YAML and JSON formats for flexibility.
        
        Args:
            config: Configuration dictionary
            name: Configuration name (without extension)
            
        Returns:
            Path to saved YAML config file
        """
        # Save as YAML (human-readable)
        yaml_path = self.config_dir / f'{name}.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Save as JSON (machine-readable)
        json_path = self.config_dir / f'{name}.json'
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return yaml_path
    
    def load_config(self, name: str = 'config') -> Dict[str, Any]:
        """Load experiment configuration.
        
        Args:
            name: Configuration name (without extension)
            
        Returns:
            Configuration dictionary
        """
        yaml_path = self.config_dir / f'{name}.yaml'
        json_path = self.config_dir / f'{name}.json'
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
        elif json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Config '{name}' not found in {self.config_dir}")
    
    def create_reproduction_script(
        self,
        experiment_name: str,
        python_script: str,
        data_download_cmd: Optional[str] = None,
        additional_setup: Optional[List[str]] = None
    ) -> Path:
        """Generate shell script to reproduce experiment.
        
        Creates a complete reproduction script that:
        - Sets up the environment
        - Downloads data if needed
        - Runs the experiment with saved configuration
        
        Args:
            experiment_name: Name for the experiment
            python_script: Python script to run
            data_download_cmd: Optional command to download data
            additional_setup: Optional list of additional setup commands
            
        Returns:
            Path to generated script
        """
        script_lines = [
            '#!/bin/bash',
            '# Reproduction script for: ' + experiment_name,
            '# Generated: ' + datetime.now().isoformat(),
            '',
            'set -e  # Exit on error',
            '',
            '# Configuration',
            f'EXPERIMENT_NAME="{experiment_name}"',
            f'SEED={self._seed or 42}',
            '',
            '# Check Python version',
            'echo "Python version: $(python --version)"',
            '',
        ]
        
        # Add data download if provided
        if data_download_cmd:
            script_lines.extend([
                '# Download data',
                f'{data_download_cmd}',
                '',
            ])
        
        # Add additional setup commands
        if additional_setup:
            script_lines.extend([
                '# Additional setup',
            ])
            script_lines.extend(additional_setup)
            script_lines.append('')
        
        # Add main experiment command
        script_lines.extend([
            '# Run experiment',
            f'python {python_script} \\',
            f'    --config {self.config_dir / "config.yaml"} \\',
            f'    --seed $SEED \\',
            f'    --output-dir {self.experiment_dir}',
            '',
            'echo "Experiment completed successfully!"',
        ])
        
        # Write script
        script_path = self.scripts_dir / f'reproduce_{experiment_name}.sh'
        with open(script_path, 'w', newline='\n') as f:
            f.write('\n'.join(script_lines))
        
        # Make executable on Unix
        if platform.system() != 'Windows':
            os.chmod(script_path, 0o755)
        
        # Also create a Windows batch file
        bat_lines = [
            '@echo off',
            f'REM Reproduction script for: {experiment_name}',
            f'REM Generated: {datetime.now().isoformat()}',
            '',
            f'set EXPERIMENT_NAME={experiment_name}',
            f'set SEED={self._seed or 42}',
            '',
            'echo Python version:',
            'python --version',
            '',
        ]
        
        if data_download_cmd:
            bat_lines.extend([
                'REM Download data',
                data_download_cmd.replace('/', '\\'),
                '',
            ])
        
        bat_lines.extend([
            'REM Run experiment',
            f'python {python_script} ^',
            f'    --config {self.config_dir / "config.yaml"} ^',
            f'    --seed %SEED% ^',
            f'    --output-dir {self.experiment_dir}',
            '',
            'echo Experiment completed successfully!',
        ])
        
        bat_path = self.scripts_dir / f'reproduce_{experiment_name}.bat'
        with open(bat_path, 'w') as f:
            f.write('\n'.join(bat_lines))
        
        return script_path
    
    def verify_environment(self, reference_env: Union[Path, EnvironmentInfo]) -> Dict[str, Any]:
        """Verify current environment matches reference.
        
        Compares current environment against a reference to identify
        potential reproducibility issues.
        
        Args:
            reference_env: Path to environment JSON or EnvironmentInfo object
            
        Returns:
            Dictionary with verification results and any mismatches
        """
        if isinstance(reference_env, Path):
            reference = EnvironmentInfo.from_json(reference_env)
        else:
            reference = reference_env
        
        current = self.log_environment()
        
        mismatches = []
        warnings = []
        
        # Check critical versions
        if current.torch_version != reference.torch_version:
            mismatches.append({
                'field': 'torch_version',
                'expected': reference.torch_version,
                'actual': current.torch_version
            })
        
        if current.numpy_version != reference.numpy_version:
            warnings.append({
                'field': 'numpy_version',
                'expected': reference.numpy_version,
                'actual': current.numpy_version
            })
        
        # Check CUDA availability
        if reference.cuda_available and not current.cuda_available:
            mismatches.append({
                'field': 'cuda_available',
                'expected': True,
                'actual': False,
                'message': 'Reference used CUDA but current environment does not have CUDA'
            })
        
        # Check key packages
        key_packages = ['scipy', 'hypothesis', 'pytest']
        for pkg in key_packages:
            ref_version = reference.packages.get(pkg)
            cur_version = current.packages.get(pkg)
            if ref_version and cur_version and ref_version != cur_version:
                warnings.append({
                    'field': f'packages.{pkg}',
                    'expected': ref_version,
                    'actual': cur_version
                })
        
        return {
            'compatible': len(mismatches) == 0,
            'mismatches': mismatches,
            'warnings': warnings,
            'reference': reference.to_dict(),
            'current': current.to_dict()
        }
    
    def get_environment_info(self) -> Optional[EnvironmentInfo]:
        """Get cached environment info.
        
        Returns:
            Cached EnvironmentInfo or None if not logged yet
        """
        return self._env_info
    
    def create_requirements_txt(self, output_path: Optional[Path] = None) -> Path:
        """Generate requirements.txt with exact versions.
        
        Args:
            output_path: Optional output path (defaults to env_dir)
            
        Returns:
            Path to generated requirements.txt
        """
        if output_path is None:
            output_path = self.env_dir / 'requirements.txt'
        
        packages = self._get_installed_packages()
        
        # Sort packages alphabetically
        sorted_packages = sorted(packages.items())
        
        lines = [
            '# Auto-generated requirements for reproducibility',
            f'# Generated: {datetime.now().isoformat()}',
            '#',
        ]
        
        for name, version in sorted_packages:
            lines.append(f'{name}=={version}')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def create_conda_environment_yml(
        self,
        env_name: str = 'microbiome-sim',
        output_path: Optional[Path] = None
    ) -> Path:
        """Generate conda environment.yml file.
        
        Args:
            env_name: Name for the conda environment
            output_path: Optional output path (defaults to env_dir)
            
        Returns:
            Path to generated environment.yml
        """
        if output_path is None:
            output_path = self.env_dir / 'environment.yml'
        
        packages = self._get_installed_packages()
        
        # Core dependencies that should come from conda
        conda_packages = [
            'python',
            'numpy',
            'scipy',
            'pytorch',
            'cudatoolkit' if torch.cuda.is_available() else None,
        ]
        conda_packages = [p for p in conda_packages if p]
        
        # Pip dependencies
        pip_packages = []
        for name, version in sorted(packages.items()):
            if name not in ['python', 'numpy', 'scipy', 'torch', 'pytorch']:
                pip_packages.append(f'{name}=={version}')
        
        env_dict = {
            'name': env_name,
            'channels': ['pytorch', 'conda-forge', 'defaults'],
            'dependencies': [
                f'python={sys.version_info.major}.{sys.version_info.minor}',
                f'numpy={np.__version__}',
                f'pytorch={torch.__version__}',
            ]
        }
        
        if torch.cuda.is_available() and torch.version.cuda:
            cuda_major = torch.version.cuda.split('.')[0]
            env_dict['dependencies'].append(f'cudatoolkit={cuda_major}.*')
        
        # Add pip dependencies
        if pip_packages:
            env_dict['dependencies'].append({'pip': pip_packages[:20]})  # Limit to avoid huge file
        
        with open(output_path, 'w') as f:
            yaml.dump(env_dict, f, default_flow_style=False, sort_keys=False)
        
        return output_path


def set_global_seeds(seed: int = 42) -> None:
    """Convenience function to set all random seeds globally.
    
    This is a standalone function for quick seed setting without
    creating a ReproducibilityManager instance.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
