"""Publication package for organizing and compiling research outputs.

This module implements the PublicationPackage class for organizing all
publication-ready outputs including figures, tables, results summaries,
and statistical analyses.

Requirements: 5.4 from publication-ready spec
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import warnings

from .evaluation import MicrobiomeEvaluator, MethodComparator, BiologicalValidator
from .figures import FigureGenerator, LaTeXTableGenerator
from .benchmarking import BenchmarkSuite


@dataclass
class PublicationPackage:
    """Comprehensive package for organizing all publication outputs.
    
    This class provides a unified interface for all publication-ready outputs
    including figures, LaTeX tables, result summaries, and statistical analyses.
    
    Attributes:
        figures: Dict mapping figure names to file paths
        tables: Dict mapping table names to LaTeX code strings
        results_summary: Dict with comprehensive results data
        statistical_significance: Dict with statistical test results
        metadata: Additional metadata about the publication package
    """
    figures: Dict[str, Path] = field(default_factory=dict)
    tables: Dict[str, str] = field(default_factory=dict)
    results_summary: Dict[str, Any] = field(default_factory=dict)
    statistical_significance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PublicationCompiler:
    """Comprehensive result compilation system for publication outputs.
    
    This class orchestrates the generation of all publication materials:
    - Evaluation metrics and statistical comparisons
    - Publication-quality figures
    - LaTeX tables for quantitative results
    - Comprehensive result summaries
    - Validation reports
    
    Attributes:
        output_dir: Base directory for all outputs
        evaluator: MicrobiomeEvaluator for computing metrics
        figure_generator: FigureGenerator for creating plots
        table_generator: LaTeXTableGenerator for creating tables
        method_comparator: MethodComparator for statistical testing
        biological_validator: Optional BiologicalValidator for biological validation
        benchmark_suite: Optional BenchmarkSuite for performance analysis
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        phylogenetic_kernel: np.ndarray,
        evaluator: Optional[MicrobiomeEvaluator] = None,
        figure_generator: Optional[FigureGenerator] = None,
        table_generator: Optional[LaTeXTableGenerator] = None,
        method_comparator: Optional[MethodComparator] = None,
        biological_validator: Optional[BiologicalValidator] = None,
        benchmark_suite: Optional[BenchmarkSuite] = None
    ):
        """Initialize publication compiler.
        
        Args:
            output_dir: Base directory for all publication outputs
            phylogenetic_kernel: Phylogenetic distance matrix for evaluations
            evaluator: Optional MicrobiomeEvaluator (created if None)
            figure_generator: Optional FigureGenerator (created if None)
            table_generator: Optional LaTeXTableGenerator (created if None)
            method_comparator: Optional MethodComparator (created if None)
            biological_validator: Optional BiologicalValidator
            benchmark_suite: Optional BenchmarkSuite
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.evaluator = evaluator or MicrobiomeEvaluator(
            phylogenetic_kernel=phylogenetic_kernel,
            biological_validator=biological_validator,
            method_comparator=method_comparator
        )
        
        self.figure_generator = figure_generator or FigureGenerator()
        self.table_generator = table_generator or LaTeXTableGenerator()
        self.method_comparator = method_comparator or MethodComparator()
        self.biological_validator = biological_validator
        self.benchmark_suite = benchmark_suite
        
        # Create subdirectories
        self.figures_dir = self.output_dir / 'figures'
        self.tables_dir = self.output_dir / 'tables'
        self.results_dir = self.output_dir / 'results'
        
        for dir_path in [self.figures_dir, self.tables_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def compile_method_comparison(
        self,
        method_data: Dict[str, Dict[str, Any]],
        metrics_to_include: Optional[List[str]] = None,
        include_biological_validation: bool = True,
        include_statistical_tests: bool = True
    ) -> PublicationPackage:
        """Compile comprehensive method comparison package.
        
        Args:
            method_data: Dict mapping method names to their data:
                        - 'real_compositions': Real microbiome samples
                        - 'generated_compositions': Generated samples
                        - 'predictions': Optional predicted compositions
                        - 'ground_truth': Optional true compositions for prediction
            metrics_to_include: List of metrics to include (default: all)
            include_biological_validation: Whether to run biological validation
            include_statistical_tests: Whether to perform statistical tests
        
        Returns:
            PublicationPackage with all compiled outputs
        """
        package = PublicationPackage()
        
        # Step 1: Evaluate all methods
        print("Evaluating methods...")
        method_results = {}
        
        for method_name, data in method_data.items():
            try:
                results = self.evaluator.evaluate_all(
                    real_compositions=data['real_compositions'],
                    generated_compositions=data['generated_compositions'],
                    predictions=data.get('predictions'),
                    ground_truth=data.get('ground_truth'),
                    include_biological_validation=include_biological_validation
                )
                method_results[method_name] = results
                print(f"  ✓ {method_name}: {len(results)} metrics computed")
            except Exception as e:
                print(f"  ✗ {method_name}: Evaluation failed - {e}")
                method_results[method_name] = {}
        
        # Step 2: Statistical comparison
        if include_statistical_tests and len(method_results) >= 2:
            print("Performing statistical comparisons...")
            try:
                comparison_results = self.evaluator.compare_methods(
                    method_results,
                    metric_names=metrics_to_include,
                    include_statistical_tests=True
                )
                package.statistical_significance = comparison_results
                print(f"  ✓ Statistical tests completed for {len(comparison_results)} metrics")
            except Exception as e:
                print(f"  ✗ Statistical comparison failed: {e}")
                package.statistical_significance = {}
        
        # Step 3: Generate figures
        print("Generating figures...")
        self._generate_comparison_figures(
            method_data, method_results, package
        )
        
        # Step 4: Generate LaTeX tables
        print("Generating LaTeX tables...")
        self._generate_comparison_tables(
            method_results, package.statistical_significance, package
        )
        
        # Step 5: Compile results summary
        print("Compiling results summary...")
        package.results_summary = self._compile_results_summary(
            method_results, package.statistical_significance
        )
        
        # Step 6: Add metadata
        package.metadata = {
            'num_methods': len(method_data),
            'metrics_evaluated': list(method_results[list(method_results.keys())[0]].keys()) if method_results else [],
            'biological_validation_included': include_biological_validation,
            'statistical_tests_included': include_statistical_tests,
            'output_directory': str(self.output_dir),
            'compilation_timestamp': np.datetime64('now').astype(str)
        }
        
        # Step 7: Save complete package
        self._save_publication_package(package)
        
        print(f"Publication package compiled successfully!")
        print(f"  - {len(package.figures)} figures generated")
        print(f"  - {len(package.tables)} tables generated")
        print(f"  - Results saved to {self.output_dir}")
        
        return package
    
    def _generate_comparison_figures(
        self,
        method_data: Dict[str, Dict[str, Any]],
        method_results: Dict[str, Dict[str, Any]],
        package: PublicationPackage
    ) -> None:
        """Generate all comparison figures."""
        try:
            # Figure 1: Generation quality comparison
            if len(method_data) >= 2:
                first_method = list(method_data.keys())[0]
                real_data = method_data[first_method]['real_compositions']
                
                # Create combined plot for all methods
                fig_path = self.figures_dir / 'generation_quality_comparison'
                
                # For now, plot the first two methods
                method_names = list(method_data.keys())[:2]
                if len(method_names) >= 2:
                    self.figure_generator.plot_generation_quality(
                        real_data=real_data,
                        generated_data=method_data[method_names[1]]['generated_compositions'],
                        output_path=fig_path,
                        method_name=method_names[1]
                    )
                    package.figures['generation_quality'] = fig_path.with_suffix('.pdf')
            
            # Figure 2: MFD comparison bar chart
            mfd_values = {}
            for method_name, results in method_results.items():
                if 'mfd' in results:
                    mfd_values[method_name] = results['mfd']
            
            if mfd_values:
                fig_path = self.figures_dir / 'mfd_comparison'
                self.figure_generator.plot_mfd_comparison(
                    method_results=mfd_values,
                    output_path=fig_path
                )
                package.figures['mfd_comparison'] = fig_path.with_suffix('.pdf')
            
            # Figure 3: Prediction accuracy (if prediction data available)
            prediction_methods = {}
            ground_truth_data = None
            time_points = None
            
            for method_name, data in method_data.items():
                if 'predictions' in data and 'ground_truth' in data:
                    prediction_methods[method_name] = data['predictions']
                    if ground_truth_data is None:
                        ground_truth_data = data['ground_truth']
                        # Create dummy time points if not provided
                        if ground_truth_data.ndim == 3:
                            time_points = np.arange(ground_truth_data.shape[1])
                        else:
                            time_points = np.array([0])
            
            if prediction_methods and ground_truth_data is not None:
                fig_path = self.figures_dir / 'prediction_accuracy'
                self.figure_generator.plot_prediction_accuracy(
                    predictions=prediction_methods,
                    ground_truth=ground_truth_data,
                    time_points=time_points,
                    output_path=fig_path
                )
                package.figures['prediction_accuracy'] = fig_path.with_suffix('.pdf')
        
        except Exception as e:
            print(f"Warning: Figure generation failed: {e}")
    
    def _generate_comparison_tables(
        self,
        method_results: Dict[str, Dict[str, Any]],
        statistical_results: Dict[str, Any],
        package: PublicationPackage
    ) -> None:
        """Generate all comparison tables."""
        try:
            # Get common metrics across all methods
            all_metrics = set()
            for results in method_results.values():
                all_metrics.update(results.keys())
            
            # Filter to numeric metrics only
            numeric_metrics = []
            for metric in all_metrics:
                # Check if all methods have numeric values for this metric
                is_numeric = True
                for results in method_results.values():
                    if metric in results:
                        value = results[metric]
                        if not isinstance(value, (int, float, np.number)):
                            is_numeric = False
                            break
                
                if is_numeric:
                    numeric_metrics.append(metric)
            
            numeric_metrics = sorted(numeric_metrics)
            
            if numeric_metrics:
                # Table 1: Method comparison table
                table_path = self.tables_dir / 'method_comparison.tex'
                latex_table = self.table_generator.generate_comparison_table(
                    results=method_results,
                    metrics=numeric_metrics,
                    output_path=table_path,
                    caption='Comprehensive method comparison across evaluation metrics',
                    label='tab:method_comparison'
                )
                package.tables['method_comparison'] = latex_table
                
                # Table 2: Statistical significance table (if statistical tests were run)
                if statistical_results:
                    sig_table_path = self.tables_dir / 'statistical_significance.tex'
                    sig_latex = self.table_generator.generate_significance_table(
                        comparison_results=statistical_results,
                        output_path=sig_table_path,
                        caption='Statistical significance tests for method comparisons',
                        label='tab:statistical_significance'
                    )
                    package.tables['statistical_significance'] = sig_latex
        
        except Exception as e:
            print(f"Warning: Table generation failed: {e}")
    
    def _compile_results_summary(
        self,
        method_results: Dict[str, Dict[str, Any]],
        statistical_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile comprehensive results summary."""
        summary = {
            'method_results': method_results,
            'statistical_comparison': statistical_results,
            'best_methods': {},
            'significant_differences': {},
            'overall_statistics': {}
        }
        
        # Identify best method for each metric
        all_metrics = set()
        for results in method_results.values():
            all_metrics.update(results.keys())
        
        for metric in all_metrics:
            # Get values for this metric
            values = {}
            for method_name, results in method_results.items():
                if metric in results and isinstance(results[metric], (int, float, np.number)):
                    values[method_name] = float(results[metric])
            
            if values:
                # Determine if lower is better
                is_error_metric = any(
                    term in metric.lower()
                    for term in ['mae', 'mse', 'error', 'loss', 'mfd', 'violation', 'distance']
                )
                
                if is_error_metric:
                    best_method = min(values.items(), key=lambda x: x[1])
                else:
                    best_method = max(values.items(), key=lambda x: x[1])
                
                summary['best_methods'][metric] = {
                    'method': best_method[0],
                    'value': best_method[1]
                }
                
                # Overall statistics for this metric
                summary['overall_statistics'][metric] = {
                    'mean': float(np.mean(list(values.values()))),
                    'std': float(np.std(list(values.values()))),
                    'min': float(min(values.values())),
                    'max': float(max(values.values())),
                    'range': float(max(values.values()) - min(values.values()))
                }
        
        # Count significant differences
        if statistical_results:
            for metric_name, metric_comparison in statistical_results.items():
                if 'pairwise_tests' in metric_comparison:
                    significant_pairs = [
                        pair for pair, test_result in metric_comparison['pairwise_tests'].items()
                        if test_result.get('significant', False)
                    ]
                    summary['significant_differences'][metric_name] = len(significant_pairs)
        
        return summary
    
    def _save_publication_package(self, package: PublicationPackage) -> None:
        """Save complete publication package to disk."""
        # Save package metadata and summary
        package_data = {
            'figures': {name: str(path) for name, path in package.figures.items()},
            'tables': package.tables,
            'results_summary': package.results_summary,
            'statistical_significance': package.statistical_significance,
            'metadata': package.metadata
        }
        
        # Convert numpy types to native Python types for JSON serialization
        package_data = self._convert_numpy_types(package_data)
        
        # Save as JSON
        with open(self.output_dir / 'publication_package.json', 'w') as f:
            json.dump(package_data, f, indent=2)
        
        # Save individual components
        with open(self.results_dir / 'method_results.json', 'w') as f:
            json.dump(self._convert_numpy_types(package.results_summary.get('method_results', {})), f, indent=2)
        
        if package.statistical_significance:
            with open(self.results_dir / 'statistical_tests.json', 'w') as f:
                json.dump(self._convert_numpy_types(package.statistical_significance), f, indent=2)
        
        # Create README with package contents
        self._create_package_readme(package)
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _create_package_readme(self, package: PublicationPackage) -> None:
        """Create README file describing package contents."""
        readme_content = []
        
        # Header
        readme_content.extend([
            "# Publication Package",
            "",
            "This directory contains all publication-ready outputs generated by the microbiome simulation system.",
            "",
            "## Contents",
            "",
            "### Figures",
            ""
        ])
        
        # Add figures
        for fig_name, fig_path in package.figures.items():
            readme_content.append(f"- `{fig_path.name}`: {fig_name.replace('_', ' ').title()}")
        
        # Tables section
        readme_content.extend([
            "",
            "### Tables",
            ""
        ])
        
        for table_name in package.tables.keys():
            readme_content.append(f"- `tables/{table_name}.tex`: {table_name.replace('_', ' ').title()}")
        
        # Results section
        readme_content.extend([
            "",
            "### Results",
            "",
            "- `results/method_results.json`: Individual method evaluation results",
            "- `results/statistical_tests.json`: Statistical significance test results",
            "- `publication_package.json`: Complete package metadata and summary",
            "",
            "## Usage",
            "",
            "### Including Figures in LaTeX",
            "",
            "```latex",
            "\\begin{figure}[htbp]",
            "    \\centering",
            "    \\includegraphics[width=0.8\\textwidth]{figures/generation_quality_comparison.pdf}",
            "    \\caption{Generation quality comparison}",
            "    \\label{fig:generation_quality}",
            "\\end{figure}",
            "```",
            "",
            "### Including Tables in LaTeX",
            "",
            "```latex",
            "\\input{tables/method_comparison.tex}",
            "```",
            "",
            "## Summary",
            ""
        ])
        
        # Add summary statistics
        num_methods = package.metadata.get('num_methods', 0)
        metrics_evaluated = package.metadata.get('metrics_evaluated', [])
        num_figures = len(package.figures)
        num_tables = len(package.tables)
        compilation_date = package.metadata.get('compilation_timestamp', 'Unknown')
        
        readme_content.extend([
            f"- **Methods evaluated**: {num_methods}",
            f"- **Metrics computed**: {len(metrics_evaluated)}",
            f"- **Figures generated**: {num_figures}",
            f"- **Tables generated**: {num_tables}",
            f"- **Compilation date**: {compilation_date}",
            ""
        ])
        
        # Add best methods summary
        if package.results_summary.get('best_methods'):
            readme_content.extend([
                "## Best Performing Methods",
                ""
            ])
            
            for metric, best_info in package.results_summary['best_methods'].items():
                method_name = best_info['method']
                value = best_info['value']
                readme_content.append(f"- **{metric.replace('_', ' ').title()}**: {method_name} ({value:.4f})")
        
        # Write README
        with open(self.output_dir / 'README.md', 'w') as f:
            f.write('\n'.join(readme_content))
    
    def validate_package_completeness(self, package: PublicationPackage) -> Dict[str, bool]:
        """Validate that the publication package is complete.
        
        Args:
            package: PublicationPackage to validate
        
        Returns:
            Dict mapping validation checks to pass/fail status
        """
        validation_results = {}
        
        # Check figures exist
        validation_results['figures_exist'] = all(
            Path(path).exists() for path in package.figures.values()
        )
        
        # Check tables are non-empty
        validation_results['tables_non_empty'] = all(
            len(table.strip()) > 0 for table in package.tables.values()
        )
        
        # Check results summary has required fields
        required_summary_fields = ['method_results', 'best_methods', 'overall_statistics']
        validation_results['summary_complete'] = all(
            field in package.results_summary for field in required_summary_fields
        )
        
        # Check metadata has required fields
        required_metadata_fields = ['num_methods', 'metrics_evaluated', 'compilation_timestamp']
        validation_results['metadata_complete'] = all(
            field in package.metadata for field in required_metadata_fields
        )
        
        # Check output directory structure
        required_dirs = [self.figures_dir, self.tables_dir, self.results_dir]
        validation_results['directory_structure'] = all(
            dir_path.exists() and dir_path.is_dir() for dir_path in required_dirs
        )
        
        # Overall validation
        validation_results['overall_valid'] = all(validation_results.values())
        
        return validation_results
    
    def generate_validation_report(self, package: PublicationPackage) -> str:
        """Generate validation report for publication package.
        
        Args:
            package: PublicationPackage to validate
        
        Returns:
            Validation report as string
        """
        validation_results = self.validate_package_completeness(package)
        
        report_lines = [
            "# Publication Package Validation Report",
            "",
            f"**Validation Date**: {np.datetime64('now').astype(str)}",
            f"**Package Directory**: {self.output_dir}",
            "",
            "## Validation Results",
            ""
        ]
        
        for check_name, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            check_display = check_name.replace('_', ' ').title()
            report_lines.append(f"- **{check_display}**: {status}")
        
        report_lines.extend([
            "",
            "## Package Contents",
            "",
            f"### Figures ({len(package.figures)})",
            ""
        ])
        
        for fig_name, fig_path in package.figures.items():
            exists = "OK" if Path(fig_path).exists() else "MISSING"
            report_lines.append(f"- {exists} `{fig_path.name}`: {fig_name.replace('_', ' ').title()}")
        
        report_lines.extend([
            "",
            f"### Tables ({len(package.tables)})",
            ""
        ])
        
        for table_name in package.tables.keys():
            report_lines.append(f"- OK `{table_name}.tex`: {table_name.replace('_', ' ').title()}")
        
        report_lines.extend([
            "",
            "### Results Summary",
            ""
        ])
        
        if package.results_summary.get('best_methods'):
            for metric, best_info in package.results_summary['best_methods'].items():
                method_name = best_info['method']
                value = best_info['value']
                report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {method_name} ({value:.4f})")
        
        # Overall status
        overall_status = "COMPLETE" if validation_results['overall_valid'] else "INCOMPLETE"
        report_lines.extend([
            "",
            f"## Overall Status: {overall_status}",
            ""
        ])
        
        if not validation_results['overall_valid']:
            failed_checks = [name for name, passed in validation_results.items() if not passed]
            report_lines.append("**Failed Checks:**")
            for check in failed_checks:
                report_lines.append(f"- {check.replace('_', ' ').title()}")
        
        report_text = '\n'.join(report_lines)
        
        # Save validation report
        with open(self.output_dir / 'validation_report.md', 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def create_publication_archive(
        self,
        package: PublicationPackage,
        archive_path: Optional[Union[str, Path]] = None,
        include_source_data: bool = False
    ) -> Path:
        """Create compressed archive of publication package.
        
        Args:
            package: PublicationPackage to archive
            archive_path: Path for archive file (default: output_dir/publication_package.zip)
            include_source_data: Whether to include source data files
        
        Returns:
            Path to created archive
        """
        if archive_path is None:
            archive_path = self.output_dir.parent / f"{self.output_dir.name}_publication_package"
        
        archive_path = Path(archive_path)
        
        try:
            # Create archive
            shutil.make_archive(
                str(archive_path),
                'zip',
                str(self.output_dir)
            )
            
            final_archive_path = archive_path.with_suffix('.zip')
            print(f"Publication archive created: {final_archive_path}")
            
            return final_archive_path
        
        except Exception as e:
            print(f"Warning: Archive creation failed: {e}")
            return archive_path


def create_publication_package(
    method_data: Dict[str, Dict[str, Any]],
    phylogenetic_kernel: np.ndarray,
    output_dir: Union[str, Path] = 'publication_outputs',
    **compiler_kwargs
) -> PublicationPackage:
    """Convenience function to create complete publication package.
    
    Args:
        method_data: Dict mapping method names to their evaluation data
        phylogenetic_kernel: Phylogenetic distance matrix
        output_dir: Directory for outputs
        **compiler_kwargs: Additional arguments for PublicationCompiler
    
    Returns:
        Complete PublicationPackage
    """
    compiler = PublicationCompiler(
        output_dir=output_dir,
        phylogenetic_kernel=phylogenetic_kernel,
        **compiler_kwargs
    )
    
    package = compiler.compile_method_comparison(method_data)
    
    # Generate validation report
    validation_report = compiler.generate_validation_report(package)
    print("\nValidation Report:")
    print(validation_report)
    
    return package