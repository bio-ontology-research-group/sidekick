"""
ShEx Schema Validation Experiment: SIDEKICK Knowledge Graph

This script validates the SIDEKICK RDF knowledge graph against its ShEx schema
to ensure structural integrity, completeness, and adherence to the data model.

Validates 10 entity types:
- Drug Collections
- Pharmaceutical Products
- SPL Documents
- 7 Clinical Association types (Side Effects, Indications, Contraindications)

Usage:
    python shex_validation.py --sidekick-ttl Sidekick_v1.ttl \
                              --shex-schema sidekick_validation.shex \
                              --output-dir results/

Author: SIDEKICK Team
License: CC BY 4.0
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import pandas as pd
from rdflib import Graph, Namespace, RDF
from tqdm import tqdm

try:
    import oxrdflib
    OXRDFLIB_AVAILABLE = True
except ImportError:
    OXRDFLIB_AVAILABLE = False
    print("Warning: oxrdflib not available. Using default RDFLib store (slower).")

try:
    from pyshex.shex_evaluator import ShExEvaluator
    PYSHEX_AVAILABLE = True
except ImportError:
    PYSHEX_AVAILABLE = False
    print("Error: pyshex not available. Install with: pip install pyshex")
    sys.exit(1)


# ============================================================================
# NAMESPACES
# ============================================================================

SIO = Namespace("http://semanticscience.org/resource/")
SK = Namespace("http://sidekick.bio2vec.net/")
OBO = Namespace("http://purl.obolibrary.org/obo/")
RXNORM = Namespace("http://purl.bioontology.org/ontology/RXNORM/")


# ============================================================================
# GRAPH LOADING
# ============================================================================

def load_sidekick_graph(filename):
    """
    Load the SIDEKICK RDF graph.
    Uses Oxigraph store if available for better performance.
    """
    print("\n" + "="*70)
    print("LOADING SIDEKICK KNOWLEDGE GRAPH")
    print("="*70)
    print(f"File: {filename}")
    
    start_time = time.time()
    
    if OXRDFLIB_AVAILABLE:
        print("Using Oxigraph store for optimal performance...")
        g = Graph(store="Oxigraph")
    else:
        print("Using default RDFLib store...")
        g = Graph()
    
    g.parse(filename, format='turtle')
    elapsed = time.time() - start_time
    
    print(f" Loaded {len(g):,} triples in {elapsed:.2f} seconds")
    
    return g


# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Define all shape validations: (shape_name, node_type, description)
VALIDATION_SHAPES = [
    ("DrugCollectionShape", SK.DrugCollection, "Drug Collections"),
    ("PharmaceuticalProductShape", SIO.SIO_010039, "Pharmaceutical Products"),
    ("SPLDocumentShape", SIO.SIO_000148, "SPL Documents"),
    ("SideEffectShape", SK.SideEffect, "Side Effects"),
    ("DiseaseIndicationShape", SK.DiseaseIndication, "Disease Indications"),
    ("PhenotypeIndicationShape", SK.PhenotypeIndication, "Phenotype Indications"),
    ("DrugIndicationShape", SK.DrugIndication, "Drug Indications"),
    ("DiseaseContraindicationShape", SK.DiseaseContraindication, "Disease Contraindications"),
    ("PhenotypeContraindicationShape", SK.PhenotypeContraindication, "Phenotype Contraindications"),
    ("DrugContraindicationShape", SK.DrugContraindication, "Drug Contraindications"),
]


# ============================================================================
# VALIDATION EXECUTION
# ============================================================================

def validate_shape(g, shex_file, shape_name, node_type, description, 
                   sample_size=None, verbose=False):
    """
    Validate all nodes of a given type against a ShEx shape.
    
    Args:
        g: RDF graph
        shex_file: Path to ShEx schema file
        shape_name: Name of the shape in the schema
        node_type: RDF type to validate
        description: Human-readable description
        sample_size: Maximum number of nodes to validate (None = all)
        verbose: Show detailed error messages
    
    Returns:
        Dict with validation results
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING: {description} ({shape_name})")
    print(f"{'='*80}")
    
    # Get all nodes of this type
    print(f"Querying for nodes of type {node_type}...")
    nodes = list(g.subjects(predicate=RDF.type, object=node_type))
    total_nodes = len(nodes)
    
    if total_nodes == 0:
        print(f"  No nodes found of type {node_type}")
        return {
            'shape_name': shape_name,
            'description': description,
            'status': 'NO_NODES',
            'total_nodes': 0,
            'validated': 0,
            'passed': 0,
            'failed': 0,
            'errors': []
        }
    
    print(f" Found {total_nodes:,} nodes")
    
    # Determine sample size
    if sample_size is None or sample_size > total_nodes:
        sample_size = total_nodes
    
    if sample_size < total_nodes:
        print(f"Validating sample of {sample_size:,} nodes...")
    else:
        print(f"Validating all {sample_size:,} nodes...")
    
    # Validate nodes
    passed = 0
    failed = 0
    errors = []
    
    shape_uri = f"http://sidekick.bio2vec.net/{shape_name}"
    
    start_time = time.time()
    
    for node in tqdm(nodes[:sample_size], desc="Validating", unit="nodes"):
        try:
            evaluator = ShExEvaluator(
                rdf=g,
                schema=shex_file,
                focus=str(node),
                start=shape_uri
            )
            
            result = evaluator.evaluate()
            
            # Check if validation passed
            if result and len(result) > 0 and result[0].result:
                passed += 1
            else:
                failed += 1
                if len(errors) < 10 or verbose:  # Store more errors in verbose mode
                    reason = result[0].reason if result and len(result) > 0 else "Unknown validation failure"
                    errors.append({
                        'node': str(node),
                        'reason': reason
                    })
        
        except Exception as e:
            failed += 1
            if len(errors) < 10 or verbose:
                errors.append({
                    'node': str(node),
                    'reason': f"Exception: {str(e)}"
                })
    
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\nValidation Results:")
    print(f"  Total nodes in graph: {total_nodes:,}")
    print(f"  Nodes validated: {sample_size:,}")
    print(f"   Passed: {passed:,} ({100*passed/sample_size:.1f}%)")
    print(f"   Failed: {failed:,} ({100*failed/sample_size:.1f}%)")
    print(f"  Time: {elapsed:.2f}s ({sample_size/elapsed:.0f} nodes/sec)")
    
    # Show sample errors
    if errors:
        print(f"\nSample Errors (showing {min(3, len(errors))} of {len(errors)}):")
        for i, error in enumerate(errors[:3]):
            node_short = error['node'][:60] + "..." if len(error['node']) > 60 else error['node']
            reason_short = error['reason'][:100] + "..." if len(error['reason']) > 100 else error['reason']
            print(f"  {i+1}. {node_short}")
            print(f"     → {reason_short}")
    
    # Determine overall status
    if failed == 0:
        status = "PASSED"
        print(f"\n VALIDATION PASSED")
    else:
        status = "FAILED"
        print(f"\n VALIDATION FAILED")
    
    return {
        'shape_name': shape_name,
        'description': description,
        'status': status,
        'total_nodes': total_nodes,
        'validated': sample_size,
        'passed': passed,
        'failed': failed,
        'pass_rate': passed / sample_size if sample_size > 0 else 0,
        'time': elapsed,
        'nodes_per_sec': sample_size / elapsed if elapsed > 0 else 0,
        'errors': errors
    }


# ============================================================================
# VALIDATION SUMMARY AND REPORTING
# ============================================================================

def print_validation_summary(results):
    """Print comprehensive validation summary."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY REPORT")
    print("="*80)
    
    # Summary table
    summary_data = []
    for result in results:
        status_symbol = {
            'PASSED': '',
            'FAILED': '',
            'NO_NODES': ''
        }.get(result['status'], '')
        
        summary_data.append([
            status_symbol,
            result['description'],
            result['total_nodes'],
            result['validated'],
            result['passed'],
            result['failed'],
            f"{result.get('pass_rate', 0)*100:.1f}%"
        ])
    
    print("\n" + pd.DataFrame(
        summary_data,
        columns=['Status', 'Entity Type', 'Total', 'Validated', 'Passed', 'Failed', 'Pass Rate']
    ).to_string(index=False))
    
    # Overall statistics
    total_nodes = sum(r['total_nodes'] for r in results)
    total_validated = sum(r['validated'] for r in results)
    total_passed = sum(r['passed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total nodes in graph: {total_nodes:,}")
    print(f"Total nodes validated: {total_validated:,}")
    print(f"Total passed: {total_passed:,} ({100*total_passed/total_validated:.1f}%)")
    print(f"Total failed: {total_failed:,} ({100*total_failed/total_validated:.1f}%)")
    
    # Shape-level summary
    passed_shapes = sum(1 for r in results if r['status'] == 'PASSED')
    failed_shapes = sum(1 for r in results if r['status'] == 'FAILED')
    no_node_shapes = sum(1 for r in results if r['status'] == 'NO_NODES')
    
    print(f"\nShape Validation:")
    print(f"   Passed shapes: {passed_shapes}/{len(results)}")
    print(f"   Failed shapes: {failed_shapes}/{len(results)}")
    print(f"    Shapes with no nodes: {no_node_shapes}/{len(results)}")
    
    if failed_shapes == 0 and no_node_shapes == 0:
        print(f"\n ALL VALIDATIONS PASSED! SIDEKICK is fully compliant with the schema.")
    elif failed_shapes > 0:
        print(f"\n  Some validations failed. Review errors above for details.")


def save_validation_results(results, output_dir):
    """Save validation results to files."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary CSV
    summary_data = []
    for result in results:
        summary_data.append({
            'Shape': result['shape_name'],
            'Description': result['description'],
            'Status': result['status'],
            'Total Nodes': result['total_nodes'],
            'Validated': result['validated'],
            'Passed': result['passed'],
            'Failed': result['failed'],
            'Pass Rate': f"{result.get('pass_rate', 0)*100:.2f}%",
            'Time (s)': f"{result.get('time', 0):.2f}",
            'Nodes/sec': f"{result.get('nodes_per_sec', 0):.0f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'validation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Save detailed errors for each shape
    errors_dir = os.path.join(output_dir, 'errors')
    os.makedirs(errors_dir, exist_ok=True)
    
    for result in results:
        if result['errors']:
            errors_df = pd.DataFrame(result['errors'])
            errors_path = os.path.join(errors_dir, f"{result['shape_name']}_errors.csv")
            errors_df.to_csv(errors_path, index=False)
            print(f"✓ Errors for {result['shape_name']} saved to: {errors_path}")
    
    # Save overall report as JSON
    import json
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed report saved to: {report_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """Main pipeline for ShEx validation experiment."""
    print("\n" + "="*80)
    print("SIDEKICK SHEX SCHEMA VALIDATION")
    print("Ensuring Structural Integrity and Completeness")
    print("="*80)
    
    # Load graph
    g = load_sidekick_graph(args.sidekick_ttl)
    
    # Run validations
    print("\n" + "="*80)
    print("RUNNING VALIDATIONS")
    print("="*80)
    print(f"ShEx Schema: {args.shex_schema}")
    print(f"Sample Size: {args.sample_size if args.sample_size else 'All nodes'}")
    print(f"Verbose: {args.verbose}")
    
    results = []
    
    for shape_name, node_type, description in VALIDATION_SHAPES:
        result = validate_shape(
            g=g,
            shex_file=args.shex_schema,
            shape_name=shape_name,
            node_type=node_type,
            description=description,
            sample_size=args.sample_size,
            verbose=args.verbose
        )
        results.append(result)
    
    # Print summary
    print_validation_summary(results)
    
    # Save results
    save_validation_results(results, args.output_dir)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    
    # Return exit code based on validation results
    failed_shapes = sum(1 for r in results if r['status'] == 'FAILED')
    if failed_shapes > 0:
        print(f"\n⚠️  {failed_shapes} shape(s) failed validation")
        return 1
    else:
        print(f"\n All shapes passed validation")
        return 0


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ShEx Schema Validation: Validate SIDEKICK knowledge graph structure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--sidekick-ttl',
        type=str,
        required=True,
        help='Path to SIDEKICK RDF graph (Sidekick_v1.ttl)'
    )
    
    parser.add_argument(
        '--shex-schema',
        type=str,
        required=True,
        help='Path to ShEx schema file (sidekick_validation.shex)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/shex_validation',
        help='Output directory for validation results'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Maximum number of nodes to validate per shape (default: all)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed error messages'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate input files
    if not os.path.exists(args.sidekick_ttl):
        print(f"Error: SIDEKICK TTL file not found: {args.sidekick_ttl}")
        sys.exit(1)
    
    if not os.path.exists(args.shex_schema):
        print(f"Error: ShEx schema file not found: {args.shex_schema}")
        sys.exit(1)
    
    # Run validation
    exit_code = main(args)
    sys.exit(exit_code)
