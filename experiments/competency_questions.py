"""
Competency Questions Experiment: SIDEKICK Ontology-Based Reasoning

This script demonstrates SIDEKICK's reasoning capabilities through a series of
competency questions (SPARQL queries) that leverage ontology hierarchies.

Includes:
- 7 local competency questions using HPO/MONDO hierarchies
- 1 federated query combining SIDEKICK + Ubergraph for anatomical reasoning

Usage:
    python competency_questions.py --sidekick-ttl Sidekick_v1.ttl \
                                   --hpo-obo hp.obo \
                                   --mondo-obo mondo.obo \
                                   --output-dir results/

Author: SIDEKICK Team
License: CC BY 4.0
"""

import argparse
import os
import sys
import time
import json
from collections import defaultdict

from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal
from tabulate import tabulate
import pandas as pd

try:
    import oxrdflib
    OXRDFLIB_AVAILABLE = True
except ImportError:
    OXRDFLIB_AVAILABLE = False
    print("Warning: oxrdflib not available. Using default RDFLib store (slower).")

try:
    import pronto
    PRONTO_AVAILABLE = True
except ImportError:
    PRONTO_AVAILABLE = False
    print("Warning: pronto not available. Ontology loading will be skipped.")


# ============================================================================
# NAMESPACES
# ============================================================================

SIO = Namespace("http://semanticscience.org/resource/")
SK = Namespace("http://sidekick.bio2vec.net/")
OBO = Namespace("http://purl.obolibrary.org/obo/")
RXNORM = Namespace("http://purl.bioontology.org/ontology/RXNORM/")


# ============================================================================
# GRAPH LOADING AND ONTOLOGY INTEGRATION
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
    
    print(f"✓ Loaded {len(g):,} triples in {elapsed:.2f} seconds")
    
    return g


def add_ontology_to_graph(g, obo_file, ontology_name):
    """
    Add ontology terms and hierarchy to the SIDEKICK graph.
    This enables hierarchical reasoning in SPARQL queries.
    """
    if not PRONTO_AVAILABLE:
        print(f"⚠️  Skipping {ontology_name} ontology loading (pronto not available)")
        return g
    
    print(f"\n{'='*70}")
    print(f"ADDING {ontology_name.upper()} ONTOLOGY")
    print(f"{'='*70}")
    print(f"File: {obo_file}")
    
    start_time = time.time()
    ont = pronto.Ontology(obo_file)
    triples_added = 0
    
    print(f"Processing {len(list(ont.terms()))} terms...")
    
    for term in ont.terms():
        if term.id is None:
            continue
        
        term_uri = URIRef(f"http://purl.obolibrary.org/obo/{term.id.replace(':', '_')}")
        
        # Add label
        if term.name:
            g.add((term_uri, RDFS.label, Literal(term.name)))
            triples_added += 1
        
        # Add hierarchy (direct parents only)
        for parent in term.superclasses(distance=1, with_self=False):
            if parent.id:
                parent_uri = URIRef(f"http://purl.obolibrary.org/obo/{parent.id.replace(':', '_')}")
                g.add((term_uri, RDFS.subClassOf, parent_uri))
                triples_added += 1
    
    elapsed = time.time() - start_time
    print(f"✓ Added {triples_added:,} triples in {elapsed:.2f} seconds")
    
    return g


# ============================================================================
# QUERY EXECUTION AND STATISTICS
# ============================================================================

def run_competency_query(g, query_name, query_sparql, explanation, 
                         show_limit=15, save_results=None):
    """
    Execute a competency question query and display results with analysis.
    
    Args:
        g: RDF graph
        query_name: Name of the competency question
        query_sparql: SPARQL query string
        explanation: Dict with reasoning type, relevance, MedDRA limitation
        show_limit: Number of results to display
        save_results: Path to save CSV results (optional)
    
    Returns:
        Dict with query results and statistics
    """
    print("\n" + "="*100)
    print(f"COMPETENCY QUESTION: {query_name}")
    print("="*100)
    print(f"Reasoning Type: {explanation['type']}")
    print(f"Clinical Relevance: {explanation['relevance']}")
    print(f"Why MedDRA Can't Do This: {explanation['meddra_limitation']}")
    print()
    
    query_header = """
    PREFIX sio: <http://semanticscience.org/resource/>
    PREFIX sk: <http://sidekick.bio2vec.net/>
    PREFIX obo: <http://purl.obolibrary.org/obo/>
    PREFIX rxnorm: <http://purl.bioontology.org/ontology/RXNORM/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    """
    
    try:
        start_time = time.time()
        results = list(g.query(query_header + query_sparql))
        elapsed = time.time() - start_time
        
        total_results = len(results)
        
        if total_results > 0:
            headers = [str(v) for v in results[0].labels]
            
            # Display sample results
            rows = []
            full_results = []
            
            for r in results[:show_limit]:
                clean_row = []
                full_row = {}
                
                for var in results[0].labels:
                    val = r[var]
                    full_row[str(var)] = str(val) if val else None
                    
                    if val is None:
                        clean_row.append("-")
                    elif isinstance(val, URIRef):
                        val_str = str(val)
                        val_str = val_str.replace("http://semanticscience.org/resource/", "sio:")
                        val_str = val_str.replace("http://sidekick.bio2vec.net/", "sk:")
                        val_str = val_str.replace("http://purl.obolibrary.org/obo/", "")
                        val_str = val_str.replace("http://purl.bioontology.org/ontology/RXNORM/", "rxnorm:")
                        clean_row.append(val_str)
                    else:
                        val_str = str(val)
                        if len(val_str) > 70:
                            val_str = val_str[:67] + "..."
                        clean_row.append(val_str)
                
                rows.append(clean_row)
                full_results.append(full_row)
            
            print(f"Query Results ({total_results:,} total, showing top {min(show_limit, total_results)}):")
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            print()
            
            # Calculate statistics
            stats = calculate_query_statistics(results, results[0].labels)
            print("Query Statistics:")
            for key, value in stats.items():
                print(f"  • {key}: {value:,}")
            print()
            
            print(f"⏱️  Query execution time: {elapsed:.3f} seconds")
            print(f" Query Status: SUCCESS")
            
            # Save results if requested
            if save_results:
                save_query_results(results, save_results, headers)
            
        else:
            print(" No results found.")
            print(f" Query Status: EXECUTED (no matches)")
            full_results = []
        
        print()
        
        return {
            'success': True,
            'total_results': total_results,
            'execution_time': elapsed,
            'sample_results': full_results[:5],
            'statistics': stats if total_results > 0 else {}
        }
        
    except Exception as e:
        print(f" Query Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'total_results': 0,
            'execution_time': 0
        }


def calculate_query_statistics(results, labels):
    """Calculate statistics from query results."""
    stats = {}
    
    # Count unique values in each column
    for label in labels:
        unique_values = set()
        for r in results:
            val = r[label]
            if val is not None:
                unique_values.add(str(val))
        stats[f"Unique {label}"] = len(unique_values)
    
    return stats


def save_query_results(results, filepath, headers):
    """Save query results to CSV file."""
    if not results:
        return
    
    data = []
    for r in results:
        row = {}
        for var in headers:
            val = r[var]
            row[var] = str(val) if val else None
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f" Results saved to: {filepath}")


# ============================================================================
# COMPETENCY QUESTIONS DEFINITIONS
# ============================================================================

# CQ1: Anatomical Reasoning - Cardiac-Related Side Effects
CQ1_CARDIAC_SIDE_EFFECTS = """
SELECT DISTINCT ?drug_name ?phenotype_label ?phenotype_id
WHERE {
    # Find the cardiac abnormality root term
    BIND(obo:HP_0001626 AS ?cardiac_root)  # Abnormality of the cardiovascular system
    
    # Find all phenotypes that are subclasses of cardiac abnormality
    ?phenotype_id rdfs:subClassOf* ?cardiac_root .
    ?phenotype_id rdfs:label ?phenotype_label .
    
    # Find drugs with side effects matching these phenotypes
    ?assoc a sk:SideEffect ;
           sk:refersToDrug ?collection ;
           sk:hasSideEffect ?phenotype_id .
    
    ?collection rdfs:label ?drug_name .
}
ORDER BY ?drug_name ?phenotype_label
"""

CQ1_EXPLANATION = {
    'type': 'Anatomical Reasoning via Ontology Hierarchy',
    'relevance': 'Identifies all drugs with ANY cardiovascular side effects by reasoning over HPO hierarchy. Critical for cardiac safety screening.',
    'meddra_limitation': 'MedDRA groups cardiac terms under "Cardiac disorders" SOC, but requires manual enumeration of specific PTs. Cannot automatically find all cardiac-related terms.'
}

# CQ2: Hierarchical Reasoning - Nervous System Effects
CQ2_NERVOUS_SYSTEM = """
SELECT DISTINCT ?drug_name ?phenotype_label ?phenotype_id
WHERE {
    # Abnormality of the nervous system
    BIND(obo:HP_0000707 AS ?neuro_root)
    
    # Find all nervous system phenotypes via transitive hierarchy
    ?phenotype_id rdfs:subClassOf* ?neuro_root .
    ?phenotype_id rdfs:label ?phenotype_label .
    
    # Find drugs with these side effects
    ?assoc a sk:SideEffect ;
           sk:refersToDrug ?collection ;
           sk:hasSideEffect ?phenotype_id .
    
    ?collection rdfs:label ?drug_name .
}
ORDER BY ?drug_name
"""

CQ2_EXPLANATION = {
    'type': 'Hierarchical Subsumption Reasoning',
    'relevance': 'Finds all drugs affecting the nervous system (CNS, PNS, autonomic) without enumerating specific symptoms. Essential for neurological drug safety.',
    'meddra_limitation': 'MedDRA has "Nervous system disorders" but organizes by symptom type rather than anatomical hierarchy. No automated way to get "all neurological effects".'
}

# CQ3: Cross-System Reasoning - Renal-Related Contraindications
CQ3_RENAL_CONTRAINDICATIONS = """
SELECT DISTINCT ?drug_name ?contraindication_type ?phenotype_label
WHERE {
    # Abnormality of the kidney
    BIND(obo:HP_0000077 AS ?kidney_root)
    
    # Find kidney-related phenotypes
    ?phenotype_id rdfs:subClassOf* ?kidney_root .
    ?phenotype_id rdfs:label ?phenotype_label .
    
    # Find drugs contraindicated in these conditions
    {
        # Phenotype contraindications
        ?assoc a sk:PhenotypeContraindication ;
               sk:refersToDrug ?collection ;
               sk:isContraindicatedInPhenotype ?phenotype_id .
        BIND("Phenotype" AS ?contraindication_type)
    }
    UNION
    {
        # Disease contraindications with kidney-related terms
        ?assoc a sk:DiseaseContraindication ;
               sk:refersToDrug ?collection ;
               sk:isContraindicatedInDisease ?disease_id .
        BIND("Disease" AS ?contraindication_type)
        ?disease_id rdfs:label ?phenotype_label .
        FILTER(CONTAINS(LCASE(?phenotype_label), "renal") || 
               CONTAINS(LCASE(?phenotype_label), "kidney"))
    }
    
    ?collection rdfs:label ?drug_name .
}
ORDER BY ?drug_name
"""

CQ3_EXPLANATION = {
    'type': 'Cross-System Anatomical and Disease Reasoning',
    'relevance': 'Identifies drugs contraindicated in ANY kidney-related condition. Crucial for prescribing safety in renal impairment.',
    'meddra_limitation': 'MedDRA separates "Renal and urinary disorders" from laboratory findings. Cannot automatically connect renal phenotypes with kidney diseases.'
}

# CQ4: Specific Anatomical Structure - Heart Rhythm Abnormalities
CQ4_CARDIAC_RHYTHM = """
SELECT DISTINCT ?drug_name ?phenotype_label ?phenotype_id
WHERE {
    # Arrhythmia (abnormality of cardiac rhythm)
    BIND(obo:HP_0011675 AS ?rhythm_root)
    
    # Find all rhythm-related phenotypes
    ?phenotype_id rdfs:subClassOf* ?rhythm_root .
    ?phenotype_id rdfs:label ?phenotype_label .
    
    # Find drugs with these side effects
    ?assoc a sk:SideEffect ;
           sk:refersToDrug ?collection ;
           sk:hasSideEffect ?phenotype_id .
    
    ?collection rdfs:label ?drug_name .
}
ORDER BY ?drug_name
"""

CQ4_EXPLANATION = {
    'type': 'Fine-Grained Anatomical/Physiological Reasoning',
    'relevance': 'Finds drugs affecting cardiac rhythm specifically, distinguishing from other cardiac effects. Critical for QT prolongation screening.',
    'meddra_limitation': 'MedDRA lists "Tachycardia", "Bradycardia", "Arrhythmia" as separate PTs. Cannot automatically recognize these as related rhythm disturbances.'
}

# CQ5: Multi-Level Hierarchy - Metabolic Abnormalities
CQ5_METABOLIC = """
SELECT DISTINCT ?drug_name ?phenotype_label ?phenotype_id
WHERE {
    # Abnormality of metabolism/homeostasis
    BIND(obo:HP_0001939 AS ?metabolic_root)
    
    # Find metabolic phenotypes
    ?phenotype_id rdfs:subClassOf* ?metabolic_root .
    ?phenotype_id rdfs:label ?phenotype_label .
    
    # Find drugs
    ?assoc a sk:SideEffect ;
           sk:refersToDrug ?collection ;
           sk:hasSideEffect ?phenotype_id .
    
    ?collection rdfs:label ?drug_name .
}
ORDER BY ?drug_name ?phenotype_label
"""

CQ5_EXPLANATION = {
    'type': 'Multi-Level Hierarchical Reasoning',
    'relevance': 'Identifies drugs with metabolic side effects at any level of granularity (from general "metabolic disorder" to specific "hyperglycemia").',
    'meddra_limitation': 'MedDRA has "Metabolism and nutrition disorders" SOC but flat structure within. Cannot reason about relationships between terms.'
}

# CQ6: Disease Indication Hierarchy - Infectious Diseases
CQ6_INFECTIOUS_DISEASE_INDICATIONS = """
SELECT DISTINCT ?drug_name ?disease_label ?disease_id
WHERE {
    # Infectious disease (MONDO)
    BIND(obo:MONDO_0005550 AS ?infectious_root)
    
    # Find all infectious diseases via hierarchy
    ?disease_id rdfs:subClassOf* ?infectious_root .
    ?disease_id rdfs:label ?disease_label .
    
    # Find drugs indicated for these diseases
    ?assoc a sk:DiseaseIndication ;
           sk:refersToDrug ?collection ;
           sk:isIndicatedForDisease ?disease_id .
    
    ?collection rdfs:label ?drug_name .
}
ORDER BY ?drug_name
"""

CQ6_EXPLANATION = {
    'type': 'Disease Taxonomy Reasoning via MONDO',
    'relevance': 'Finds all antimicrobial/antiviral drugs by reasoning over disease hierarchy. Useful for infectious disease treatment analysis.',
    'meddra_limitation': 'MedDRA is for adverse events, not indications. Cannot automatically group bacterial, viral, fungal infections as "infectious diseases".'
}

# CQ7: Combined Reasoning - Drugs Safe for Cardiac Patients
CQ7_CARDIAC_SAFE_DRUGS = """
SELECT DISTINCT ?drug_name 
       (COUNT(DISTINCT ?cardiac_side_effect) AS ?cardiac_side_effect_count)
       (COUNT(DISTINCT ?cardiac_contraindication) AS ?cardiac_contraindication_count)
WHERE {
    ?collection a sk:DrugCollection ;
                rdfs:label ?drug_name .
    
    # Count cardiac-related side effects
    OPTIONAL {
        BIND(obo:HP_0001626 AS ?cardiac_root)
        ?cardiac_phenotype rdfs:subClassOf* ?cardiac_root .
        
        ?se_assoc a sk:SideEffect ;
                  sk:refersToDrug ?collection ;
                  sk:hasSideEffect ?cardiac_phenotype .
        BIND(?cardiac_phenotype AS ?cardiac_side_effect)
    }
    
    # Count cardiac-related contraindications
    OPTIONAL {
        ?contra_assoc a sk:PhenotypeContraindication ;
                      sk:refersToDrug ?collection ;
                      sk:isContraindicatedInPhenotype ?cardiac_phenotype_contra .
        
        ?cardiac_phenotype_contra rdfs:subClassOf* obo:HP_0001626 .
        BIND(?cardiac_phenotype_contra AS ?cardiac_contraindication)
    }
}
GROUP BY ?drug_name
HAVING (?cardiac_side_effect_count = 0 && ?cardiac_contraindication_count = 0)
ORDER BY ?drug_name
LIMIT 50
"""

CQ7_EXPLANATION = {
    'type': 'Combined Safety Reasoning (Side Effects + Contraindications)',
    'relevance': 'Identifies drugs with NO cardiac side effects or contraindications. Useful for prescribing in cardiac patients.',
    'meddra_limitation': 'Would require manually enumerating ALL cardiac-related MedDRA PTs for both adverse events and contraindications.'
}

# CQ8: Federated Query - Anatomical Parts of Heart
CQ8_FEDERATED_HEART_ANATOMY = """
SELECT DISTINCT ?drug_name ?phenotype_label ?heart_part_label
WHERE {
    # Local query: Find drugs and their side effects
    ?assoc a sk:SideEffect ;
           sk:refersToDrug ?collection ;
           sk:hasSideEffect ?phenotype_id .
    
    ?collection rdfs:label ?drug_name .
    ?phenotype_id rdfs:label ?phenotype_label .
    
    # Federated query to Ubergraph: Find anatomical parts of heart
    SERVICE <https://ubergraph.apps.renci.org/sparql> {
        # What anatomical entity does this phenotype affect?
        ?phenotype_id obo:UPHENO_0000001 ?affected_entity .
        
        # Is that entity part of the heart (transitively)?
        ?affected_entity obo:BFO_0000050* obo:UBERON_0000948 .
        
        # Get the anatomical part label
        ?affected_entity rdfs:label ?heart_part_label .
        
        # Filter out the heart itself
        FILTER(?affected_entity != obo:UBERON_0000948)
    }
}
ORDER BY ?drug_name ?heart_part_label
"""

CQ8_EXPLANATION = {
    'type': 'Federated Axiomatic Reasoning Across Knowledge Graphs',
    'relevance': 'Identifies drugs affecting specific anatomical parts of the heart (valves, ventricles, etc.) using federated SPARQL and HPO axioms.',
    'meddra_limitation': 'MedDRA has no anatomical axioms. Cannot reason about which specific heart structures are affected by "cardiac disorders" without manual annotation.'
}


# ============================================================================
# SUMMARY REPORTING
# ============================================================================

def generate_competency_report(results_dict, output_dir):
    """Generate and save summary report of all competency questions."""
    print("\n" + "="*100)
    print("COMPETENCY QUESTIONS SUMMARY REPORT")
    print("="*100)
    
    summary_data = []
    for cq_name, result in results_dict.items():
        summary_data.append([
            cq_name,
            " Yes" if result['success'] else " No",
            result.get('total_results', 0),
            f"{result.get('execution_time', 0):.3f}s" if result['success'] else "N/A"
        ])
    
    print(tabulate(summary_data, 
                   headers=["Competency Question", "Success", "Results Found", "Query Time"],
                   tablefmt="grid"))
    print()
    
    # Overall statistics
    total_cqs = len(results_dict)
    successful_cqs = sum(1 for r in results_dict.values() if r['success'])
    total_results = sum(r.get('total_results', 0) for r in results_dict.values())
    avg_time = sum(r.get('execution_time', 0) for r in results_dict.values()) / total_cqs
    
    print(f"Overall Statistics:")
    print(f"  • Total Competency Questions: {total_cqs}")
    print(f"  • Successfully Executed: {successful_cqs}/{total_cqs} ({100*successful_cqs/total_cqs:.1f}%)")
    print(f"  • Total Results Found: {total_results:,}")
    print(f"  • Average Query Time: {avg_time:.3f}s")
    print()
    
    # Save summary to file
    summary_df = pd.DataFrame(summary_data, 
                             columns=["Competency Question", "Success", "Results Found", "Query Time"])
    summary_path = os.path.join(output_dir, 'competency_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"📊 Summary saved to: {summary_path}")
    
    # Save detailed results as JSON
    json_path = os.path.join(output_dir, 'competency_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"📊 Detailed results saved to: {json_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """Main pipeline for competency questions experiment."""
    print("\n" + "="*100)
    print("SIDEKICK COMPETENCY QUESTIONS EVALUATION")
    print("Demonstrating Ontology-Based Reasoning Capabilities")
    print("="*100)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, 'queries')
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load SIDEKICK graph
    g = load_sidekick_graph(args.sidekick_ttl)
    
    # Step 2: Add ontologies for hierarchical reasoning
    if args.hpo_obo:
        g = add_ontology_to_graph(g, args.hpo_obo, "HPO")
    
    if args.mondo_obo:
        g = add_ontology_to_graph(g, args.mondo_obo, "MONDO")
    
    # Step 3: Execute all competency questions
    print("\n" + "="*100)
    print("EXECUTING COMPETENCY QUESTIONS")
    print("="*100)
    
    results = {}
    
    # CQ1: Cardiac Side Effects
    results['CQ1: Cardiac Side Effects'] = run_competency_query(
        g, "Find all drugs with cardiovascular-related side effects",
        CQ1_CARDIAC_SIDE_EFFECTS, CQ1_EXPLANATION,
        show_limit=15,
        save_results=os.path.join(results_dir, 'cq1_cardiac_side_effects.csv')
    )
    
    # CQ2: Nervous System
    results['CQ2: Nervous System Effects'] = run_competency_query(
        g, "Find all drugs with nervous system side effects",
        CQ2_NERVOUS_SYSTEM, CQ2_EXPLANATION,
        show_limit=15,
        save_results=os.path.join(results_dir, 'cq2_nervous_system.csv')
    )
    
    # CQ3: Renal Contraindications
    results['CQ3: Renal Contraindications'] = run_competency_query(
        g, "Find drugs contraindicated in kidney-related conditions",
        CQ3_RENAL_CONTRAINDICATIONS, CQ3_EXPLANATION,
        show_limit=15,
        save_results=os.path.join(results_dir, 'cq3_renal_contraindications.csv')
    )
    
    # CQ4: Cardiac Rhythm
    results['CQ4: Cardiac Rhythm Abnormalities'] = run_competency_query(
        g, "Find drugs affecting cardiac rhythm specifically",
        CQ4_CARDIAC_RHYTHM, CQ4_EXPLANATION,
        show_limit=15,
        save_results=os.path.join(results_dir, 'cq4_cardiac_rhythm.csv')
    )
    
    # CQ5: Metabolic Effects
    results['CQ5: Metabolic Abnormalities'] = run_competency_query(
        g, "Find drugs with metabolic side effects",
        CQ5_METABOLIC, CQ5_EXPLANATION,
        show_limit=15,
        save_results=os.path.join(results_dir, 'cq5_metabolic.csv')
    )
    
    # CQ6: Infectious Disease Indications
    results['CQ6: Infectious Disease Indications'] = run_competency_query(
        g, "Find drugs indicated for infectious diseases",
        CQ6_INFECTIOUS_DISEASE_INDICATIONS, CQ6_EXPLANATION,
        show_limit=15,
        save_results=os.path.join(results_dir, 'cq6_infectious_diseases.csv')
    )
    
    # CQ7: Cardiac-Safe Drugs
    results['CQ7: Cardiac-Safe Drugs'] = run_competency_query(
        g, "Find drugs with no cardiac side effects or contraindications",
        CQ7_CARDIAC_SAFE_DRUGS, CQ7_EXPLANATION,
        show_limit=50,
        save_results=os.path.join(results_dir, 'cq7_cardiac_safe.csv')
    )
    
    # CQ8: Federated Query (optional - may timeout)
    if args.run_federated:
        print("\n⚠️  Federated query enabled. This may take 60+ seconds...")
        results['CQ8: Federated Heart Anatomy'] = run_competency_query(
            g, "Find drugs affecting specific anatomical parts of heart (federated)",
            CQ8_FEDERATED_HEART_ANATOMY, CQ8_EXPLANATION,
            show_limit=50,
            save_results=os.path.join(results_dir, 'cq8_federated_heart.csv')
        )
    else:
        print("\n Skipping federated query (use --run-federated to enable)")
    
    # Step 4: Generate summary report
    generate_competency_report(results, args.output_dir)
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE")
    print("="*100)
    print(f"\nResults saved to: {args.output_dir}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Competency Questions Experiment: Demonstrate SIDEKICK reasoning capabilities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--sidekick-ttl',
        type=str,
        required=True,
        help='Path to SIDEKICK RDF graph (Sidekick_v1.ttl)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--hpo-obo',
        type=str,
        default=None,
        help='Path to HPO ontology (hp.obo) for hierarchical reasoning'
    )
    
    parser.add_argument(
        '--mondo-obo',
        type=str,
        default=None,
        help='Path to MONDO ontology (mondo.obo) for disease hierarchy reasoning'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/competency_questions',
        help='Output directory for results and summaries'
    )
    
    parser.add_argument(
        '--run-federated',
        action='store_true',
        help='Run federated query to Ubergraph (CQ8) - may be slow'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate input files
    if not os.path.exists(args.sidekick_ttl):
        print(f"Error: SIDEKICK TTL file not found: {args.sidekick_ttl}")
        sys.exit(1)
    
    if args.hpo_obo and not os.path.exists(args.hpo_obo):
        print(f"Warning: HPO OBO file not found: {args.hpo_obo}")
        print("Hierarchical reasoning may be limited without ontologies.")
    
    if args.mondo_obo and not os.path.exists(args.mondo_obo):
        print(f"Warning: MONDO OBO file not found: {args.mondo_obo}")
        print("Disease hierarchy reasoning may be limited.")
    
    # Run experiment
    main(args)
