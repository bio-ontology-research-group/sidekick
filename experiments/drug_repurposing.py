"""
Drug Repurposing Experiment: SIDEKICK (HPO) vs OnSIDES (MedDRA)

This script evaluates SIDEKICK's performance on drug target prediction using 
side effect similarity (BMA + Resnik semantic similarity on HPO ontology).

Compares against published OnSIDES results (hardcoded values).

Usage:
    python drug_repurposing.py --sidekick-csv side_effects_mapped.csv \
                                --drugbank-xml full_database.xml \
                                --hpo-obo hp.obo \
                                --output-dir results/

Author: SIDEKICK Team
License: CC BY 4.0
"""

import argparse
import os
import sys
import pickle
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tqdm import tqdm
import xml.etree.ElementTree as ET


# ============================================================================
# PART 1: HPO ONTOLOGY LOADING AND HIERARCHY CONSTRUCTION
# ============================================================================

def parse_obo_file(obo_file):
    """
    Parse HPO OBO file to extract terms and relationships.
    
    Returns:
        - terms_dict: {hpo_id: {'name': str, 'is_a': [parent_ids]}}
    """
    print(f"\nParsing HPO ontology from: {obo_file}")
    
    terms_dict = {}
    current_term = None
    
    with open(obo_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line == "[Term]":
                if current_term and 'id' in current_term:
                    terms_dict[current_term['id']] = current_term
                current_term = {'is_a': []}
            
            elif line.startswith("id:") and current_term is not None:
                current_term['id'] = line.split("id:")[1].strip()
            
            elif line.startswith("name:") and current_term is not None:
                current_term['name'] = line.split("name:")[1].strip()
            
            elif line.startswith("is_a:") and current_term is not None:
                parent = line.split("is_a:")[1].split("!")[0].strip()
                current_term['is_a'].append(parent)
    
    # Add last term
    if current_term and 'id' in current_term:
        terms_dict[current_term['id']] = current_term
    
    print(f"Loaded {len(terms_dict)} HPO terms")
    
    return terms_dict


def build_hpo_graph(terms_dict):
    """
    Build NetworkX directed graph from HPO terms.
    
    Returns:
        - G: NetworkX DiGraph (child -> parent edges)
    """
    print("\nBuilding HPO hierarchy graph...")
    
    G = nx.DiGraph()
    
    # Add all nodes
    for hpo_id, term_data in terms_dict.items():
        G.add_node(hpo_id, name=term_data.get('name', ''))
    
    # Add edges (child -> parent)
    edge_count = 0
    for hpo_id, term_data in terms_dict.items():
        for parent_id in term_data.get('is_a', []):
            if parent_id in G:
                G.add_edge(hpo_id, parent_id)
                edge_count += 1
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Check connectivity
    if not nx.is_weakly_connected(G):
        n_components = nx.number_weakly_connected_components(G)
        print(f"Warning: Graph has {n_components} disconnected components")
    else:
        print("✓ Graph is fully connected")
    
    return G


# ============================================================================
# PART 2: INFORMATION CONTENT CALCULATION
# ============================================================================

def compute_hpo_ic_from_data(drug_hpo_dict, hpo_graph, cache_file=None):
    """
    Compute Information Content (IC) for HPO terms based on drug annotations.
    
    IC = -log(P(term)), where P(term) = frequency of term in dataset
    Propagates counts up the hierarchy (children -> parents).
    
    Args:
        drug_hpo_dict: {drug_name: set of HPO IDs}
        hpo_graph: NetworkX DiGraph
        cache_file: Optional pickle cache file
    
    Returns:
        ic_values: {hpo_id: IC value}
    """
    # Check cache
    if cache_file and os.path.exists(cache_file):
        print(f"\nLoading cached HPO IC values from {cache_file}...")
        with open(cache_file, 'rb') as f:
            ic_values = pickle.load(f)
        print(f"Loaded IC values for {len(ic_values)} terms")
        return ic_values
    
    print("\nComputing HPO Information Content...")
    
    # Count term occurrences (with ancestor propagation)
    term_counts = defaultdict(int)
    
    print("Propagating term counts up the hierarchy...")
    for drug, hpo_ids in tqdm(drug_hpo_dict.items(), desc="Processing drugs"):
        seen_in_drug = set()
        
        for hpo_id in hpo_ids:
            if hpo_id in hpo_graph:
                # Add term and all ancestors
                try:
                    ancestors = nx.ancestors(hpo_graph, hpo_id)
                    for term in [hpo_id] + list(ancestors):
                        seen_in_drug.add(term)
                except nx.NetworkXError:
                    seen_in_drug.add(hpo_id)
        
        # Count each unique term once per drug
        for term in seen_in_drug:
            term_counts[term] += 1
    
    # Calculate IC values
    total_drugs = len(drug_hpo_dict)
    ic_values = {}
    
    print("Calculating IC values...")
    for term in hpo_graph.nodes():
        count = term_counts.get(term, 0)
        if count > 0:
            probability = count / total_drugs
            ic_values[term] = -np.log(probability)
        else:
            ic_values[term] = 0.0
    
    # Statistics
    ic_vals = [v for v in ic_values.values() if v > 0]
    print(f"\nIC Statistics:")
    print(f"  Terms with IC > 0: {len(ic_vals)}")
    print(f"  Mean IC: {np.mean(ic_vals):.4f}")
    print(f"  Median IC: {np.median(ic_vals):.4f}")
    print(f"  Max IC: {np.max(ic_vals):.4f}")
    
    # Cache results
    if cache_file:
        print(f"\nCaching IC values to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(ic_values, f)
        print("✓ Cached successfully")
    
    return ic_values


# ============================================================================
# PART 3: PRECOMPUTE ANCESTORS AND SIMILARITIES
# ============================================================================

def precompute_hpo_ancestors(hpo_graph, cache_file=None):
    """
    Pre-compute ancestors for all HPO terms.
    
    Returns:
        ancestors_cache: {hpo_id: set of ancestor IDs (including self)}
    """
    # Check cache
    if cache_file and os.path.exists(cache_file):
        print(f"\nLoading cached HPO ancestors from {cache_file}...")
        with open(cache_file, 'rb') as f:
            ancestors_cache = pickle.load(f)
        print(f"Loaded ancestors for {len(ancestors_cache)} terms")
        return ancestors_cache
    
    print("\nPre-computing HPO ancestors...")
    ancestors_cache = {}
    
    for term in tqdm(hpo_graph.nodes(), desc="Computing ancestors"):
        try:
            # Include term itself plus all ancestors
            ancestors_cache[term] = set(nx.ancestors(hpo_graph, term)) | {term}
        except nx.NetworkXError:
            ancestors_cache[term] = {term}
    
    print(f"Cached ancestors for {len(ancestors_cache)} terms")
    
    # Cache results
    if cache_file:
        print(f"\nCaching to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(ancestors_cache, f)
        print("✓ Cached successfully")
    
    return ancestors_cache


def precompute_term_similarities(unique_hpo_ids, hpo_graph, ic_values, 
                                 ancestors_cache, cache_file=None):
    """
    Pre-compute Resnik similarities for all pairs of HPO terms.
    
    Resnik similarity = IC of Most Informative Common Ancestor (MICA)
    
    Returns:
        similarity_cache: {(hpo_id1, hpo_id2): similarity_score}
    """
    # Check cache
    if cache_file and os.path.exists(cache_file):
        print(f"\nLoading cached HPO term similarities from {cache_file}...")
        with open(cache_file, 'rb') as f:
            similarity_cache = pickle.load(f)
        print(f"Loaded {len(similarity_cache)} cached similarities")
        return similarity_cache
    
    print("\nPre-computing HPO term similarities...")
    print(f"Unique HPO terms: {len(unique_hpo_ids)}")
    print(f"Total pairs: {len(unique_hpo_ids) * (len(unique_hpo_ids) + 1) // 2}")
    
    similarity_cache = {}
    unique_hpo_list = list(unique_hpo_ids)
    
    for i in tqdm(range(len(unique_hpo_list)), desc="Computing similarities"):
        hpo1 = unique_hpo_list[i]
        
        if hpo1 not in hpo_graph:
            continue
        
        for j in range(i, len(unique_hpo_list)):
            hpo2 = unique_hpo_list[j]
            
            if hpo2 not in hpo_graph:
                continue
            
            # Same term
            if hpo1 == hpo2:
                similarity = ic_values.get(hpo1, 0.0)
            else:
                # Find common ancestors
                ancestors1 = ancestors_cache.get(hpo1, set())
                ancestors2 = ancestors_cache.get(hpo2, set())
                common_ancestors = ancestors1 & ancestors2
                
                if not common_ancestors:
                    similarity = 0.0
                else:
                    # Resnik: IC of Most Informative Common Ancestor (MICA)
                    similarity = max([ic_values.get(anc, 0.0) for anc in common_ancestors])
            
            # Store both directions
            similarity_cache[(hpo1, hpo2)] = similarity
            similarity_cache[(hpo2, hpo1)] = similarity
    
    print(f"Cached {len(similarity_cache)} term-term similarities")
    
    # Cache results
    if cache_file:
        print(f"\nCaching to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(similarity_cache, f)
        print("✓ Cached successfully")
    
    return similarity_cache


# ============================================================================
# PART 4: BMA SIMILARITY CALCULATION
# ============================================================================

def bma_similarity(drug1_hpo, drug2_hpo, similarity_cache):
    """
    Calculate Best Match Average (BMA) similarity between two drugs.
    
    BMA(D1, D2) = 0.5 * (avg_max(D1->D2) + avg_max(D2->D1))
    
    Where avg_max(D1->D2) = average of best matches for each term in D1 to D2
    
    Args:
        drug1_hpo: set of HPO IDs for drug 1
        drug2_hpo: set of HPO IDs for drug 2
        similarity_cache: precomputed term similarities
    
    Returns:
        BMA similarity score
    """
    if len(drug1_hpo) == 0 or len(drug2_hpo) == 0:
        return 0.0
    
    # Direction 1: For each term in drug1, find best match in drug2
    scores_1_to_2 = []
    for hpo1 in drug1_hpo:
        best_score = max([similarity_cache.get((hpo1, hpo2), 0.0) for hpo2 in drug2_hpo])
        scores_1_to_2.append(best_score)
    
    # Direction 2: For each term in drug2, find best match in drug1
    scores_2_to_1 = []
    for hpo2 in drug2_hpo:
        best_score = max([similarity_cache.get((hpo2, hpo1), 0.0) for hpo1 in drug1_hpo])
        scores_2_to_1.append(best_score)
    
    # Average both directions
    avg_1_to_2 = np.mean(scores_1_to_2)
    avg_2_to_1 = np.mean(scores_2_to_1)
    
    bma_score = 0.5 * (avg_1_to_2 + avg_2_to_1)
    
    return bma_score


# ============================================================================
# PART 5: DATA LOADING
# ============================================================================

def load_sidekick_data(csv_path):
    """
    Load SIDEKICK side effects data.
    
    Returns:
        drug_side_effects: {drug_name: set of HPO IDs}
    """
    print("\n" + "="*70)
    print("LOADING SIDEKICK DATA")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    
    # Filter out root term
    df = df[df['side_effect_hpo_id'] != 'HP:0000001'].copy()
    df = df[df['side_effect_hpo_term'] != 'All'].copy()
    
    print(f"Rows after filtering root term: {len(df)}")
    
    # Filter for single-ingredient drugs only
    df_single = df[~df['ingredients'].str.contains(',', na=False)].copy()
    
    print(f"Single-ingredient drugs: {len(df_single)}")
    
    # Group by ingredient
    drug_side_effects = {}
    for ingredient, group in df_single.groupby('ingredients'):
        hpo_ids = set(group['side_effect_hpo_id'].dropna())
        if len(hpo_ids) > 0:
            drug_side_effects[ingredient.lower()] = hpo_ids
    
    print(f"Unique single-ingredient drugs: {len(drug_side_effects)}")
    
    # Statistics
    side_effect_counts = [len(hpo_ids) for hpo_ids in drug_side_effects.values()]
    print(f"\nSide effects per drug:")
    print(f"  Mean: {np.mean(side_effect_counts):.1f}")
    print(f"  Median: {np.median(side_effect_counts):.1f}")
    print(f"  Min: {np.min(side_effect_counts)}")
    print(f"  Max: {np.max(side_effect_counts)}")
    
    return drug_side_effects


def parse_drugbank_xml(xml_path):
    """
    Parse DrugBank XML to extract drug-target associations.
    
    Returns:
        drug_to_targets: {drug_name: set of target IDs}
    """
    print("\n" + "="*70)
    print("LOADING DRUGBANK (GROUND TRUTH)")
    print("="*70)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'db': 'http://www.drugbank.ca'}
    
    drug_to_targets = {}
    
    for drug in root.findall('db:drug', ns):
        name_elem = drug.find('db:name', ns)
        if name_elem is None:
            continue
        
        drug_name = name_elem.text.lower()
        targets = set()
        
        for target in drug.findall('db:targets/db:target', ns):
            target_id_elem = target.find('db:id', ns)
            if target_id_elem is not None:
                targets.add(target_id_elem.text)
        
        if targets:
            drug_to_targets[drug_name] = targets
    
    print(f"Loaded {len(drug_to_targets)} drugs with targets")
    
    # Statistics
    target_counts = [len(targets) for targets in drug_to_targets.values()]
    print(f"\nTargets per drug:")
    print(f"  Mean: {np.mean(target_counts):.1f}")
    print(f"  Median: {np.median(target_counts):.1f}")
    print(f"  Max: {np.max(target_counts)}")
    
    return drug_to_targets


def create_matched_evaluation_set(sidekick_drugs, drug_to_targets):
    """
    Create matched drug pairs for evaluation.
    
    Positive pairs: drugs that share at least one protein target
    Negative pairs: drugs with no shared targets
    
    Returns:
        - matched_drugs: list of drug names in both datasets
        - positive_pairs: set of (drug1, drug2) tuples
        - negative_pairs: set of (drug1, drug2) tuples
    """
    print("\n" + "="*70)
    print("CREATING MATCHED EVALUATION SET")
    print("="*70)
    
    # Find drugs in both SIDEKICK and DrugBank
    matched_drugs = list(set(sidekick_drugs.keys()) & set(drug_to_targets.keys()))
    
    print(f"Drugs in SIDEKICK: {len(sidekick_drugs)}")
    print(f"Drugs in DrugBank: {len(drug_to_targets)}")
    print(f"Matched drugs: {len(matched_drugs)}")
    
    # Create drug pairs
    positive_pairs = set()
    negative_pairs = set()
    
    print("\nGenerating drug pairs...")
    for drug1, drug2 in tqdm(combinations(matched_drugs, 2), 
                             total=len(matched_drugs)*(len(matched_drugs)-1)//2,
                             desc="Creating pairs"):
        targets1 = drug_to_targets[drug1]
        targets2 = drug_to_targets[drug2]
        
        # Check if drugs share any targets
        if len(targets1 & targets2) > 0:
            positive_pairs.add((drug1, drug2))
        else:
            negative_pairs.add((drug1, drug2))
    
    print(f"\nPositive pairs (shared targets): {len(positive_pairs)}")
    print(f"Negative pairs (no shared targets): {len(negative_pairs)}")
    print(f"Total pairs: {len(positive_pairs) + len(negative_pairs)}")
    
    return matched_drugs, positive_pairs, negative_pairs


# ============================================================================
# PART 6: EVALUATION
# ============================================================================

def evaluate_sidekick(sidekick_drugs, positive_pairs, negative_pairs, 
                     similarity_cache):
    """
    Evaluate SIDEKICK using BMA + Resnik similarity.
    
    Returns:
        y_true: ground truth labels (1=positive, 0=negative)
        y_scores: predicted similarity scores
        metrics: dict with AUC, mean scores, etc.
    """
    print("\n" + "="*70)
    print("EVALUATING SIDEKICK (HPO + BMA + RESNIK)")
    print("="*70)
    
    # Combine all pairs
    all_pairs = list(positive_pairs) + list(negative_pairs)
    y_true = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    y_scores = []
    
    print("Computing similarities for all drug pairs...")
    for drug1, drug2 in tqdm(all_pairs, desc="Computing similarities"):
        hpo1 = sidekick_drugs[drug1]
        hpo2 = sidekick_drugs[drug2]
        
        similarity = bma_similarity(hpo1, hpo2, similarity_cache)
        y_scores.append(similarity)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_scores)
    mean_pos = y_scores[y_true == 1].mean()
    mean_neg = y_scores[y_true == 0].mean()
    delta = mean_pos - mean_neg
    
    metrics = {
        'AUC': auc,
        'Mean_Positive': mean_pos,
        'Mean_Negative': mean_neg,
        'Delta': delta
    }
    
    print(f"\nSIDEKICK (HPO + BMA + Resnik) Results:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Mean similarity (positive pairs): {mean_pos:.4f}")
    print(f"  Mean similarity (negative pairs): {mean_neg:.4f}")
    print(f"  Delta (discrimination): {delta:.4f}")
    
    return y_true, y_scores, metrics


# ============================================================================
# PART 7: VISUALIZATION AND COMPARISON
# ============================================================================

def plot_roc_comparison(y_true, y_scores, sidekick_metrics, output_path):
    """
    Plot ROC curve comparing SIDEKICK vs OnSIDES (hardcoded).
    
    OnSIDES values from paper (Table 2):
    - AUC: 0.6612
    - Mean(+): 0.4412
    - Mean(-): 0.2700
    """
    print("\n" + "="*70)
    print("GENERATING ROC COMPARISON PLOT")
    print("="*70)
    
    # Hardcoded OnSIDES results from paper
    ONSIDES_AUC = 0.6612
    
    plt.figure(figsize=(10, 8))
    
    # Plot SIDEKICK ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, color='#27AE60', lw=3.5, linestyle='-',
             label=f'SIDEKICK (HPO+BMA+Resnik) - AUC = {sidekick_metrics["AUC"]:.4f}',
             alpha=0.9)
    
    # Plot OnSIDES as reference (single point or line - we don't have full curve)
    # Since we only have AUC, we'll show it as a reference line
    plt.axhline(y=ONSIDES_AUC, color='#E74C3C', lw=2.5, linestyle='--',
                label=f'OnSIDES (MedDRA+BMA+Resnik) - AUC = {ONSIDES_AUC:.4f} [Published]',
                alpha=0.7)
    
    # Random classifier
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':',
             label='Random classifier', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('Drug Target Prediction: SIDEKICK vs OnSIDES', 
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11, framealpha=0.95)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {output_path}")


def print_comparison_table(sidekick_metrics):
    """
    Print comparison table with hardcoded OnSIDES values from paper.
    """
    # Hardcoded OnSIDES results from Table 2 of paper
    ONSIDES_RESULTS = {
        'AUC': 0.6612,
        'Mean_Positive': 0.4412,
        'Mean_Negative': 0.2700,
        'Delta': 0.1711
    }
    
    print("\n" + "="*70)
    print("FINAL COMPARISON: SIDEKICK vs OnSIDES")
    print("="*70)
    print(f"\n{'Method':<40} {'AUC-ROC':<10} {'Mean(+)':<10} {'Mean(-)':<10} {'Δ':<10}")
    print("-" * 80)
    
    # OnSIDES (from paper)
    print(f"{'OnSIDES (MedDRA+BMA+Resnik)':<40} "
          f"{ONSIDES_RESULTS['AUC']:<10.4f} "
          f"{ONSIDES_RESULTS['Mean_Positive']:<10.4f} "
          f"{ONSIDES_RESULTS['Mean_Negative']:<10.4f} "
          f"{ONSIDES_RESULTS['Delta']:<10.4f}")
    
    # SIDEKICK (computed)
    print(f"{'SIDEKICK (HPO+BMA+Resnik)':<40} "
          f"{sidekick_metrics['AUC']:<10.4f} "
          f"{sidekick_metrics['Mean_Positive']:<10.4f} "
          f"{sidekick_metrics['Mean_Negative']:<10.4f} "
          f"{sidekick_metrics['Delta']:<10.4f}")
    
    # Calculate improvement
    improvement = ((sidekick_metrics['AUC'] - ONSIDES_RESULTS['AUC']) / 
                   ONSIDES_RESULTS['AUC']) * 100
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"\nHPO vs MedDRA (both using BMA+Resnik):")
    print(f"  SIDEKICK (HPO):     AUC = {sidekick_metrics['AUC']:.4f}")
    print(f"  OnSIDES (MedDRA):   AUC = {ONSIDES_RESULTS['AUC']:.4f}")
    print(f"  → SIDEKICK improves by: {improvement:.2f}%")
    print(f"\n  SIDEKICK discrimination (Δ): {sidekick_metrics['Delta']:.4f}")
    print(f"  OnSIDES discrimination (Δ):  {ONSIDES_RESULTS['Delta']:.4f}")
    
    delta_improvement = ((sidekick_metrics['Delta'] - ONSIDES_RESULTS['Delta']) / 
                        ONSIDES_RESULTS['Delta']) * 100
    print(f"  → Discrimination improves by: {delta_improvement:.2f}%")


def save_results(y_true, y_scores, sidekick_metrics, output_dir):
    """
    Save evaluation results to files.
    """
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_scores
    })
    results_path = os.path.join(output_dir, 'sidekick_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to: {results_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([sidekick_metrics])
    metrics_path = os.path.join(output_dir, 'sidekick_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save comparison with OnSIDES
    comparison_data = {
        'Method': [
            'OnSIDES (MedDRA+BMA+Resnik)',
            'SIDEKICK (HPO+BMA+Resnik)'
        ],
        'AUC': [0.6612, sidekick_metrics['AUC']],
        'Mean_Positive': [0.4412, sidekick_metrics['Mean_Positive']],
        'Mean_Negative': [0.2700, sidekick_metrics['Mean_Negative']],
        'Delta': [0.1711, sidekick_metrics['Delta']]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, 'comparison_summary.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison summary saved to: {comparison_path}")


# ============================================================================
# PART 8: MAIN PIPELINE
# ============================================================================

def main(args):
    """
    Main pipeline for drug repurposing experiment.
    """
    print("\n" + "="*70)
    print("SIDEKICK DRUG REPURPOSING EXPERIMENT")
    print("Predicting Drug Targets via Side Effect Similarity")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Step 1: Load HPO ontology
    terms_dict = parse_obo_file(args.hpo_obo)
    hpo_graph = build_hpo_graph(terms_dict)
    
    # Step 2: Load data
    sidekick_drugs = load_sidekick_data(args.sidekick_csv)
    drug_to_targets = parse_drugbank_xml(args.drugbank_xml)
    
    # Step 3: Create evaluation set
    matched_drugs, positive_pairs, negative_pairs = create_matched_evaluation_set(
        sidekick_drugs, drug_to_targets)
    
    # Step 4: Compute Information Content
    ic_cache = os.path.join(cache_dir, 'hpo_ic.pkl')
    ic_values = compute_hpo_ic_from_data(sidekick_drugs, hpo_graph, ic_cache)
    
    # Step 5: Pre-compute ancestors
    ancestors_cache_file = os.path.join(cache_dir, 'hpo_ancestors.pkl')
    ancestors_cache = precompute_hpo_ancestors(hpo_graph, ancestors_cache_file)
    
    # Step 6: Get unique HPO IDs from matched drugs
    unique_hpo_ids = set()
    for drug in matched_drugs:
        if drug in sidekick_drugs:
            unique_hpo_ids.update(sidekick_drugs[drug])
    
    print(f"\nUnique HPO IDs in matched drugs: {len(unique_hpo_ids)}")
    
    # Step 7: Pre-compute term similarities
    similarity_cache_file = os.path.join(cache_dir, 'hpo_similarities.pkl')
    similarity_cache = precompute_term_similarities(
        unique_hpo_ids, hpo_graph, ic_values, ancestors_cache, similarity_cache_file)
    
    # Step 8: Evaluate SIDEKICK
    y_true, y_scores, sidekick_metrics = evaluate_sidekick(
        sidekick_drugs, positive_pairs, negative_pairs, similarity_cache)
    
    # Step 9: Print comparison with OnSIDES (hardcoded)
    print_comparison_table(sidekick_metrics)
    
    # Step 10: Save results
    save_results(y_true, y_scores, sidekick_metrics, args.output_dir)
    
    # Step 11: Generate visualization
    roc_path = os.path.join(args.output_dir, 'roc_comparison.png')
    plot_roc_comparison(y_true, y_scores, sidekick_metrics, roc_path)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Drug Repurposing Experiment: Predict drug targets via side effect similarity',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--sidekick-csv',
        type=str,
        required=True,
        help='Path to SIDEKICK side_effects_mapped.csv file'
    )
    
    parser.add_argument(
        '--drugbank-xml',
        type=str,
        required=True,
        help='Path to DrugBank full_database.xml file (ground truth)'
    )
    
    parser.add_argument(
        '--hpo-obo',
        type=str,
        required=True,
        help='Path to HPO ontology (hp.obo) file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/drug_repurposing',
        help='Output directory for results and plots'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate input files
    if not os.path.exists(args.sidekick_csv):
        print(f"Error: SIDEKICK CSV not found: {args.sidekick_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.drugbank_xml):
        print(f"Error: DrugBank XML not found: {args.drugbank_xml}")
        sys.exit(1)
    
    if not os.path.exists(args.hpo_obo):
        print(f"Error: HPO OBO file not found: {args.hpo_obo}")
        sys.exit(1)
    
    # Run experiment
    main(args)
