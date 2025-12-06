"""
SIDEKICK Step 2: Map Side Effects to HPO with Validation
Maps only side effects to Human Phenotype Ontology using Graph-RAG with validation
"""

import os
import re
import json
import time
import requests
import networkx as nx
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.auto import tqdm
from getpass import getpass
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve
import obonet 
import pickle
from sentence_transformers import SentenceTransformer  
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR = "data/extracted"
OUTPUT_DIR = "data/mapped"
HPO_OBO_URL = "http://purl.obolibrary.org/obo/hp.obo"
HPO_OBO_FILE = "data/hp.obo"
CACHE_FILE = "data/term_mapping_cache.json"
EMBEDDINGS_FILE = "data/hpo_embeddings.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 1  # Number of terms to process in one API call
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MODEL = "google/gemini-2.5-flash"
TOP_K_SEMANTIC = 10  # Number of semantic matches to retrieve
TOP_K_GRAPH = 15  # Number of related graph nodes to include

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ============================================================================
# HPO ONTOLOGY LOADING AND VALIDATION
# ============================================================================

def download_hpo_obo():
    """Download the HPO OBO file if it doesn't exist."""
    if not os.path.exists(HPO_OBO_FILE):
        print(f"Downloading HPO ontology file from {HPO_OBO_URL}...")
        urlretrieve(HPO_OBO_URL, HPO_OBO_FILE)
        print(f"Downloaded to {HPO_OBO_FILE}")
    else:
        print(f"HPO file already exists at {HPO_OBO_FILE}")

def build_hpo_graph():
    """Build a NetworkX graph from the HPO OBO file."""
    print("Building HPO graph...")
    graph = obonet.read_obo(HPO_OBO_FILE)
    
    # Create dictionary mapping term names to IDs
    name_to_id = {}
    synonym_to_id = {}
    id_to_name = {}
    
    for node_id, data in graph.nodes(data=True):
        # Add primary name
        if 'name' in data:
            name = data['name'].lower()
            name_to_id[name] = node_id
            id_to_name[node_id] = data['name']
        
        # Add synonyms
        if 'synonym' in data:
            for synonym_text in data['synonym']:
                # Extract the actual synonym text from the OBO format
                match = re.search(r'"([^"]*)"', synonym_text)
                if match:
                    synonym = match.group(1).lower()
                    synonym_to_id[synonym] = node_id
    
    print(f"HPO graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"Mapped {len(name_to_id)} primary names and {len(synonym_to_id)} synonyms")
    
    return graph, name_to_id, synonym_to_id, id_to_name

def validate_hpo_mapping(hpo_id, hpo_term, id_to_name):
    """
    Validate that HPO ID exists and matches the provided term.
    
    Args:
        hpo_id: HPO ID to validate
        hpo_term: HPO term to validate
        id_to_name: Dictionary mapping HPO IDs to official names
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if ID exists in ontology
    if hpo_id not in id_to_name:
        return False
    
    # Check if term matches the official term for this ID
    official_term = id_to_name[hpo_id]
    if official_term.lower() != hpo_term.lower():
        return False
    
    return True

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def load_mapping_cache():
    """Load existing term mapping cache or create a new one."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(f"Loaded mapping cache with {len(cache)} entries")
        return cache
    else:
        print("No existing cache found. Creating new cache.")
        return {}

def save_mapping_cache(cache):
    """Save the term mapping cache to disk."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    print(f"Saved mapping cache with {len(cache)} entries")

# ============================================================================
# DIRECT MATCHING
# ============================================================================

def direct_match_term(term, name_to_id, synonym_to_id, graph):
    """Try to match a term directly to HPO terms or synonyms."""
    term_lower = term.lower()
    
    # Check for exact match in primary names
    if term_lower in name_to_id:
        node_id = name_to_id[term_lower]
        return {
            'hpo_id': node_id,
            'hpo_term': graph.nodes[node_id].get('name'),
            'match_type': 'direct_name'
        }
    
    # Check for exact match in synonyms
    if term_lower in synonym_to_id:
        node_id = synonym_to_id[term_lower]
        return {
            'hpo_id': node_id,
            'hpo_term': graph.nodes[node_id].get('name'),
            'match_type': 'direct_synonym'
        }
    
    return None

# ============================================================================
# EMBEDDING AND SEMANTIC SEARCH
# ============================================================================

def compute_and_store_hpo_embeddings(graph, force_recompute=False):
    """Compute and store embeddings for all HPO terms."""
    if os.path.exists(EMBEDDINGS_FILE) and not force_recompute:
        print(f"Loading existing HPO embeddings from {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    
    print("Computing embeddings for all HPO terms...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Prepare texts to embed (node names and synonyms)
    texts_to_embed = []
    text_metadata = []
    
    for node_id, data in graph.nodes(data=True):
        if 'name' in data:
            # Add primary name
            texts_to_embed.append(data['name'])
            text_metadata.append({
                'node_id': node_id,
                'text_type': 'name',
                'text': data['name']
            })
            
            # Add synonyms if they exist
            if 'synonym' in data:
                for synonym_text in data['synonym']:
                    match = re.search(r'"([^"]*)"', synonym_text)
                    if match:
                        synonym = match.group(1)
                        texts_to_embed.append(synonym)
                        text_metadata.append({
                            'node_id': node_id,
                            'text_type': 'synonym',
                            'text': synonym
                        })
    
    # Compute embeddings in batches to manage memory
    batch_size = 1000
    all_embeddings = []
    
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    
    # Create embeddings dictionary
    embeddings_dict = {
        'metadata': text_metadata,
        'embeddings': embeddings,
        'model_name': EMBEDDING_MODEL
    }
    
    # Save embeddings
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"Computed and saved embeddings for {len(text_metadata)} HPO terms and synonyms")
    return embeddings_dict

def find_semantic_matches(term, embeddings_dict, graph, top_k=TOP_K_SEMANTIC):
    """Find semantically similar HPO terms for a given term using embeddings."""
    model = SentenceTransformer(embeddings_dict['model_name'])
    
    # Compute embedding for the query term
    query_embedding = model.encode([term])[0].reshape(1, -1)
    
    # Compute cosine similarity with all HPO term embeddings
    similarities = cosine_similarity(
        query_embedding, 
        embeddings_dict['embeddings']
    )[0]
    
    # Get top-k most similar terms
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Get unique node IDs (avoiding duplicates from synonyms)
    seen_node_ids = set()
    semantic_matches = []
    
    for idx in top_indices:
        node_id = embeddings_dict['metadata'][idx]['node_id']
        if node_id in seen_node_ids:
            continue
            
        seen_node_ids.add(node_id)
        node_name = graph.nodes[node_id].get('name')
        similarity = similarities[idx]
        
        semantic_matches.append({
            'hpo_id': node_id,
            'hpo_term': node_name,
            'similarity': float(similarity),
            'matched_text': embeddings_dict['metadata'][idx]['text']
        })
    
    return semantic_matches

# ============================================================================
# GRAPH TRAVERSAL AND CONTEXT CREATION
# ============================================================================

def traverse_graph(seed_nodes, graph, max_nodes=TOP_K_GRAPH):
    """Traverse the HPO graph starting from seed nodes to find related concepts."""
    related_nodes = {}  # Dictionary to store node_id -> relationship
    
    # Add seed nodes first
    for node in seed_nodes:
        related_nodes[node] = 'seed'
    
    # Breadth-first traversal
    frontier = list(seed_nodes)
    visited = set(seed_nodes)
    
    while frontier and len(related_nodes) < max_nodes:
        current = frontier.pop(0)
        
        # Get parents (is_a relationships in HPO)
        for parent in graph.predecessors(current):
            if parent not in visited:
                related_nodes[parent] = 'parent'
                visited.add(parent)
                frontier.append(parent)
                
                if len(related_nodes) >= max_nodes:
                    break
        
        # Get children (more specific terms)
        for child in graph.successors(current):
            if child not in visited:
                related_nodes[child] = 'child'
                visited.add(child)
                frontier.append(child)
                
                if len(related_nodes) >= max_nodes:
                    break
                    
        # Get siblings (terms sharing a parent)
        for parent in graph.predecessors(current):
            for sibling in graph.successors(parent):
                if sibling != current and sibling not in visited:
                    related_nodes[sibling] = 'sibling'
                    visited.add(sibling)
                    frontier.append(sibling)
                    
                    if len(related_nodes) >= max_nodes:
                        break
    
    return related_nodes

def create_enriched_context(graph, term_batch, semantic_matches_dict):
    """Create rich context for the API prompt using semantic matches and graph traversal."""
    hpo_info = "Here are semantically relevant HPO terms for the batch:\n\n"
    
    # Get all seed nodes from semantic matches
    all_seed_nodes = set()
    for term, matches in semantic_matches_dict.items():
        top_matches = matches[:5]  # Take top 5 matches for each term
        all_seed_nodes.update([match['hpo_id'] for match in top_matches])
    
    # Traverse graph to find related nodes
    related_nodes = traverse_graph(all_seed_nodes, graph)
    
    # First, add the direct semantic matches with similarity scores
    hpo_info += "## Top Semantic Matches\n"
    for term, matches in semantic_matches_dict.items():
        hpo_info += f"\nFor '{term}':\n"
        for i, match in enumerate(matches[:5]):  # Show top 5 matches per term
            hpo_info += f"- {match['hpo_term']} ({match['hpo_id']})\n"
            # Show which text matched (helpful for synonyms)
            if match['matched_text'].lower() != match['hpo_term'].lower():
                hpo_info += f"  └─ Matched via synonym: '{match['matched_text']}'\n"
    
    # Add graph structure information
    hpo_info += "\n## HPO Graph Context\n"
    
    # First add the seed nodes with their definitions
    hpo_info += "\n### Core Terms:\n"
    for node_id in all_seed_nodes:
        node_data = graph.nodes[node_id]
        name = node_data.get('name', 'Unknown')
        definition = node_data.get('def', 'No definition available')
        if isinstance(definition, list) and len(definition) > 0:
            definition = definition[0]
        # Extract definition text
        def_match = re.search(r'"([^"]*)"', definition) if isinstance(definition, str) else None
        def_text = def_match.group(1) if def_match else "No definition available"
        
        hpo_info += f"- {name} ({node_id})\n"
        hpo_info += f"  Definition: {def_text[:200]}{'...' if len(def_text) > 200 else ''}\n"
        
        # Add synonyms for this term
        if 'synonym' in node_data and len(node_data['synonym']) > 0:
            synonym_texts = []
            for syn in node_data['synonym'][:5]:  # Take first 5 synonyms
                syn_match = re.search(r'"([^"]*)"', syn)
                if syn_match:
                    synonym_texts.append(syn_match.group(1))
            if synonym_texts:
                hpo_info += f"  Synonyms: {', '.join(synonym_texts)}\n"
    
    # Add parent-child relationships
    hpo_info += "\n### Term Relationships:\n"
    for node_id, rel_type in related_nodes.items():
        if rel_type != 'seed':  # Skip seed nodes as they were already covered
            node_data = graph.nodes[node_id]
            name = node_data.get('name', 'Unknown')
            
            rel_symbol = "⊂" if rel_type == 'child' else "⊃" if rel_type == 'parent' else "∼"
            
            hpo_info += f"- {name} ({node_id}) {rel_symbol}\n"
            
            # For parents and children, show the relationship
            if rel_type in ['parent', 'child']:
                related_to = []
                if rel_type == 'parent':
                    # Find which seeds this is a parent of
                    for seed in all_seed_nodes:
                        if seed in graph.successors(node_id):
                            related_to.append(graph.nodes[seed].get('name', 'Unknown'))
                else:  # child
                    # Find which seeds this is a child of
                    for seed in all_seed_nodes:
                        if seed in graph.predecessors(node_id):
                            related_to.append(graph.nodes[seed].get('name', 'Unknown'))
                
                if related_to:
                    relation = "Parent of" if rel_type == 'parent' else "Child of"
                    hpo_info += f"  └─ {relation}: {', '.join(related_to[:3])}\n"
    
    return hpo_info

# ============================================================================
# API INTEGRATION WITH VALIDATION
# ============================================================================

def call_api_with_validation(batch_terms, api_key, enriched_context, id_to_name):
    """
    Call the OpenRouter API to map terms to HPO with validation.
    Only returns mappings that pass validation.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://jupyter.org"
    }
    
    terms_list = "\n".join([f"- {term}" for term in batch_terms])
    
    prompt = f"""You are an expert in medical terminology and ontologies. Your task is to map clinical terms to the Human Phenotype Ontology (HPO).

For each term provided, find the most appropriate HPO term and ID. Use the provided Graph RAG context to make the best possible matches.

Terms to map:
{terms_list}

You should respond with a JSON object where each key is the original term and each value is an object with 'hpo_id' and 'hpo_term'.

Example format:
{{
  "headache": {{
    "hpo_id": "HP:0002315",
    "hpo_term": "Headache"
  }},
  "difficulty breathing": {{
    "hpo_id": "HP:0002094",
    "hpo_term": "Dyspnea"
  }}
}}

CRITICAL: Ensure the hpo_id and hpo_term match exactly with the HPO ontology. Double-check your mappings.

Please provide only the JSON result with no additional explanation or text.

## GRAPH RAG CONTEXT FROM HPO ONTOLOGY
{enriched_context}"""
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                
                # Extract the JSON from the response
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content
                
                # Clean up the json string
                json_str = json_str.strip()
                if json_str.startswith('```') and json_str.endswith('```'):
                    json_str = json_str[3:-3].strip()
                
                try:
                    result = json.loads(json_str)
                    
                    # VALIDATION: Only keep mappings that pass validation
                    validated_result = {}
                    for term, mapping in result.items():
                        if 'hpo_id' in mapping and 'hpo_term' in mapping:
                            hpo_id = mapping['hpo_id'].strip()
                            hpo_term = mapping['hpo_term'].strip()
                            
                            # Validate against ontology
                            if validate_hpo_mapping(hpo_id, hpo_term, id_to_name):
                                validated_result[term] = {
                                    'hpo_id': hpo_id,
                                    'hpo_term': hpo_term
                                }
                            else:
                                print(f"  Warning: Validation failed for '{term}' -> {hpo_id}:{hpo_term}")
                    
                    return validated_result
                    
                except json.JSONDecodeError as e:
                    print(f"  JSON decode error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
            
            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"  API request failed: {response.status_code} - {response.text}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        except Exception as e:
            print(f"  Error calling OpenRouter API: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    # If all retries failed, return empty result
    return {}

# ============================================================================
# SIDE EFFECTS MAPPING
# ============================================================================

def map_side_effects_with_graphrag(drug_info, graph, name_to_id, synonym_to_id, id_to_name, cache, api_key, embeddings_dict):
    """Map side effects to HPO using direct matching first, then validated Graph RAG."""
    
    # Get all unique side effect terms
    all_terms = set(drug_info['side_effects'])
    
    # Handle empty side effects gracefully
    if not all_terms:
        print(f"Warning: No side effects found for drug {drug_info['set_id']}")
        return {
            'set_id': drug_info['set_id'],
            'version': drug_info['version'],
            'side_effects': []
        }
    
    # Try direct matching first and separate cached from new terms
    direct_matches = {}
    terms_needing_rag = []
    
    for term in all_terms:
        # Check cache first
        if term in cache:
            direct_matches[term] = cache[term]
            continue
        
        # Try direct matching
        match = direct_match_term(term, name_to_id, synonym_to_id, graph)
        if match:
            direct_matches[term] = match
            cache[term] = match  # Update cache with direct match
        else:
            terms_needing_rag.append(term)
    
    # Process remaining terms with Graph RAG and validation
    rag_matches = {}
    if terms_needing_rag:
        # Create batches of terms for API calls
        batches = [terms_needing_rag[i:i + BATCH_SIZE] for i in range(0, len(terms_needing_rag), BATCH_SIZE)]
        
        for batch in tqdm(batches, desc="Processing term batches with Graph RAG"):
            try:
                # Find semantic matches for each term in batch
                batch_semantic_matches = {}
                for term in batch:
                    semantic_matches = find_semantic_matches(term, embeddings_dict, graph)
                    batch_semantic_matches[term] = semantic_matches
                
                # Create enriched context using graph traversal
                enriched_context = create_enriched_context(graph, batch, batch_semantic_matches)
                
                # Call OpenRouter API with validation
                batch_results = call_api_with_validation(batch, api_key, enriched_context, id_to_name)
                
                # Process the batch results (all are already validated)
                for term, mapping in batch_results.items():
                    rag_matches[term] = {
                        'hpo_id': mapping['hpo_id'],
                        'hpo_term': mapping['hpo_term'],
                        'match_type': 'semantic_graph_rag'
                    }
                    cache[term] = rag_matches[term]  # Update cache with validated RAG match
                    
            except Exception as e:
                print(f"  Error processing batch {batch}: {str(e)}")
                # Continue with the next batch instead of failing
                continue
    
    # Combine all matches
    all_mappings = {**direct_matches, **rag_matches}
    
    # Create mapped drug info structure
    mapped_drug_info = {
        'set_id': drug_info['set_id'],
        'version': drug_info['version'],
        'side_effects': []
    }
    
    # Map side effects
    for side_effect in drug_info['side_effects']:
        mapping = all_mappings.get(side_effect, {
            'hpo_id': 'HP:0000001',  # Root HPO term as fallback
            'hpo_term': 'All',
            'match_type': 'no_match'
        })
        
        mapped_drug_info['side_effects'].append({
            'side_effect_name': side_effect,
            'hpo_mapping': {
                'hpo_id': mapping['hpo_id'],
                'hpo_term': mapping['hpo_term']
            }
        })
    
    return mapped_drug_info

# ============================================================================
# FILE PARSING
# ============================================================================

def parse_drug_file(file_path):
    """
    Parse a drug information file and extract only side effects.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract set_id and version using regex
        set_id_match = re.search(r'<set_id>(.*?)</set_id>', content)
        version_match = re.search(r'<version>(.*?)</version>', content)
        
        set_id = set_id_match.group(1) if set_id_match else "unknown"
        version = version_match.group(1) if version_match else "unknown"
        
        # Initialize drug info dictionary
        drug_info = {
            'set_id': set_id,
            'version': version,
            'side_effects': []
        }
        
        # Extract side effects (handle both tag variations)
        # First try with regular side_effect_name
        side_effect_pattern1 = r'<side_effect>\s*<side_effect_name>(.*?)</side_effect_name>.*?</side_effect>'
        side_effects1 = re.findall(side_effect_pattern1, content, re.DOTALL)
        
        # Then try with side_side_effect_name (which appears in some files)
        side_effect_pattern2 = r'<side_effect>\s*<side_side_effect_name>(.*?)</side_side_effect_name>.*?</side_effect>'
        side_effects2 = re.findall(side_effect_pattern2, content, re.DOTALL)
        
        # Combine results
        side_effects = side_effects1 + side_effects2
        drug_info['side_effects'] = [se.strip() for se in side_effects if se.strip()]
        
        # If no side effects found with nested tags, try simpler patterns
        if not drug_info['side_effects']:
            side_effect_name_pattern1 = r'<side_effect_name>(.*?)</side_effect_name>'
            side_effect_name_pattern2 = r'<side_side_effect_name>(.*?)</side_side_effect_name>'
            
            side_effects = re.findall(side_effect_name_pattern1, content) + re.findall(side_effect_name_pattern2, content)
            drug_info['side_effects'] = [se.strip() for se in side_effects if se.strip()]
        
        # Clean side effect names of any XML tags
        drug_info['side_effects'] = [re.sub(r'<[^>]*>', '', se) for se in drug_info['side_effects']]
        
        # Log a warning if nothing was found
        if not drug_info['side_effects']:
            print(f"Warning: No side effects found in {file_path}")
        
        return drug_info
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Return empty but valid structure instead of None
        return {
            'set_id': "unknown" if 'set_id' not in locals() else set_id,
            'version': "unknown" if 'version' not in locals() else version,
            'side_effects': []
        }

# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def generate_output_xml(mapped_drug_info):
    """Generate XML output for the mapped side effects."""
    xml_lines = []
    xml_lines.append(f"<set_id>{mapped_drug_info['set_id']}</set_id>")
    xml_lines.append(f"<version>{mapped_drug_info['version']}</version>")
    xml_lines.append("")
    xml_lines.append("<drug_information>")
    
    # Add side effects
    if mapped_drug_info['side_effects']:
        xml_lines.append("  <side_effects>")
        for side_effect in mapped_drug_info['side_effects']:
            xml_lines.append("    <side_effect>")
            xml_lines.append(f"      <side_effect_name>{side_effect['side_effect_name']}</side_effect_name>")
            xml_lines.append("      <hpo_mapping>")
            xml_lines.append(f"        <hpo_id>{side_effect['hpo_mapping']['hpo_id']}</hpo_id>")
            xml_lines.append(f"        <hpo_term>{side_effect['hpo_mapping']['hpo_term']}</hpo_term>")
            xml_lines.append("      </hpo_mapping>")
            xml_lines.append("    </side_effect>")
        xml_lines.append("  </side_effects>")
    
    xml_lines.append("</drug_information>")
    
    return "\n".join(xml_lines)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def map_side_effects_to_hpo():
    """Main function to process all drug files and map side effects to HPO terms using validated Graph RAG."""
    
    print("="*80)
    print("SIDEKICK STEP 2: MAP SIDE EFFECTS TO HPO WITH VALIDATION")
    print("="*80)
    
    # Get OpenRouter API key
    api_key = getpass("Enter your OpenRouter API key (will be hidden): ")
    
    # Download HPO if needed
    download_hpo_obo()
    
    # Build HPO graph with validation support
    graph, name_to_id, synonym_to_id, id_to_name = build_hpo_graph()
    
    # Compute or load HPO embeddings
    embeddings_dict = compute_and_store_hpo_embeddings(graph)
    
    # Load mapping cache
    cache = load_mapping_cache()
    
    # Get list of files to process
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    print(f"\nFound {len(input_files)} files to process")
    
    # Track statistics
    stats = {
        'total_files': len(input_files),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_side_effects': 0,
        'direct_matches': 0,
        'rag_matches': 0,
        'root_fallbacks': 0
    }
    
    # Process each file
    for i, file_name in enumerate(tqdm(input_files, desc="Processing drug files")):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name.replace('.txt', '_mapped.txt'))
        
        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"Skipping {file_name} - output already exists")
            stats['skipped'] += 1
            continue
        
        try:
            # Parse drug file
            drug_info = parse_drug_file(input_path)
            if not drug_info:
                print(f"Failed to parse {file_name}")
                stats['errors'] += 1
                continue
            
            # Track side effects count
            stats['total_side_effects'] += len(drug_info['side_effects'])
            
            # Map side effects to HPO using validated Graph RAG
            mapped_drug_info = map_side_effects_with_graphrag(
                drug_info, graph, name_to_id, synonym_to_id, id_to_name, cache, api_key, embeddings_dict
            )
            
            # Count mapping types
            for se in mapped_drug_info['side_effects']:
                hpo_id = se['hpo_mapping']['hpo_id']
                if hpo_id == 'HP:0000001':
                    stats['root_fallbacks'] += 1
            
            # Generate output XML
            output_xml = generate_output_xml(mapped_drug_info)
            
            # Save output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_xml)
            
            stats['processed'] += 1
            
            # Save cache every 10 files to prevent data loss
            if i % 10 == 0:
                save_mapping_cache(cache)
        
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            stats['errors'] += 1
    
    # Save final cache
    save_mapping_cache(cache)
    
    # Print final statistics
    print("\n" + "="*80)
    print("MAPPING COMPLETE - FINAL STATISTICS")
    print("="*80)
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped (already exists): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total side effects processed: {stats['total_side_effects']}")
    print(f"Root fallbacks (no match): {stats['root_fallbacks']}")
    print(f"\nAll mappings are validated against HPO ontology!")
    print("="*80)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    map_side_effects_to_hpo()
