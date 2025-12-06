"""
SIDEKICK Step 3: Indications & Contraindications Processing Pipeline
Complete pipeline: Extract → Classify → Map (Disease/Phenotype to MONDO/HPO, Drug/Chemical to RxNorm)
Includes validation and self-reference cleaning
"""

import os
import csv
import re
import json
import requests
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from getpass import getpass
from urllib.request import urlretrieve
import obonet 
import pickle
from sentence_transformers import SentenceTransformer  
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline as hf_pipeline
from time import sleep

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
DATA_DIR = "data"
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")

# Output files
IC_CSV = os.path.join(DATA_DIR, "indications_contraindications.csv")
IC_CLASSIFIED_CSV = os.path.join(DATA_DIR, "indications_contraindications_classified.csv")
IC_DISEASE_PHENOTYPE_CSV = os.path.join(DATA_DIR, "indications_contraindications_mapped_disease_phenotype.csv")
IC_DRUG_CHEMICAL_CSV = os.path.join(DATA_DIR, "indications_contraindications_mapped_drug_chemical.csv")

# Progress files
CLASSIFICATION_PROGRESS = os.path.join(DATA_DIR, "classification_progress.json")
MAPPING_PROGRESS = os.path.join(DATA_DIR, "mapping_progress.json")
DRUG_MAPPING_PROGRESS = os.path.join(DATA_DIR, "drug_mapping_progress.json")

# Ontology files
HPO_OBO_URL = "http://purl.obolibrary.org/obo/hp.obo"
HPO_OBO_FILE = os.path.join(DATA_DIR, "hp.obo")
MONDO_OBO_URL = "http://purl.obolibrary.org/obo/mondo.obo"
MONDO_OBO_FILE = os.path.join(DATA_DIR, "mondo.obo")

# Embeddings
HPO_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "hpo_embeddings_ic.pkl")
MONDO_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "mondo_embeddings.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# API Configuration
MODEL = "google/gemini-2.5-flash"
LITE_MODEL = "google/gemini-2.5-flash-lite"
MAX_RETRIES = 3
RETRY_DELAY = 1

# Graph RAG Configuration
TOP_K_SEMANTIC = 10
TOP_K_GRAPH = 15

# Root nodes
HPO_ROOT = "HP:0000001"
MONDO_ROOT = "MONDO:0000001"

# Classification categories
CATEGORIES = [
    "Disease",
    "Phenotype", 
    "Drug/Chemical",
    "Allergy/Hypersensitivity",
    "Patient_Population",
    "Procedure",
    "Other"
]

ALLERGY_KEYWORDS = [
    'hypersensitivity', 'hypersensitive', 'sensitivity', 'sensitive',
    'allergy', 'allergic', 'anaphylaxis', 'anaphylactic'
]

# RxNav Configuration
RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
NER_MODEL = "alvaroalon2/biobert_chemical_ner"
DRUG_FALLBACK = "other"

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# STEP 1: EXTRACT INDICATIONS & CONTRAINDICATIONS
# ============================================================================

def extract_indications_contraindications_from_file(file_path):
    """Extract indications and contraindications from an extracted file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
        
        # Extract set_id
        set_id_match = re.search(r'<set_id>(.*?)</set_id>', content)
        set_id = set_id_match.group(1).strip() if set_id_match else None
        
        # Extract version
        version_match = re.search(r'<version>(.*?)</version>', content)
        version = version_match.group(1).strip() if version_match else None
        
        if not set_id:
            return None, None, []
        
        items = []
        
        # Extract indications
        indications_section = re.search(r'<indications>(.*?)</indications>', content, re.DOTALL)
        if indications_section:
            indication_blocks = re.findall(r'<indication>(.*?)</indication>', indications_section.group(1), re.DOTALL)
            
            for block in indication_blocks:
                name_match = re.search(r'<indication_name>(.*?)</indication_name>', block)
                if name_match:
                    indication_name = name_match.group(1).strip()
                    clean_name = re.sub(r'<[^>]*>', '', indication_name)
                    if clean_name and clean_name.lower() != 'none':
                        items.append((clean_name, 'I'))
        
        # Extract contraindications
        contraindications_section = re.search(r'<contraindications>(.*?)</contraindications>', content, re.DOTALL)
        if contraindications_section:
            contraindication_blocks = re.findall(r'<contraindication>(.*?)</contraindication>', contraindications_section.group(1), re.DOTALL)
            
            for block in contraindication_blocks:
                name_match = re.search(r'<contraindication_name>(.*?)</contraindication_name>', block)
                if name_match:
                    contraindication_name = name_match.group(1).strip()
                    clean_name = re.sub(r'<[^>]*>', '', contraindication_name)
                    if clean_name and clean_name.lower() != 'none':
                        items.append((clean_name, 'C'))
        
        return set_id, version, items
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, []

def create_indications_contraindications_csv():
    """Extract I&C from all extracted files and create CSV."""
    print("="*80)
    print("STEP 1: EXTRACTING INDICATIONS & CONTRAINDICATIONS")
    print("="*80)
    
    # Load product ingredients data
    ingredients_file = os.path.join(DATA_DIR, 'human_product_ingredients.csv')
    
    try:
        ingredients_df = pd.read_csv(ingredients_file)
        product_dict = {}
        for _, row in ingredients_df.iterrows():
            product_dict[row['set_id']] = {
                'ingredients': row['ingredients'],
                'ingredient_rxcuis': row['ingredient_rxcuis']
            }
        print(f"Loaded product data for {len(product_dict)} drugs")
    except Exception as e:
        print(f"Error loading ingredients file: {e}")
        return 0
    
    # Find all extracted files
    extracted_files = [f for f in os.listdir(EXTRACTED_DIR) if f.endswith('.txt')]
    
    # Prepare CSV data
    csv_data = []
    processed_files = 0
    total_indications = 0
    total_contraindications = 0
    
    print(f"Processing {len(extracted_files)} extracted files...")
    for idx, file_name in enumerate(extracted_files):
        if idx % 100 == 0 and idx > 0:
            print(f"Progress: {idx}/{len(extracted_files)} files processed")
            
        file_path = os.path.join(EXTRACTED_DIR, file_name)
        set_id, version, items = extract_indications_contraindications_from_file(file_path)
        
        if set_id is None or not items:
            continue
        
        product_info = product_dict.get(set_id)
        if not product_info:
            print(f"Warning: No product info found for set_id {set_id}")
            continue
        
        for name, item_type in items:
            csv_data.append([
                product_info['ingredients'],
                product_info['ingredient_rxcuis'],
                set_id,
                version if version else "",
                name,
                item_type
            ])
            
            if item_type == 'I':
                total_indications += 1
            else:
                total_contraindications += 1
        
        processed_files += 1
    
    print(f"Successfully processed {processed_files} files")
    print(f"Total indications: {total_indications}")
    print(f"Total contraindications: {total_contraindications}")
    
    # Write to CSV file
    header = ['ingredients', 'ingredient_rxcuis', 'set_id', 'spl_version', 
              'indication_contraindication_name', 'type']
    
    with open(IC_CSV, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(csv_data)
    
    print(f"Created {IC_CSV} with {len(csv_data)} rows")
    return len(csv_data)

# ============================================================================
# STEP 2: CLASSIFY INDICATIONS & CONTRAINDICATIONS
# ============================================================================

def contains_allergy_keyword(text):
    """Check if text contains allergy-related keywords."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ALLERGY_KEYWORDS)

def create_classification_prompt(terms):
    """Create a prompt for classifying multiple terms at once."""
    
    prompt = """You are a medical terminology expert. Classify each of the following medical indications or contraindications into ONE of these categories:

**Categories:**

1. **Disease** - Medical conditions, disorders, syndromes, infections with clear specific name
   Examples: "chronic lymphocytic leukemia", "hypertension", "diabetes", "myasthenia gravis", "glaucoma"

2. **Phenotype** - Observable clinical signs, symptoms, or physical manifestations
   Examples: "pain", "nausea", "bleeding", "seizures", "diarrhea", "increased risk of aspiration", "hypocalcemia"

3. **Drug/Chemical** - Drug interactions, concomitant medication use, or chemical exposures
   Examples: "coadministration with strong OATP1B inhibitors", "coadministration with efavirenz", "use with warfarin"

4. **Allergy/Hypersensitivity** - Allergic reactions or hypersensitivity to specific substances
   Examples: "hypersensitivity to penicillin", "allergy to sulfonamides"

5. **Patient_Population** - Demographics, life stages, or specific patient groups. Use ONLY if just a group is mentioned without any additional context.
   Examples: "pregnancy", "breastfeeding", "postmenopausal women", "pediatric patients"

6. **Procedure** - Medical or surgical procedures and clinical contexts
   Examples: "treatment following cataract surgery"

7. **Other** - Terms that don't clearly fit into the above categories

**Instructions:**
- Classify each term into exactly ONE category
- Choose the MOST SPECIFIC category that applies
- Return your answer as a JSON object with term as key and category as value

**Terms to classify:**
"""
    
    for i, term in enumerate(terms, 1):
        prompt += f"{i}. {term}\n"
    
    prompt += """\n**Return format (JSON only, no additional text):**
{
  "term1": "Category",
  "term2": "Category",
  ...
}"""
    
    return prompt

def classify_terms_batch(terms, api_key, batch_size=15):
    """Classify a batch of terms using OpenRouter API."""
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = create_classification_prompt(terms)
    
    data = {
        "model": LITE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 20000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            classifications = json.loads(json_match.group())
            return classifications
        else:
            print(f"Warning: Could not parse JSON from response")
            return {}
            
    except Exception as e:
        print(f"Error during API call: {e}")
        return {}

def classify_indications_contraindications(api_key, limit=None, batch_size=15):
    """Classify all indications and contraindications."""
    print("\n" + "="*80)
    print("STEP 2: CLASSIFYING INDICATIONS & CONTRAINDICATIONS")
    print("="*80)
    
    print(f"Loading {IC_CSV}...")
    df = pd.read_csv(IC_CSV)
    print(f"Loaded {len(df)} rows")
    
    unique_terms = df['indication_contraindication_name'].dropna().unique()
    print(f"Found {len(unique_terms)} unique terms")
    
    if limit is not None:
        unique_terms = unique_terms[:limit]
        print(f"Limited to {len(unique_terms)} terms")
    
    # Load progress
    term_to_class = {}
    if os.path.exists(CLASSIFICATION_PROGRESS):
        try:
            with open(CLASSIFICATION_PROGRESS, 'r') as f:
                term_to_class = json.load(f)
            print(f"Loaded {len(term_to_class)} previously classified terms")
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
    
    # Auto-classify allergy terms
    allergy_terms = []
    remaining_terms = []
    
    for term in unique_terms:
        if term in term_to_class:
            continue
        elif contains_allergy_keyword(term):
            term_to_class[term] = "Allergy/Hypersensitivity"
            allergy_terms.append(term)
        else:
            remaining_terms.append(term)
    
    print(f"Auto-classified {len(allergy_terms)} allergy terms")
    print(f"Remaining terms to classify: {len(remaining_terms)}")
    
    # Classify remaining terms
    total_batches = (len(remaining_terms) + batch_size - 1) // batch_size
    
    for i in range(0, len(remaining_terms), batch_size):
        batch = remaining_terms[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        classifications = classify_terms_batch(batch, api_key, batch_size)
        
        for term in batch:
            if term in classifications:
                term_to_class[term] = classifications[term]
            else:
                term_to_class[term] = "Other"
                print(f"Warning: No classification for '{term}', assigned to 'Other'")
        
        # Save progress
        with open(CLASSIFICATION_PROGRESS, 'w') as f:
            json.dump(term_to_class, f, indent=2)
        
        if batch_num < total_batches:
            sleep(1)
    
    # Apply classifications
    print("Applying classifications to dataframe...")
    df['class'] = df['indication_contraindication_name'].map(term_to_class)
    
    # Handle unmapped terms
    df['class'].fillna('Other', inplace=True)
    
    # Save classified CSV
    df.to_csv(IC_CLASSIFIED_CSV, index=False)
    print(f"Saved {IC_CLASSIFIED_CSV}")
    
    # Clean up progress
    if os.path.exists(CLASSIFICATION_PROGRESS):
        os.remove(CLASSIFICATION_PROGRESS)
    
    # Print statistics
    print("\nClassification Summary:")
    print(df['class'].value_counts())
    
    return df

# ============================================================================
# STEP 3: MAP DISEASE & PHENOTYPE TERMS TO MONDO & HPO
# ============================================================================

def download_ontology_file(url, filepath, name):
    """Download an ontology OBO file if it doesn't exist."""
    if not os.path.exists(filepath):
        print(f"Downloading {name} ontology from {url}...")
        urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")
    else:
        print(f"{name} file already exists at {filepath}")

def build_ontology_graph(obo_file, ontology_name):
    """Build a NetworkX graph from an OBO file."""
    print(f"Building {ontology_name} graph...")
    graph = obonet.read_obo(obo_file)
    
    name_to_id = {}
    synonym_to_id = {}
    id_to_name = {}
    
    for node_id, data in graph.nodes(data=True):
        if 'name' in data:
            name = data['name'].lower()
            name_to_id[name] = node_id
            id_to_name[node_id] = data['name']
        
        if 'synonym' in data:
            for synonym_text in data['synonym']:
                match = re.search(r'"([^"]*)"', synonym_text)
                if match:
                    synonym = match.group(1).lower()
                    synonym_to_id[synonym] = node_id
    
    print(f"{ontology_name} graph built with {len(graph.nodes)} nodes")
    return graph, name_to_id, synonym_to_id, id_to_name

def exact_match_term(term, name_to_id, synonym_to_id, id_to_name):
    """Try to match a term directly to ontology terms or synonyms."""
    term_lower = term.lower()
    
    if term_lower in name_to_id:
        node_id = name_to_id[term_lower]
        return {
            'ontology_id': node_id,
            'ontology_term': id_to_name[node_id],
            'match_type': 'exact_name'
        }
    
    if term_lower in synonym_to_id:
        node_id = synonym_to_id[term_lower]
        return {
            'ontology_id': node_id,
            'ontology_term': id_to_name[node_id],
            'match_type': 'exact_synonym'
        }
    
    return None

def compute_ontology_embeddings(graph, embeddings_file, ontology_name, force_recompute=False):
    """Compute and store embeddings for all ontology terms."""
    if os.path.exists(embeddings_file) and not force_recompute:
        print(f"Loading existing {ontology_name} embeddings")
        with open(embeddings_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing {ontology_name} embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts_to_embed = []
    text_metadata = []
    
    for node_id, data in graph.nodes(data=True):
        if 'name' in data:
            texts_to_embed.append(data['name'])
            text_metadata.append({
                'node_id': node_id,
                'text_type': 'name',
                'text': data['name']
            })
            
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
    
    batch_size = 1000
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc=f"{ontology_name} embeddings"):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    
    embeddings_dict = {
        'metadata': text_metadata,
        'embeddings': embeddings,
        'model_name': EMBEDDING_MODEL
    }
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"Saved {len(text_metadata)} {ontology_name} embeddings")
    return embeddings_dict

def find_semantic_matches(term, embeddings_dict, graph, top_k=TOP_K_SEMANTIC):
    """Find semantically similar ontology terms using embeddings."""
    model = SentenceTransformer(embeddings_dict['model_name'])
    query_embedding = model.encode([term])[0].reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding, embeddings_dict['embeddings'])[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    seen_node_ids = set()
    semantic_matches = []
    
    for idx in top_indices:
        node_id = embeddings_dict['metadata'][idx]['node_id']
        if node_id in seen_node_ids:
            continue
            
        seen_node_ids.add(node_id)
        node_name = graph.nodes[node_id].get('name')
        
        semantic_matches.append({
            'ontology_id': node_id,
            'ontology_term': node_name,
            'similarity': float(similarities[idx]),
            'matched_text': embeddings_dict['metadata'][idx]['text']
        })
    
    return semantic_matches

def traverse_graph(seed_nodes, graph, max_nodes=TOP_K_GRAPH):
    """Traverse the ontology graph to find related concepts."""
    related_nodes = {}
    
    for node in seed_nodes:
        related_nodes[node] = 'seed'
    
    frontier = list(seed_nodes)
    visited = set(seed_nodes)
    
    while frontier and len(related_nodes) < max_nodes:
        current = frontier.pop(0)
        
        for parent in graph.predecessors(current):
            if parent not in visited:
                related_nodes[parent] = 'parent'
                visited.add(parent)
                frontier.append(parent)
                if len(related_nodes) >= max_nodes:
                    break
        
        for child in graph.successors(current):
            if child not in visited:
                related_nodes[child] = 'child'
                visited.add(child)
                frontier.append(child)
                if len(related_nodes) >= max_nodes:
                    break
    
    return related_nodes

def create_enriched_context(graph, term, semantic_matches):
    """Create rich context for the API prompt."""
    context = f"## Context for mapping: '{term}'\n\n"
    
    seed_nodes = [match['ontology_id'] for match in semantic_matches[:5]]
    related_nodes = traverse_graph(seed_nodes, graph)
    
    context += "### Top Semantic Matches:\n"
    for i, match in enumerate(semantic_matches[:5], 1):
        context += f"{i}. {match['ontology_term']} ({match['ontology_id']})\n"
        if match['matched_text'].lower() != match['ontology_term'].lower():
            context += f"   └─ Via: '{match['matched_text']}'\n"
    
    context += "\n### Related Terms:\n"
    for node_id, rel_type in list(related_nodes.items())[:10]:
        if rel_type != 'seed':
            node_data = graph.nodes[node_id]
            name = node_data.get('name', 'Unknown')
            context += f"- {name} ({node_id})\n"
    
    return context

def validate_ontology_mapping(ontology_id, ontology_term, graph, id_to_name):
    """Validate that the LLM output exists in the ontology."""
    if ontology_id not in graph.nodes:
        return False, "ID does not exist in ontology"
    
    actual_term = id_to_name.get(ontology_id, "")
    if actual_term.lower() != ontology_term.lower():
        return False, f"Term mismatch"
    
    return True, "Valid"

def call_llm_with_validation(term, api_key, enriched_context, ontology_name, graph, id_to_name, root_id, root_term):
    """Call LLM to map term with validation and retries."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are an expert in medical terminology and ontologies. Map the following clinical term to the {ontology_name} ontology.

Term to map: "{term}"

Return ONLY a JSON object with 'ontology_id' and 'ontology_term' fields.

Example:
{{
  "ontology_id": "MONDO:0005015",
  "ontology_term": "diabetes mellitus"
}}

{enriched_context}

Return ONLY the JSON object, no additional text."""
    
    for attempt in range(MAX_RETRIES):
        try:
            data = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    ontology_id = result.get('ontology_id', '')
                    ontology_term = result.get('ontology_term', '')
                    
                    is_valid, message = validate_ontology_mapping(ontology_id, ontology_term, graph, id_to_name)
                    
                    if is_valid:
                        return {
                            'ontology_id': ontology_id,
                            'ontology_term': ontology_term,
                            'match_type': 'llm_graphrag'
                        }
                    else:
                        if attempt < MAX_RETRIES - 1:
                            sleep(RETRY_DELAY)
            
            elif response.status_code == 429:
                wait_time = RETRY_DELAY * (attempt + 1)
                sleep(wait_time)
            else:
                if attempt < MAX_RETRIES - 1:
                    sleep(RETRY_DELAY)
        
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY)
    
    # Fallback to root
    return {
        'ontology_id': root_id,
        'ontology_term': root_term,
        'match_type': 'fallback_root'
    }

def map_single_term(term, graph, name_to_id, synonym_to_id, id_to_name, embeddings_dict, 
                   api_key, ontology_name, root_id, root_term):
    """Map a single term using exact match first, then Graph RAG."""
    exact_match = exact_match_term(term, name_to_id, synonym_to_id, id_to_name)
    if exact_match:
        return exact_match
    
    semantic_matches = find_semantic_matches(term, embeddings_dict, graph)
    enriched_context = create_enriched_context(graph, term, semantic_matches)
    
    result = call_llm_with_validation(term, api_key, enriched_context, ontology_name, 
                                     graph, id_to_name, root_id, root_term)
    
    return result

def map_disease_phenotype_terms(api_key, limit=None):
    """Map Disease and Phenotype terms to MONDO and HPO with validation."""
    print("\n" + "="*80)
    print("STEP 3: MAPPING DISEASE & PHENOTYPE TERMS")
    print("="*80)
    
    # Download ontologies
    download_ontology_file(HPO_OBO_URL, HPO_OBO_FILE, "HPO")
    download_ontology_file(MONDO_OBO_URL, MONDO_OBO_FILE, "MONDO")
    
    # Build graphs
    hpo_graph, hpo_name_to_id, hpo_synonym_to_id, hpo_id_to_name = build_ontology_graph(HPO_OBO_FILE, "HPO")
    mondo_graph, mondo_name_to_id, mondo_synonym_to_id, mondo_id_to_name = build_ontology_graph(MONDO_OBO_FILE, "MONDO")
    
    hpo_root_term = hpo_id_to_name.get(HPO_ROOT, "All")
    mondo_root_term = mondo_id_to_name.get(MONDO_ROOT, "disease or disorder")
    
    # Compute embeddings
    hpo_embeddings = compute_ontology_embeddings(hpo_graph, HPO_EMBEDDINGS_FILE, "HPO")
    mondo_embeddings = compute_ontology_embeddings(mondo_graph, MONDO_EMBEDDINGS_FILE, "MONDO")
    
    # Load classified data
    print(f"\nLoading {IC_CLASSIFIED_CSV}...")
    df = pd.read_csv(IC_CLASSIFIED_CSV)
    
    # Filter to Disease and Phenotype
    df_filtered = df[df['class'].isin(['Disease', 'Phenotype'])].copy()
    print(f"Filtered to {len(df_filtered)} Disease/Phenotype rows")
    
    # Get unique terms
    disease_terms = df_filtered[df_filtered['class'] == 'Disease']['indication_contraindication_name'].unique()
    phenotype_terms = df_filtered[df_filtered['class'] == 'Phenotype']['indication_contraindication_name'].unique()
    
    print(f"Disease terms: {len(disease_terms)}")
    print(f"Phenotype terms: {len(phenotype_terms)}")
    
    if limit is not None:
        disease_terms = disease_terms[:limit]
        phenotype_terms = phenotype_terms[:limit]
        print(f"Limited to {len(disease_terms)} + {len(phenotype_terms)} terms")
    
    # Load progress
    progress = {}
    if os.path.exists(MAPPING_PROGRESS):
        with open(MAPPING_PROGRESS, 'r') as f:
            progress = json.load(f)
        print(f"Loaded {len(progress)} previously mapped terms")
    
    # Map Disease terms to MONDO
    print("\nMapping Disease → MONDO...")
    for term in tqdm(disease_terms, desc="Disease → MONDO"):
        progress_key = f"{term}||Disease"
        
        if progress_key in progress:
            continue
        
        result = map_single_term(
            term, mondo_graph, mondo_name_to_id, mondo_synonym_to_id, mondo_id_to_name,
            mondo_embeddings, api_key, "MONDO", MONDO_ROOT, mondo_root_term
        )
        
        progress[progress_key] = result
        
        with open(MAPPING_PROGRESS, 'w') as f:
            json.dump(progress, f, indent=2)
        
        sleep(0.5)
    
    # Map Phenotype terms to HPO
    print("\nMapping Phenotype → HPO...")
    for term in tqdm(phenotype_terms, desc="Phenotype → HPO"):
        progress_key = f"{term}||Phenotype"
        
        if progress_key in progress:
            continue
        
        result = map_single_term(
            term, hpo_graph, hpo_name_to_id, hpo_synonym_to_id, hpo_id_to_name,
            hpo_embeddings, api_key, "HPO", HPO_ROOT, hpo_root_term
        )
        
        progress[progress_key] = result
        
        with open(MAPPING_PROGRESS, 'w') as f:
            json.dump(progress, f, indent=2)
        
        sleep(0.5)
    
    # Apply mappings
    print("\nApplying mappings to dataframe...")
    
    def get_mapping(row):
        progress_key = f"{row['indication_contraindication_name']}||{row['class']}"
        return progress.get(progress_key, {})
    
    df_filtered['ontology_id'] = df_filtered.apply(
        lambda row: get_mapping(row).get('ontology_id', ''), axis=1
    )
    df_filtered['ontology_term'] = df_filtered.apply(
        lambda row: get_mapping(row).get('ontology_term', ''), axis=1
    )
    df_filtered['match_type'] = df_filtered.apply(
        lambda row: get_mapping(row).get('match_type', ''), axis=1
    )
    
    # Save output
    df_filtered.to_csv(IC_DISEASE_PHENOTYPE_CSV, index=False)
    print(f"Saved {IC_DISEASE_PHENOTYPE_CSV}")
    
    # Clean up progress
    if os.path.exists(MAPPING_PROGRESS):
        os.remove(MAPPING_PROGRESS)
    
    # Print statistics
    print("\nMapping Statistics:")
    print(df_filtered['match_type'].value_counts())
    
    return df_filtered

# ============================================================================
# STEP 4: MAP DRUG/CHEMICAL TERMS TO RXNORM WITH SELF-REFERENCE CLEANING
# ============================================================================

def load_ner_model():
    """Load the Hugging Face BioBERT chemical NER model."""
    try:
        print(f"Loading NER model ({NER_MODEL})...")
        ner_pipeline = hf_pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
        print("Model loaded successfully")
        return ner_pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Install required packages: pip install transformers torch")
        raise

def extract_drug_entities(text, ner_pipeline):
    """Extract drug/chemical entities from text using NER."""
    try:
        entities = ner_pipeline(text)
        
        entity_texts = []
        for entity in entities:
            entity_text = entity['word'].strip().replace('##', '')
            if len(entity_text) > 2:
                entity_texts.append(entity_text)
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entity_texts:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity)
        
        entity_texts = unique_entities
        
    except Exception as e:
        entity_texts = []
    
    # Fallback heuristics if no entities found
    if not entity_texts:
        cleaned_text = text.lower()
        remove_phrases = [
            'coadministration with', 'use with', 'concomitant use of',
            'avoid use with', 'concurrent use with', 'in combination with'
        ]
        
        for phrase in remove_phrases:
            cleaned_text = cleaned_text.replace(phrase, '')
        
        cleaned_text = cleaned_text.strip()
        parts = re.split(r',|\sand\s|\sor\s', cleaned_text)
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 2:
                entity_texts.append(part)
    
    return entity_texts

def is_ingredient(rxcui):
    """Check if an RxCUI represents an ingredient (TTY = IN)."""
    url = f"{RXNAV_BASE_URL}/rxcui/{rxcui}/property.json"
    params = {'propName': 'TTY'}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'propConceptGroup' in data and 'propConcept' in data['propConceptGroup']:
                prop_concepts = data['propConceptGroup']['propConcept']
                if prop_concepts and len(prop_concepts) > 0:
                    tty = prop_concepts[0].get('propValue', '')
                    sleep(0.1)
                    return tty == 'IN'
    except Exception as e:
        pass
    
    return False

def query_rxnav_exact(drug_name):
    """Query RxNav API for exact drug name match."""
    url = f"{RXNAV_BASE_URL}/rxcui.json"
    params = {'name': drug_name}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                rxcuis = data['idGroup']['rxnormId']
                if rxcuis and len(rxcuis) > 0:
                    for rxcui in rxcuis:
                        if is_ingredient(rxcui):
                            return {'rxcui': rxcui, 'name': drug_name}
    except Exception as e:
        pass
    
    return None

def query_rxnav_approximate(drug_name):
    """Query RxNav API for approximate drug name match."""
    url = f"{RXNAV_BASE_URL}/approximateTerm.json"
    params = {'term': drug_name, 'maxEntries': 10}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'approximateGroup' in data and 'candidate' in data['approximateGroup']:
                candidates = data['approximateGroup']['candidate']
                if candidates and len(candidates) > 0:
                    for candidate in candidates:
                        rxcui = candidate.get('rxcui')
                        if rxcui and is_ingredient(rxcui):
                            return {
                                'rxcui': rxcui,
                                'name': candidate.get('name', drug_name)
                            }
    except Exception as e:
        pass
    
    return None

def lookup_drug_in_rxnav(drug_name):
    """Lookup drug name in RxNav, trying exact then approximate match."""
    result = query_rxnav_exact(drug_name)
    if result:
        return result
    
    result = query_rxnav_approximate(drug_name)
    if result:
        return result
    
    return None

def remove_self_references(rxcuis_list, ingredient_names_list, source_rxcuis, source_ingredients):
    """
    Remove self-references from mapped drugs.
    Returns cleaned rxcuis and ingredient_names lists.
    """
    # Parse source ingredients/rxcuis (case-insensitive)
    source_rxcuis_set = set(item.strip().lower() for item in str(source_rxcuis).split(','))
    source_ingredients_set = set(item.strip().lower() for item in str(source_ingredients).split(','))
    
    # Filter rxcuis
    cleaned_rxcuis = [
        rxcui for rxcui in rxcuis_list 
        if rxcui.lower() not in source_rxcuis_set
    ]
    
    # Filter ingredient names
    cleaned_names = [
        name for name in ingredient_names_list 
        if name.lower() not in source_ingredients_set
    ]
    
    # Return fallback if empty after cleaning
    if not cleaned_rxcuis:
        cleaned_rxcuis = [DRUG_FALLBACK]
    if not cleaned_names:
        cleaned_names = [DRUG_FALLBACK]
    
    return cleaned_rxcuis, cleaned_names

def map_drug_term_with_cleaning(text, ner_pipeline, source_rxcuis, source_ingredients):
    """
    Map a drug/chemical term to RxNorm with self-reference cleaning.
    """
    # Extract entities using NER
    entities = extract_drug_entities(text, ner_pipeline)
    
    if not entities:
        return {
            'rxcuis': [DRUG_FALLBACK],
            'ingredient_names': [DRUG_FALLBACK],
            'match_type': 'no_entities_found'
        }
    
    # Lookup each entity in RxNav
    rxcuis = []
    ingredient_names = []
    
    for entity in entities:
        result = lookup_drug_in_rxnav(entity)
        if result:
            rxcuis.append(result['rxcui'])
            ingredient_names.append(result['name'])
    
    # If no matches found, use fallback
    if not rxcuis:
        return {
            'rxcuis': [DRUG_FALLBACK],
            'ingredient_names': [DRUG_FALLBACK],
            'match_type': 'no_rxnav_match'
        }
    
    # CLEAN SELF-REFERENCES
    cleaned_rxcuis, cleaned_names = remove_self_references(
        rxcuis, ingredient_names, source_rxcuis, source_ingredients
    )
    
    return {
        'rxcuis': cleaned_rxcuis,
        'ingredient_names': cleaned_names,
        'match_type': 'rxnav_match'
    }

def map_drug_chemical_terms(limit=None):
    """Map Drug/Chemical terms to RxNorm with built-in self-reference cleaning."""
    print("\n" + "="*80)
    print("STEP 4: MAPPING DRUG/CHEMICAL TERMS TO RXNORM")
    print("="*80)
    
    # Load NER model
    ner_pipeline = load_ner_model()
    
    # Load classified data
    print(f"\nLoading {IC_CLASSIFIED_CSV}...")
    df = pd.read_csv(IC_CLASSIFIED_CSV)
    
    # Filter to Drug/Chemical
    df_filtered = df[df['class'] == 'Drug/Chemical'].copy()
    print(f"Filtered to {len(df_filtered)} Drug/Chemical rows")
    
    # Get unique terms
    unique_terms = df_filtered['indication_contraindication_name'].unique()
    print(f"Found {len(unique_terms)} unique Drug/Chemical terms")
    
    if limit is not None:
        unique_terms = unique_terms[:limit]
        print(f"Limited to {len(unique_terms)} terms")
    
    # Load progress
    progress = {}
    if os.path.exists(DRUG_MAPPING_PROGRESS):
        with open(DRUG_MAPPING_PROGRESS, 'r') as f:
            progress = json.load(f)
        print(f"Loaded {len(progress)} previously mapped terms")
    
    # Build a lookup for source ingredients/rxcuis per term
    # We need this for self-reference cleaning
    term_to_source = {}
    for _, row in df_filtered.iterrows():
        term = row['indication_contraindication_name']
        if term not in term_to_source:
            term_to_source[term] = {
                'source_rxcuis': row['ingredient_rxcuis'],
                'source_ingredients': row['ingredients']
            }
    
    # Map each term with self-reference cleaning
    print("\nMapping terms to RxNorm with self-reference cleaning...")
    for term in tqdm(unique_terms, desc="Drug/Chemical → RxNorm"):
        if term in progress:
            continue
        
        # Get source data for this term
        source_data = term_to_source.get(term, {
            'source_rxcuis': '',
            'source_ingredients': ''
        })
        
        result = map_drug_term_with_cleaning(
            term, 
            ner_pipeline, 
            source_data['source_rxcuis'],
            source_data['source_ingredients']
        )
        
        progress[term] = result
        
        with open(DRUG_MAPPING_PROGRESS, 'w') as f:
            json.dump(progress, f, indent=2)
    
    # Apply mappings
    print("\nApplying mappings to dataframe...")
    
    def format_rxcuis(term):
        result = progress.get(term, {})
        rxcuis = result.get('rxcuis', [])
        rxcuis = [r for r in rxcuis if r != DRUG_FALLBACK]
        return ', '.join(rxcuis) if rxcuis else DRUG_FALLBACK
    
    def format_ingredient_names(term):
        result = progress.get(term, {})
        names = result.get('ingredient_names', [])
        names = [n for n in names if n != DRUG_FALLBACK]
        return ', '.join(names) if names else DRUG_FALLBACK
    
    df_filtered['rxcuis'] = df_filtered['indication_contraindication_name'].map(format_rxcuis)
    df_filtered['ingredient_names'] = df_filtered['indication_contraindication_name'].map(format_ingredient_names)
    df_filtered['match_type'] = df_filtered['indication_contraindication_name'].map(
        lambda x: progress.get(x, {}).get('match_type', '')
    )
    
    # Save output (already clean!)
    df_filtered.to_csv(IC_DRUG_CHEMICAL_CSV, index=False)
    print(f"Saved {IC_DRUG_CHEMICAL_CSV} (with self-references already cleaned)")
    
    # Clean up progress
    if os.path.exists(DRUG_MAPPING_PROGRESS):
        os.remove(DRUG_MAPPING_PROGRESS)
    
    # Print statistics
    print("\nMapping Statistics:")
    print(f"Total rows: {len(df_filtered)}")
    print("\nMatch type distribution:")
    print(df_filtered['match_type'].value_counts())
    
    fallback_count = (df_filtered['rxcuis'] == DRUG_FALLBACK).sum()
    valid_count = (df_filtered['rxcuis'] != DRUG_FALLBACK).sum()
    
    print(f"\nTerms with fallback '{DRUG_FALLBACK}': {fallback_count}")
    print(f"Terms with valid RxCUI mappings: {valid_count}")
    
    return df_filtered

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_indications_contraindications_pipeline(api_key, limit_classification=None, 
                                               limit_disease_phenotype=None,
                                               limit_drug_chemical=None):
    """
    Run the complete indications & contraindications processing pipeline.
    
    Args:
        api_key: OpenRouter API key
        limit_classification: Limit for classification (None for all)
        limit_disease_phenotype: Limit for disease/phenotype mapping (None for all)
        limit_drug_chemical: Limit for drug/chemical mapping (None for all)
    """
    
    print("="*80)
    print("SIDEKICK INDICATIONS & CONTRAINDICATIONS PROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Extract
    create_indications_contraindications_csv()
    
    # Step 2: Classify
    classify_indications_contraindications(api_key, limit=limit_classification)
    
    # Delete intermediate CSV
    if os.path.exists(IC_CSV):
        os.remove(IC_CSV)
        print(f"\nDeleted intermediate file: {IC_CSV}")
        print("(All data is preserved in classified CSV)")
    
    # Step 3: Map Disease/Phenotype
    map_disease_phenotype_terms(api_key, limit=limit_disease_phenotype)
    
    # Step 4: Map Drug/Chemical
    map_drug_chemical_terms(limit=limit_drug_chemical)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nFinal Output Files:")
    print(f"  1. {IC_CLASSIFIED_CSV} (all I&C with classifications)")
    print(f"  2. {IC_DISEASE_PHENOTYPE_CSV} (Disease/Phenotype mapped to MONDO/HPO)")
    print(f"  3. {IC_DRUG_CHEMICAL_CSV} (Drug/Chemical mapped to RxNorm, self-references cleaned)")
    print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Get API key
    print("Please enter your OpenRouter API key:")
    api_key = getpass()
    
    # Run full pipeline
    # Set limits to None for production (process all terms)
    # Set limits to small numbers for testing
    run_indications_contraindications_pipeline(
        api_key=api_key,
        limit_classification=None,      # None = all terms
        limit_disease_phenotype=None,   # None = all terms
        limit_drug_chemical=None        # None = all terms
    )
    
    print("\n✓ Indications & Contraindications processing complete!")
