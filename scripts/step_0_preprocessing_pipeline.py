"""
SIDEKICK Data Preprocessing Pipeline
Combines human drug filtering, SPL extraction, deduplication, and cleanup
"""

import pandas as pd
import requests
import json
import csv
import os
import zipfile
import tempfile
import shutil
from tqdm.auto import tqdm
from urllib.parse import urlparse
import time
import hashlib
from difflib import SequenceMatcher
from collections import defaultdict
import glob
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import io

# ============================================================================
# CONSTANTS
# ============================================================================

DAILYMED_FILES = [
    "dm_spl_release_human_rx_part1.zip",
    "dm_spl_release_human_rx_part2.zip",
    "dm_spl_release_human_rx_part3.zip",
    "dm_spl_release_human_rx_part4.zip",
    "dm_spl_release_human_rx_part5.zip"
]

# ============================================================================
# STEP 1: EXTRACT HUMAN SET IDs FROM DAILYMED ARCHIVES
# ============================================================================

def extract_human_setids_from_local_files(zip_files_dir=".", cleanup=True):
    """
    Extract all SET IDs from local DailyMed human prescription label zip files
    
    Args:
        zip_files_dir: Directory containing the zip files
        cleanup: Whether to delete extracted directories after processing
        
    Returns:
        Set of human drug SET IDs
    """
    human_setids = set()
    
    print(f"Looking for zip files in: {os.path.abspath(zip_files_dir)}")
    
    # Check which files exist
    available_files = []
    for filename in DAILYMED_FILES:
        zip_path = os.path.join(zip_files_dir, filename)
        if os.path.exists(zip_path):
            available_files.append((filename, zip_path))
            print(f"Found: {filename}")
        else:
            print(f"Missing: {filename}")
    
    if not available_files:
        print("No DailyMed zip files found!")
        print(f"Please ensure the zip files are in: {os.path.abspath(zip_files_dir)}")
        return set()
    
    print(f"\nProcessing {len(available_files)} zip files...")
    
    try:
        # Process each available zip file
        for i, (filename, zip_path) in enumerate(available_files, 1):
            print(f"\nProcessing {filename} ({i}/{len(available_files)})...")
            
            extract_dir = os.path.join(zip_files_dir, f"temp_extract_{i}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                print(f"Extracting {filename}...")
                zip_file.extractall(extract_dir)
                
                prescription_dir = os.path.join(extract_dir, "prescription")
                if os.path.exists(prescription_dir):
                    prescription_files = [f for f in os.listdir(prescription_dir) if f.endswith('.zip')]
                    
                    print(f"Found {len(prescription_files)} prescription files in {filename}")
                    
                    if i == 1 and len(prescription_files) > 0:
                        print("Sample filenames for debugging:")
                        for j, sample_file in enumerate(prescription_files[:5]):
                            print(f"  {j+1}: {sample_file}")
                    
                    for prescription_file in prescription_files:
                        base_name = prescription_file.replace('.zip', '')
                        
                        if '_' in base_name:
                            parts = base_name.split('_', 1)
                            if len(parts) >= 2:
                                setid = parts[1]
                            else:
                                setid = base_name
                        else:
                            setid = base_name
                        
                        setid = setid.strip().lower()
                        
                        if len(setid) > 0:
                            human_setids.add(setid)
                    
                    if i == 1 and len(human_setids) > 0:
                        sample_setids = list(human_setids)[:3]
                        print(f"Sample SET IDs extracted: {sample_setids}")
                        
                else:
                    print(f"Warning: No prescription folder found in {filename}")
            
            if cleanup:
                shutil.rmtree(extract_dir, ignore_errors=True)
                
            print(f"Processed {filename}, total unique SET IDs so far: {len(human_setids)}")
    
    except Exception as e:
        print(f"Error processing zip files: {e}")
        raise
    
    print(f"\nCompleted! Found {len(human_setids)} unique human drug SET IDs from {len(available_files)} zip files")
    return human_setids

def save_human_setids(setids, filename="human_setids.txt"):
    """Save human SET IDs to a file for future use"""
    with open(filename, 'w') as f:
        for setid in sorted(setids):
            f.write(f"{setid}\n")
    print(f"Saved {len(setids)} SET IDs to {filename}")

def load_human_setids(filename="human_setids.txt"):
    """Load human SET IDs from a file"""
    if not os.path.exists(filename):
        return None
    
    setids = set()
    with open(filename, 'r') as f:
        for line in f:
            setid = line.strip().lower()
            if setid:
                setids.add(setid)
    
    print(f"Loaded {len(setids)} SET IDs from {filename}")
    return setids

# ============================================================================
# STEP 2: CREATE HUMAN DRUG CSV WITH RXNORM MAPPING
# ============================================================================

def get_ingredients(rxcui):
    """Get active ingredients for a product RxCUI using RxNorm API"""
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        ing_names = []
        ing_rxcuis = []
        
        if 'allRelatedGroup' in data and 'conceptGroup' in data['allRelatedGroup']:
            for group in data['allRelatedGroup']['conceptGroup']:
                if group.get('tty') == 'IN' and 'conceptProperties' in group:
                    for prop in group['conceptProperties']:
                        ing_names.append(prop['name'])
                        ing_rxcuis.append(prop['rxcui'])
        
        return ing_names, ing_rxcuis
    except Exception as e:
        print(f"Error getting ingredients for RxCUI {rxcui}: {e}")
        return [], []

def parse_mapping_file(file_path=None, sample_data=None):
    """Parse the FDA RxNorm mapping file"""
    if file_path and os.path.exists(file_path):
        print(f"Reading mapping file: {file_path}")
        data = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 5:
                    data.append({
                        'SETID': parts[0],
                        'SPL_VERSION': parts[1],
                        'RXCUI': parts[2],
                        'RXSTRING': parts[3],
                        'RXTTY': parts[4]
                    })
        return data
    elif sample_data:
        print("Using provided sample data")
        data = []
        for line in sample_data.strip().split('\n'):
            parts = line.strip().split('|')
            if len(parts) >= 5 and parts[0] != 'SETID':
                data.append({
                    'SETID': parts[0],
                    'SPL_VERSION': parts[1],
                    'RXCUI': parts[2],
                    'RXSTRING': parts[3],
                    'RXTTY': parts[4]
                })
        return data
    else:
        print("No mapping file or sample data provided")
        return []

def filter_mapping_data_by_human_setids(mapping_data, human_setids):
    """Filter mapping data to only include entries with human SET IDs"""
    print(f"Filtering mapping data using {len(human_setids)} human SET IDs...")
    
    original_count = len(mapping_data)
    filtered_data = []
    
    for item in mapping_data:
        setid = item['SETID'].strip().lower()
        if setid in human_setids:
            filtered_data.append(item)
    
    filtered_count = len(filtered_data)
    print(f"Filtered mapping data: {original_count} -> {filtered_count} entries ({filtered_count/original_count*100:.1f}% retained)")
    
    return filtered_data

def create_human_product_ingredient_csv(mapping_data, human_setids, output_file=None, limit=None):
    """
    Process mapping data to create product-ingredient CSV for human drugs only
    """
    filtered_mapping_data = filter_mapping_data_by_human_setids(mapping_data, human_setids)
    
    if not filtered_mapping_data:
        print("No human drugs found in mapping data after filtering")
        return pd.DataFrame()
    
    products = {}
    for item in filtered_mapping_data:
        rxcui = item['RXCUI']
        setid = item['SETID']
        spl_version = item['SPL_VERSION']
        
        product_key = (rxcui, setid)
        
        if product_key not in products:
            products[product_key] = {
                'rxcui': rxcui,
                'setid': setid,
                'spl_version': spl_version,
                'psn_name': ''
            }
        
        if item['RXTTY'] == 'PSN':
            products[product_key]['psn_name'] = item['RXSTRING']
        elif item['RXTTY'] == 'SCD' and not products[product_key]['psn_name']:
            products[product_key]['psn_name'] = item['RXSTRING']
        elif item['RXTTY'] == 'SBD' and not products[product_key]['psn_name']:
            products[product_key]['psn_name'] = item['RXSTRING']
    
    print(f"Found {len(products)} unique human drug (RXCUI, SETID) combinations")
    
    product_keys = list(products.keys())
    if limit and limit < len(product_keys):
        print(f"Limiting to {limit} products")
        product_keys = product_keys[:limit]
    
    unique_rxcuis = list(set([rxcui for rxcui, setid in product_keys]))
    print(f"Fetching ingredients for {len(unique_rxcuis)} unique human drug RXCUIs")
    
    ingredient_cache = {}
    
    for rxcui in tqdm(unique_rxcuis, desc="Fetching ingredient data for human drugs"):
        for product_key in product_keys:
            if product_key[0] == rxcui and not products[product_key]['psn_name']:
                products[product_key]['psn_name'] = f"Product {rxcui}"
        
        ing_names, ing_rxcuis = get_ingredients(rxcui)
        ingredient_cache[rxcui] = (ing_names, ing_rxcuis)
    
    print("Building final results...")
    results = []
    
    for product_key in tqdm(product_keys, desc="Building results"):
        rxcui, setid = product_key
        product = products[product_key]
        
        ing_names, ing_rxcuis = ingredient_cache[rxcui]
        
        results.append({
            'product_rxcui': rxcui,
            'product_name': product['psn_name'],
            'ingredients': ','.join(ing_names),
            'ingredient_rxcuis': ','.join(ing_rxcuis),
            'set_id': product['setid'],
            'spl_version': product['spl_version'],
            'is_human_drug': True
        })
    
    df = pd.DataFrame(results)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"CSV file created: {output_file}")
    
    print(f"Final CSV contains {len(df)} rows with human drugs only")
    
    return df

# ============================================================================
# STEP 3: EXTRACT SPL XML FILES FROM ARCHIVES
# ============================================================================

def extract_version_from_xml(xml_content):
    """Extract version number from SPL XML content"""
    try:
        root = ET.fromstring(xml_content)
        
        namespaces = {'': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
        
        found_version = None
        
        version_elements = root.findall(".//versionNumber", namespaces)
        if version_elements:
            for version_element in version_elements:
                if 'value' in version_element.attrib:
                    found_version = version_element.attrib['value']
                    break
        
        if not found_version:
            version_elements = root.findall(".//versionNumber")
            if version_elements:
                for version_element in version_elements:
                    if 'value' in version_element.attrib:
                        found_version = version_element.attrib['value']
                        break
        
        if not found_version:
            found_version = "1"
            
        return found_version
        
    except Exception as e:
        print(f"Error parsing XML for version: {str(e)}")
        return "1"

def build_setid_index(dailymed_zip_dir="."):
    """Build an index of all available set_ids and their locations"""
    setid_index = {}
    
    print("Building index of SET IDs from DailyMed archives...")
    
    for filename in DAILYMED_FILES:
        zip_path = os.path.join(dailymed_zip_dir, filename)
        if not os.path.exists(zip_path):
            print(f"Warning: {filename} not found in {dailymed_zip_dir}")
            continue
            
        print(f"Indexing {filename}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as main_zip:
                prescription_files = [f for f in main_zip.namelist() 
                                    if f.startswith('prescription/') and f.endswith('.zip')]
                
                print(f"Found {len(prescription_files)} prescription files in {filename}")
                
                for prescription_zip in prescription_files:
                    base_filename = os.path.basename(prescription_zip)
                    if '_' in base_filename:
                        base_name = base_filename.replace('.zip', '')
                        parts = base_name.split('_', 1)
                        if len(parts) >= 2:
                            setid = parts[1].strip().lower()
                            
                            if setid not in setid_index:
                                setid_index[setid] = (zip_path, prescription_zip)
                                
        except Exception as e:
            print(f"Error indexing {filename}: {str(e)}")
            continue
    
    print(f"Built index with {len(setid_index)} unique SET IDs")
    return setid_index

def extract_spl_from_local_archives(csv_file, output_dir="data/spls", dailymed_zip_dir=".", limit=None):
    """Extract SPL XML files from local DailyMed archives based on set_ids in CSV"""
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
    df = pd.read_csv(csv_file)
    
    if 'set_id' not in df.columns:
        raise ValueError("CSV must contain 'set_id' column")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    unique_set_ids = df['set_id'].unique()
    unique_set_ids = [str(setid).strip().lower() for setid in unique_set_ids]
    
    if limit is not None:
        unique_set_ids = unique_set_ids[:limit]
        
    print(f"Need to extract {len(unique_set_ids)} unique SET IDs from CSV")
    
    setid_index = build_setid_index(dailymed_zip_dir)
    
    results = {
        "successful": [],
        "failed": [],
        "not_found": []
    }
    
    print(f"\nExtracting XML files...")
    
    for setid in tqdm(unique_set_ids, desc="Extracting SPL files"):
        try:
            if setid not in setid_index:
                results["not_found"].append({"set_id": setid, "reason": "SET ID not found in archives"})
                continue
                
            main_zip_path, prescription_zip_path = setid_index[setid]
            
            with zipfile.ZipFile(main_zip_path, 'r') as main_zip:
                with main_zip.open(prescription_zip_path) as prescription_zip_data:
                    prescription_zip_content = prescription_zip_data.read()
                    
                    with zipfile.ZipFile(io.BytesIO(prescription_zip_content), 'r') as prescription_zip:
                        xml_files = [f for f in prescription_zip.namelist() if f.endswith('.xml')]
                        
                        if not xml_files:
                            results["failed"].append({"set_id": setid, "reason": "No XML file in prescription zip"})
                            continue
                            
                        xml_filename = xml_files[0]
                        
                        with prescription_zip.open(xml_filename) as xml_file:
                            xml_content = xml_file.read()
                            
                            version = extract_version_from_xml(xml_content)
                            
                            output_filename = f"{setid}_{version}.xml"
                            output_path = os.path.join(output_dir, output_filename)
                            
                            if os.path.exists(output_path):
                                results["successful"].append({
                                    "set_id": setid,
                                    "version": version,
                                    "file_path": output_path,
                                    "status": "already_existed"
                                })
                                continue
                            
                            with open(output_path, 'wb') as f:
                                f.write(xml_content)
                            
                            results["successful"].append({
                                "set_id": setid,
                                "version": version, 
                                "file_path": output_path,
                                "status": "extracted"
                            })
                            
        except Exception as e:
            results["failed"].append({"set_id": setid, "reason": str(e)})
            continue
    
    log_file = os.path.join(output_dir, "extraction_log.json")
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nExtraction complete:")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Not found in archives: {len(results['not_found'])}")
    print(f"  Log saved to: {log_file}")
    
    return results

# ============================================================================
# STEP 4: DEDUPLICATE BY ADVERSE REACTIONS
# ============================================================================

def extract_adverse_reactions_section(xml_file_path):
    """Extract only the adverse reactions section (code '34084-4') from SPL XML file"""
    try:
        with open(xml_file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'xml')
        
        file_name = os.path.basename(xml_file_path)
        set_id = version = None
        if '_' in file_name:
            parts = file_name.replace('.xml', '').split('_')
            if len(parts) >= 2:
                set_id = parts[0]
                version = parts[1]
        
        adverse_reactions_text = ""
        has_adverse_reactions = False
        
        for section in soup.find_all('section'):
            code_elem = section.find('code')
            
            if code_elem and code_elem.get('code') == '34084-4':
                has_adverse_reactions = True
                
                section_title = ""
                if section.find('title'):
                    section_title = section.find('title').text.strip()
                elif code_elem.get('displayName'):
                    section_title = code_elem.get('displayName').strip()
                
                text_parts = []
                if section_title:
                    text_parts.append(section_title)
                    text_parts.append("=" * len(section_title))
                
                for table in section.find_all('table'):
                    text_parts.append("\nTABLE:")
                    
                    for row in table.find_all('tr'):
                        cells = []
                        for cell in row.find_all(['th', 'td']):
                            cell_text = cell.get_text().strip().replace('\n', ' ')
                            cells.append(cell_text)
                        
                        if cells:
                            text_parts.append(" | ".join(cells))
                    
                    text_parts.append("")
                    table.extract()
                
                section_text = section.get_text().strip()
                if section_text:
                    lines = []
                    current_line = ""
                    
                    for line in section_text.split('\n'):
                        line = line.strip()
                        if line:
                            if current_line:
                                current_line += " " + line
                            else:
                                current_line = line
                        else:
                            if current_line:
                                lines.append(current_line)
                                current_line = ""
                    
                    if current_line:
                        lines.append(current_line)
                    
                    text_parts.extend(lines)
                
                adverse_reactions_text = "\n".join(text_parts)
                break
        
        while "\n\n\n" in adverse_reactions_text:
            adverse_reactions_text = adverse_reactions_text.replace("\n\n\n", "\n\n")
        
        return adverse_reactions_text, has_adverse_reactions, set_id, version
    
    except Exception as e:
        return f"ERROR: {str(e)}", False, None, None

def get_text_hash(text):
    """Generate MD5 hash of text for fast exact matching"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def calculate_text_similarity(text1, text2):
    """Calculate similarity ratio between two texts using SequenceMatcher"""
    return SequenceMatcher(None, text1, text2).ratio()

def find_spl_file_path(set_id, version, spls_dir="data/spls"):
    """Find the SPL file path for a given set_id and version"""
    expected_filename = f"{set_id}_{version}.xml"
    expected_path = os.path.join(spls_dir, expected_filename)
    
    if os.path.exists(expected_path):
        return expected_path
    
    fallback_filename = f"{set_id}.xml"
    fallback_path = os.path.join(spls_dir, fallback_filename)
    
    if os.path.exists(fallback_path):
        return fallback_path
    
    for filename in os.listdir(spls_dir):
        if set_id.lower() in filename.lower() and filename.endswith('.xml'):
            return os.path.join(spls_dir, filename)
    
    return None

def deduplicate_by_adverse_reactions(csv_file, spls_dir="data/spls", output_file="unique_setids_adverse_reactions.csv", 
                                   similarity_threshold=0.95, save_analysis=True):
    """
    Deduplicate set IDs within each product RxCUI group based on adverse reactions section text
    """
    
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if 'product_rxcui' not in df.columns or 'set_id' not in df.columns:
        raise ValueError("CSV must contain 'product_rxcui' and 'set_id' columns")
    
    print(f"Original CSV has {len(df)} rows with {df['product_rxcui'].nunique()} unique product RxCUIs")
    
    rxcui_groups = df.groupby('product_rxcui')
    print(f"Found {len(rxcui_groups)} product RxCUI groups")
    
    stats = {
        'total_rxcuis': len(rxcui_groups),
        'total_original_setids': len(df),
        'total_unique_setids': 0,
        'setids_without_adverse_reactions': 0,
        'setids_with_adverse_reactions': 0,
        'rxcuis_with_duplicates': 0,
        'setids_removed': 0,
        'processing_errors': 0
    }
    
    unique_setids = []
    analysis_log = []
    
    print("Processing product RxCUI groups (focusing on adverse reactions section)...")
    
    for rxcui, group in tqdm(rxcui_groups, desc="Deduplicating by adverse reactions"):
        group_setids = group['set_id'].unique()
        
        if len(group_setids) == 1:
            row = group.iloc[0].copy()
            row['is_unique'] = True
            row['duplicate_count'] = 1
            row['has_adverse_reactions'] = 'unknown'
            row['similarity_group'] = 0
            unique_setids.append(row)
            stats['total_unique_setids'] += 1
            continue
        
        setid_adverse_reactions = {}
        setids_without_adverse_reactions = []
        
        for set_id in group_setids:
            row = group[group['set_id'] == set_id].iloc[0]
            version = row.get('spl_version', '1')
            
            spl_path = find_spl_file_path(set_id, version, spls_dir)
            
            if spl_path is None:
                print(f"Warning: SPL file not found for set_id {set_id}")
                stats['processing_errors'] += 1
                continue
            
            adverse_text, has_adverse_reactions, _, _ = extract_adverse_reactions_section(spl_path)
            
            if adverse_text.startswith("ERROR:"):
                print(f"Warning: Could not extract adverse reactions for set_id {set_id}: {adverse_text}")
                stats['processing_errors'] += 1
                continue
            
            if not has_adverse_reactions:
                setids_without_adverse_reactions.append(set_id)
                stats['setids_without_adverse_reactions'] += 1
            else:
                setid_adverse_reactions[set_id] = {
                    'text': adverse_text,
                    'hash': get_text_hash(adverse_text),
                    'length': len(adverse_text),
                    'row': row
                }
                stats['setids_with_adverse_reactions'] += 1
        
        for set_id in setids_without_adverse_reactions:
            row = group[group['set_id'] == set_id].iloc[0].copy()
            row['is_unique'] = True
            row['duplicate_count'] = 1
            row['has_adverse_reactions'] = False
            row['similarity_group'] = -1
            unique_setids.append(row)
            stats['total_unique_setids'] += 1
            
            analysis_log.append({
                'product_rxcui': rxcui,
                'set_id': set_id,
                'has_adverse_reactions': False,
                'is_representative': True,
                'duplicate_of': None,
                'text_length': 0
            })
        
        if not setid_adverse_reactions:
            continue
        
        text_groups = []
        processed_setids = set()
        
        for set_id, data in setid_adverse_reactions.items():
            if set_id in processed_setids:
                continue
            
            current_group = [set_id]
            processed_setids.add(set_id)
            
            for other_setid, other_data in setid_adverse_reactions.items():
                if other_setid in processed_setids:
                    continue
                
                if data['hash'] == other_data['hash']:
                    current_group.append(other_setid)
                    processed_setids.add(other_setid)
                    continue
                
                similarity = calculate_text_similarity(data['text'], other_data['text'])
                if similarity >= similarity_threshold:
                    current_group.append(other_setid)
                    processed_setids.add(other_setid)
            
            text_groups.append(current_group)
        
        group_has_duplicates = len(text_groups) < len(setid_adverse_reactions)
        if group_has_duplicates:
            stats['rxcuis_with_duplicates'] += 1
        
        for i, similarity_group in enumerate(text_groups):
            representative_setid = similarity_group[0]
            representative_row = setid_adverse_reactions[representative_setid]['row'].copy()
            
            representative_row['is_unique'] = True
            representative_row['duplicate_count'] = len(similarity_group)
            representative_row['has_adverse_reactions'] = True
            representative_row['similarity_group'] = i
            representative_row['duplicate_setids'] = ','.join(similarity_group[1:]) if len(similarity_group) > 1 else ''
            
            unique_setids.append(representative_row)
            stats['total_unique_setids'] += 1
            
            for j, setid in enumerate(similarity_group):
                analysis_log.append({
                    'product_rxcui': rxcui,
                    'set_id': setid,
                    'has_adverse_reactions': True,
                    'is_representative': (j == 0),
                    'similarity_group': i,
                    'text_length': setid_adverse_reactions[setid]['length'],
                    'text_hash': setid_adverse_reactions[setid]['hash'][:16],
                    'duplicate_of': representative_setid if j > 0 else None
                })
        
        stats['setids_removed'] += (len(setid_adverse_reactions) - len(text_groups))
    
    results_df = pd.DataFrame(unique_setids)
    
    stats['deduplication_ratio'] = stats['total_unique_setids'] / stats['total_original_setids']
    stats['setids_saved'] = stats['total_original_setids'] - stats['total_unique_setids']
    
    results_df.to_csv(output_file, index=False)
    print(f"\nDeduplicated CSV saved to: {output_file}")
    
    if save_analysis and analysis_log:
        analysis_file = output_file.replace('.csv', '_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump({
                'statistics': stats,
                'analysis': analysis_log
            }, f, indent=2)
        print(f"Analysis log saved to: {analysis_file}")
    
    print(f"\n=== ADVERSE REACTIONS DEDUPLICATION SUMMARY ===")
    print(f"Original set IDs: {stats['total_original_setids']:,}")
    print(f"Unique set IDs: {stats['total_unique_setids']:,}")
    print(f"Set IDs removed: {stats['setids_removed']:,}")
    print(f"Deduplication ratio: {stats['deduplication_ratio']:.1%}")
    print(f"")
    print(f"Set IDs without adverse reactions (kept as unique): {stats['setids_without_adverse_reactions']:,}")
    print(f"Set IDs with adverse reactions: {stats['setids_with_adverse_reactions']:,}")
    print(f"Product RxCUIs with duplicates: {stats['rxcuis_with_duplicates']:,} / {stats['total_rxcuis']:,}")
    print(f"Processing errors: {stats['processing_errors']:,}")
    
    return results_df, stats

# ============================================================================
# STEP 5: COPY UNIQUE SPL FILES
# ============================================================================

def copy_unique_spl_files(csv_file, source_dir="data/spls", target_dir="data/spls_unique"):
    """Copy unique SPL XML files based on set_ids from the deduplicated CSV"""
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    print(f"Reading unique set IDs from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if 'set_id' not in df.columns:
        raise ValueError("CSV must contain 'set_id' column")
    
    unique_set_ids = df['set_id'].unique()
    print(f"Found {len(unique_set_ids)} unique set IDs in CSV")
    
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory: {os.path.abspath(target_dir)}")
    
    source_xml_files = glob.glob(os.path.join(source_dir, "*.xml"))
    print(f"Found {len(source_xml_files)} XML files in source directory")
    
    setid_to_file = {}
    
    print("Building set_id to file mapping...")
    for xml_file in source_xml_files:
        filename = os.path.basename(xml_file)
        
        if '_' in filename:
            set_id = filename.split('_')[0].lower()
        else:
            set_id = filename.replace('.xml', '').lower()
        
        if set_id not in setid_to_file:
            setid_to_file[set_id] = xml_file
    
    print(f"Built mapping for {len(setid_to_file)} unique set IDs from filenames")
    
    stats = {
        'unique_setids_in_csv': len(unique_set_ids),
        'files_found': 0,
        'files_copied': 0,
        'files_not_found': 0,
        'copy_errors': 0,
        'files_already_existed': 0
    }
    
    not_found_setids = []
    copy_errors = []
    
    print("Copying unique SPL files...")
    
    for set_id in tqdm(unique_set_ids, desc="Copying SPL files"):
        set_id_lower = str(set_id).strip().lower()
        
        if set_id_lower in setid_to_file:
            source_file = setid_to_file[set_id_lower]
            target_file = os.path.join(target_dir, os.path.basename(source_file))
            
            stats['files_found'] += 1
            
            try:
                if os.path.exists(target_file):
                    stats['files_already_existed'] += 1
                    continue
                
                shutil.copy2(source_file, target_file)
                stats['files_copied'] += 1
                
            except Exception as e:
                copy_errors.append({
                    'set_id': set_id,
                    'source_file': source_file,
                    'error': str(e)
                })
                stats['copy_errors'] += 1
        else:
            not_found_setids.append(set_id)
            stats['files_not_found'] += 1
    
    print(f"\n=== COPY SUMMARY ===")
    print(f"Unique set IDs in CSV: {stats['unique_setids_in_csv']:,}")
    print(f"Files found: {stats['files_found']:,}")
    print(f"Files copied: {stats['files_copied']:,}")
    print(f"Files already existed: {stats['files_already_existed']:,}")
    print(f"Files not found: {stats['files_not_found']:,}")
    print(f"Copy errors: {stats['copy_errors']:,}")
    
    if not_found_setids:
        print(f"\nFiles not found for {len(not_found_setids)} set IDs:")
        for set_id in not_found_setids[:10]:
            print(f"  - {set_id}")
        if len(not_found_setids) > 10:
            print(f"  ... and {len(not_found_setids) - 10} more")
    
    if copy_errors:
        print(f"\nCopy errors for {len(copy_errors)} files:")
        for error in copy_errors[:5]:
            print(f"  - {error['set_id']}: {error['error']}")
        if len(copy_errors) > 5:
            print(f"  ... and {len(copy_errors) - 5} more errors")
    
    print(f"\nUnique SPL files copied to: {os.path.abspath(target_dir)}")
    
    return stats

# ============================================================================
# STEP 6: CLEANUP - DELETE OLD AND RENAME UNIQUE
# ============================================================================

def cleanup_and_rename_spl_directory(old_dir="data/spls", unique_dir="data/spls_unique"):
    """
    Delete the old SPL directory with duplicates and rename unique directory
    This makes data/spls contain only the deduplicated files
    """
    print("\n=== CLEANUP AND RENAME ===")
    
    # Check if directories exist
    if not os.path.exists(unique_dir):
        print(f"Error: Unique directory not found: {unique_dir}")
        return False
    
    # Count files in unique directory
    unique_files = glob.glob(os.path.join(unique_dir, "*.xml"))
    print(f"Unique directory contains {len(unique_files)} XML files")
    
    # Delete old directory if it exists
    if os.path.exists(old_dir):
        old_files = glob.glob(os.path.join(old_dir, "*.xml"))
        print(f"Deleting old directory with {len(old_files)} files: {old_dir}")
        shutil.rmtree(old_dir)
        print(f"Deleted: {old_dir}")
    else:
        print(f"Old directory does not exist (already deleted): {old_dir}")
    
    # Rename unique directory to old directory name
    print(f"Renaming {unique_dir} -> {old_dir}")
    os.rename(unique_dir, old_dir)
    
    # Verify
    final_files = glob.glob(os.path.join(old_dir, "*.xml"))
    print(f"\n✓ Cleanup complete!")
    print(f"✓ {old_dir} now contains {len(final_files)} deduplicated SPL files")
    
    return True

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_preprocessing_pipeline(
    dailymed_zip_dir=".",
    rxnorm_mapping_file="rxnorm_mappings.txt",
    use_cached_setids=True,
    setids_file="human_setids.txt",
    human_csv_output="human_product_ingredients.csv",
    spls_output_dir="data/spls",
    unique_csv_output="unique_human_ingredients_adverse_reactions.csv",
    similarity_threshold=0.95,
    limit=None
):
    """
    Run the complete preprocessing pipeline:
    1. Extract human SET IDs from DailyMed archives
    2. Create human drug CSV with RxNorm mapping
    3. Extract SPL XML files
    4. Deduplicate by adverse reactions
    5. Copy unique files
    6. Cleanup: delete old directory and rename unique to final
    
    Args:
        dailymed_zip_dir: Directory containing DailyMed zip files
        rxnorm_mapping_file: Path to RxNorm mapping file
        use_cached_setids: Use cached SET IDs if available
        setids_file: Path to cached SET IDs file
        human_csv_output: Output path for human drugs CSV
        spls_output_dir: Directory to save extracted SPL files
        unique_csv_output: Output path for deduplicated CSV
        similarity_threshold: Similarity threshold for deduplication (0-1)
        limit: Limit number of products (None for all)
    
    Returns:
        dict: Pipeline execution statistics
    """
    
    print("=" * 80)
    print("SIDEKICK DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    pipeline_stats = {}
    
    # STEP 1: Extract human SET IDs
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING HUMAN SET IDs")
    print("=" * 80)
    
    human_setids = None
    if use_cached_setids:
        human_setids = load_human_setids(setids_file)
    
    if human_setids is None:
        print("Extracting SET IDs from DailyMed archives (this may take 5-10 minutes)...")
        human_setids = extract_human_setids_from_local_files(dailymed_zip_dir)
        
        if human_setids:
            save_human_setids(human_setids, setids_file)
        else:
            print("ERROR: Failed to extract human SET IDs")
            return None
    
    pipeline_stats['human_setids_count'] = len(human_setids)
    
    # STEP 2: Create human drug CSV
    print("\n" + "=" * 80)
    print("STEP 2: CREATING HUMAN DRUG CSV WITH RXNORM MAPPING")
    print("=" * 80)
    
    mapping_data = parse_mapping_file(rxnorm_mapping_file)
    
    if not mapping_data:
        print("ERROR: No mapping data found")
        return None
    
    df_human = create_human_product_ingredient_csv(
        mapping_data, 
        human_setids, 
        output_file=human_csv_output,
        limit=limit
    )
    
    pipeline_stats['human_products_count'] = len(df_human)
    
    # STEP 3: Extract SPL XML files
    print("\n" + "=" * 80)
    print("STEP 3: EXTRACTING SPL XML FILES")
    print("=" * 80)
    
    extraction_results = extract_spl_from_local_archives(
        csv_file=human_csv_output,
        output_dir=spls_output_dir,
        dailymed_zip_dir=dailymed_zip_dir,
        limit=limit
    )
    
    pipeline_stats['spls_extracted'] = len(extraction_results['successful'])
    pipeline_stats['spls_failed'] = len(extraction_results['failed'])
    
    # STEP 4: Deduplicate by adverse reactions
    print("\n" + "=" * 80)
    print("STEP 4: DEDUPLICATING BY ADVERSE REACTIONS")
    print("=" * 80)
    
    df_unique, dedup_stats = deduplicate_by_adverse_reactions(
        csv_file=human_csv_output,
        spls_dir=spls_output_dir,
        output_file=unique_csv_output,
        similarity_threshold=similarity_threshold,
        save_analysis=True
    )
    
    pipeline_stats['deduplication'] = dedup_stats
    
    # STEP 5: Copy unique files
    print("\n" + "=" * 80)
    print("STEP 5: COPYING UNIQUE SPL FILES")
    print("=" * 80)
    
    copy_stats = copy_unique_spl_files(
        csv_file=unique_csv_output,
        source_dir=spls_output_dir,
        target_dir="data/spls_unique"
    )
    
    pipeline_stats['copy'] = copy_stats
    
    # STEP 6: Cleanup and rename
    print("\n" + "=" * 80)
    print("STEP 6: CLEANUP - DELETE OLD SPL DIRECTORY AND RENAME UNIQUE")
    print("=" * 80)
    
    cleanup_success = cleanup_and_rename_spl_directory(
        old_dir=spls_output_dir,
        unique_dir="data/spls_unique"
    )
    
    pipeline_stats['cleanup_success'] = cleanup_success
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"✓ Human SET IDs extracted: {pipeline_stats['human_setids_count']:,}")
    print(f"✓ Human products in CSV: {pipeline_stats['human_products_count']:,}")
    print(f"✓ SPL files extracted: {pipeline_stats['spls_extracted']:,}")
    print(f"✓ Unique SET IDs after deduplication: {dedup_stats['total_unique_setids']:,}")
    print(f"✓ Deduplication ratio: {dedup_stats['deduplication_ratio']:.1%}")
    print(f"✓ Final SPL files in {spls_output_dir}: {copy_stats['files_copied']:,}")
    print(f"✓ Cleanup successful: {cleanup_success}")
    
    print("\n" + "=" * 80)
    print("OUTPUT FILES:")
    print("=" * 80)
    print(f"  1. {setids_file} - Cached human SET IDs")
    print(f"  2. {human_csv_output} - All human products")
    print(f"  3. {unique_csv_output} - Deduplicated products")
    print(f"  4. {spls_output_dir}/ - Final deduplicated SPL XML files")
    print("=" * 80)
    
    return pipeline_stats

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the complete preprocessing pipeline
    # Make sure the 5 DailyMed zip files are in the current directory
    
    stats = run_preprocessing_pipeline(
        dailymed_zip_dir=".",  # Directory containing DailyMed zip files
        rxnorm_mapping_file="rxnorm_mappings.txt",  # RxNorm mapping file
        use_cached_setids=True,  # Use cached SET IDs if available
        setids_file="human_setids.txt",  # Cache file for SET IDs
        human_csv_output="human_product_ingredients.csv",  # Output CSV
        spls_output_dir="data/spls",  # Final output directory (will contain unique files only)
        unique_csv_output="unique_human_ingredients_adverse_reactions.csv",  # Deduplicated CSV
        similarity_threshold=0.95,  # Similarity threshold for deduplication
        limit=None  # Set to a number for testing, None for full processing
    )
    
    print("\nPipeline statistics:")
    print(json.dumps(stats, indent=2))
