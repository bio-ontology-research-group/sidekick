import os
import csv
import re
import pandas as pd

def extract_side_effects_from_mapped_file(file_path):
    """
    Extract side effects with HPO mappings from a mapped file.
    
    Args:
        file_path: Path to the mapped file
    
    Returns:
        Tuple of (set_id, side_effects_list)
        where side_effects_list contains tuples of (side_effect_name, hpo_term, hpo_id)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
        
        # Extract set_id
        set_id_match = re.search(r'<set_id>(.*?)</set_id>', content)
        set_id = set_id_match.group(1) if set_id_match else None
        
        if not set_id:
            return None, []
        
        # Initialize side effects list
        side_effects = []
        
        # Extract side effects
        side_effects_sections = re.findall(r'<side_effects>(.*?)</side_effects>', content, re.DOTALL)
        for section in side_effects_sections:
            side_effect_blocks = re.findall(r'<side_effect>(.*?)</side_effect>', section, re.DOTALL)
            
            for block in side_effect_blocks:
                # Extract side effect name
                name_match = re.search(r'<side_effect_name>(.*?)</side_effect_name>', block)
                if not name_match:
                    name_match = re.search(r'<side_side_effect_name>(.*?)</side_side_effect_name>', block)
                
                if name_match:
                    side_effect_name = name_match.group(1).strip()
                    
                    # Extract HPO mapping
                    hpo_mapping = re.search(r'<hpo_mapping>(.*?)</hpo_mapping>', block, re.DOTALL)
                    if hpo_mapping:
                        mapping_content = hpo_mapping.group(1)
                        
                        # Extract HPO ID
                        hpo_id_match = re.search(r'<hpo_id>(.*?)</hpo_id>', mapping_content)
                        hpo_id = hpo_id_match.group(1).strip() if hpo_id_match else ""
                        
                        # Extract HPO term
                        hpo_term_match = re.search(r'<hpo_term>(.*?)</hpo_term>', mapping_content)
                        hpo_term = hpo_term_match.group(1).strip() if hpo_term_match else ""
                    else:
                        hpo_id = ""
                        hpo_term = ""
                    
                    # Clean name of any XML tags
                    clean_name = re.sub(r'<[^>]*>', '', side_effect_name)
                    
                    side_effects.append((clean_name, hpo_term, hpo_id))
        
        return set_id, side_effects
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, []

def create_side_effects_csv(data_dir, output_file):
    """
    Create a CSV file with side effects from mapped files joined with product information.
    
    Args:
        data_dir: Path to the data directory
        output_file: Path to the output CSV file
    
    Returns:
        Number of rows written
    """
    # Load product ingredients data
    ingredients_file = os.path.join(data_dir, 'human_product_ingredients.csv')
    
    try:
        ingredients_df = pd.read_csv(ingredients_file)
        # Create dictionary for lookup: set_id -> (ingredients, ingredient_rxcuis, spl_version)
        product_dict = {}
        for _, row in ingredients_df.iterrows():
            product_dict[row['set_id']] = {
                'ingredients': row['ingredients'],
                'ingredient_rxcuis': row['ingredient_rxcuis'],
                'spl_version': row['spl_version']
            }
        print(f"Loaded product data for {len(product_dict)} drugs")
    except Exception as e:
        print(f"Error loading ingredients file: {e}")
        return 0
    
    # Find all mapped files
    mapped_dir = os.path.join(data_dir, 'mapped')
    mapped_files = [f for f in os.listdir(mapped_dir) if f.endswith('_mapped.txt')]
    
    # Prepare CSV data
    csv_data = []
    processed_files = 0
    total_side_effects = 0
    
    # Process each mapped file
    print(f"Processing {len(mapped_files)} mapped files...")
    for idx, file_name in enumerate(mapped_files):
        if idx % 100 == 0 and idx > 0:
            print(f"Progress: {idx}/{len(mapped_files)} files processed, {total_side_effects} side effects extracted")
            
        file_path = os.path.join(mapped_dir, file_name)
        set_id, side_effects = extract_side_effects_from_mapped_file(file_path)
        
        if set_id is None:
            continue
        
        if not side_effects:
            continue
        
        # Get product info for this set_id
        product_info = product_dict.get(set_id)
        if not product_info:
            print(f"Warning: No product info found for set_id {set_id}")
            continue
        
        # Add each side effect to CSV data
        for side_effect_name, hpo_term, hpo_id in side_effects:
            csv_data.append([
                product_info['ingredients'],
                product_info['ingredient_rxcuis'],
                set_id,
                product_info['spl_version'],
                side_effect_name,
                hpo_term,
                hpo_id
            ])
        
        total_side_effects += len(side_effects)
        processed_files += 1
    
    print(f"Successfully processed {processed_files} files")
    print(f"Total side effects extracted: {total_side_effects}")
    
    # Write to CSV file
    header = ['ingredients', 'ingredient_rxcuis', 'set_id', 'spl_version', 
              'side_effect_name', 'side_effect_hpo_term', 'side_effect_hpo_id']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(csv_data)
    
    print(f"Wrote {len(csv_data)} rows to {output_file}")
    
    return len(csv_data)

# Main execution
data_directory = "data"
output_file = os.path.join(data_directory, "side_effects_mapped.csv")

# Check if the data directory exists
if not os.path.exists(data_directory):
    print(f"Error: Data directory {data_directory} not found")
    exit(1)

# Check if the mapped directory exists
mapped_dir = os.path.join(data_directory, 'mapped')
if not os.path.exists(mapped_dir):
    print(f"Error: Mapped directory {mapped_dir} not found")
    exit(1)

# Create the CSV file
num_rows = create_side_effects_csv(data_directory, output_file)

print(f"\nSuccessfully created {output_file} with {num_rows} side effects")
