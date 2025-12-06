# FDA SPL XML Extraction with OpenRouter
# Jupyter Notebook Version

import os
import glob
import time
import json
import requests
from bs4 import BeautifulSoup
import re
from tqdm.auto import tqdm
from getpass import getpass
import random

# Create directories
os.makedirs("data/spls", exist_ok=True)
os.makedirs("data/raw_text", exist_ok=True)
os.makedirs("data/extracted", exist_ok=True)

# Function to extract clean text from SPL XML files
def extract_clean_text_from_spl(xml_file_path, blacklist_codes=None):
    """
    Extract clean, structured text from an SPL XML file excluding blacklisted sections
    
    Args:
        xml_file_path (str): Path to the SPL XML file
        blacklist_codes (list): List of section codes to exclude
    
    Returns:
        str: Extracted text content with preserved structure
        str: set_id extracted from file or content
        str: version extracted from file or content
    """
    # Default blacklist if none provided
    if blacklist_codes is None:
        blacklist_codes = [
            '51945-4',  # PACKAGE LABEL.PRINCIPAL DISPLAY PANEL
            '48780-1',  # SPL LISTING DATA ELEMENTS SECTION / SPL product data elements section
            '44425-7',  # STORAGE AND HANDLING SECTION
            '34069-5',  # HOW SUPPLIED SECTION
            '34093-5',  # REFERENCES SECTION
            '59845-8',  # INSTRUCTIONS FOR USE SECTION
            '42230-3',  # SPL PATIENT PACKAGE INSERT SECTION
            '42231-1',  # SPL MEDGUIDE SECTION
            '53413-1',  # OTC - QUESTIONS SECTION
            '50565-1',  # OTC - KEEP OUT OF REACH OF CHILDREN SECTION
            '34076-0',  # INFORMATION FOR PATIENTS SECTION
            '55106-9',  # OTC - ACTIVE INGREDIENT SECTION
            '50569-3',  # OTC - ASK DOCTOR SECTION
            '50568-5',  # OTC - ASK DOCTOR/PHARMACIST SECTION
            '34068-7',  # DOSAGE & ADMINISTRATION SECTION
            '51727-6',  # INACTIVE INGREDIENT SECTION
            '55105-1',  # OTC - PURPOSE SECTION
            '50570-1',  # OTC - DO NOT USE SECTION
            '50567-7',  # OTC - WHEN USING SECTION
            '50566-9',  # OTC - STOP USE SECTION
            '53414-9',  # OTC - PREGNANCY OR BREAST FEEDING SECTION
            '34090-1',  # Clinical Pharmacology
            '43682-4',  # Pharmacokinetics
            '43679-0',  # Mechanism Of Action
            '43681-6',  # Pharmacodynamics
        ]
    
    try:
        # Read the XML file
        with open(xml_file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Use BeautifulSoup to parse the XML
        soup = BeautifulSoup(content, 'xml')
        
        # Extract set_id and version from filename or document
        set_id = None
        version = None
        
        # Try to get from filename first
        file_name = os.path.basename(xml_file_path)
        if '_' in file_name:
            parts = file_name.replace('.xml', '').split('_')
            if len(parts) >= 2:
                set_id = parts[0]
                version = parts[1]
        
        # If not found in filename, try to extract from document
        if not set_id or not version:
            # Try to find setId element
            set_id_elem = soup.find('setId')
            if set_id_elem and 'root' in set_id_elem.attrs:
                set_id = set_id_elem['root']
            
            # Try to find versionNumber
            version_elem = soup.find('versionNumber')
            if version_elem and 'value' in version_elem.attrs:
                version = version_elem['value']
        
        # Initialize the output text with set_id and version
        text_output = []
        if set_id:
            text_output.append(f"<set_id>{set_id}</set_id>")
        if version:
            text_output.append(f"<version>{version}</version>")
        
        text_output.append("=" * 80)
        text_output.append("")
        
        # Process all sections that aren't in the blacklist
        for section in soup.find_all('section'):
            # Check if section should be excluded
            section_code = None
            code_elem = section.find('code')
            
            if code_elem and code_elem.get('code'):
                section_code = code_elem.get('code')
                
                # Skip if the section code is in the blacklist
                if section_code in blacklist_codes:
                    continue
            
            # Get section title
            section_title = None
            if section.find('title'):
                section_title = section.find('title').text.strip()
            elif code_elem and code_elem.get('displayName'):
                section_title = code_elem.get('displayName').strip()
            
            # Add section title
            if section_title:
                text_output.append(f"\n{section_title}")
                text_output.append("=" * len(section_title))
            
            # Process tables to preserve their structure
            for table in section.find_all('table'):
                text_output.append("\nTABLE:")
                
                # Extract rows
                for row in table.find_all('tr'):
                    cells = []
                    for cell in row.find_all(['th', 'td']):
                        cell_text = cell.get_text().strip().replace('\n', ' ')
                        cells.append(cell_text)
                    
                    if cells:
                        text_output.append(" | ".join(cells))
                
                text_output.append("")
                # Remove the table to prevent duplicating content
                table.extract()
            
            # Get section text content
            section_text = section.get_text().strip()
            if section_text:
                # Clean up the text
                lines = []
                current_line = ""
                
                # Split by line breaks but preserve paragraph structure
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
                
                # Add the last line if there is one
                if current_line:
                    lines.append(current_line)
                
                # Add the cleaned text
                for line in lines:
                    text_output.append(line)
                
                text_output.append("")
        
        # Join all text parts
        full_text = "\n".join(text_output)
        
        # Clean up multiple newlines
        while "\n\n\n" in full_text:
            full_text = full_text.replace("\n\n\n", "\n\n")
        
        return full_text, set_id, version
    
    except Exception as e:
        print(f"Error extracting text from {xml_file_path}: {str(e)}")
        return f"ERROR: Could not extract text - {str(e)}", None, None

# Function to call OpenRouter API
def call_openrouter_api(text, api_key, model="google/gemini-2.5-flash", retry_count=3, retry_delay=5):
    """
    Call the OpenRouter API to extract information from the SPL text
    
    Args:
        text (str): Text content from SPL XML file
        api_key (str): OpenRouter API key
        model (str): Model identifier to use
        retry_count (int): Number of retries if API call fails
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Extracted information in XML format
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://jupyter.org"  # Updated for Jupyter
    }
    
    prompt = f"""You are an expert in extracting information from FDA drug labels. I have provided the text from a drug package label. Please extract the following information in a structured format: 
IMPORTANT: Only respond with the extracted XML. Do not repeat any part of these instructions or the input text in your response.

1. Indications (what the drug is used for) 
2. Contraindications (when the drug should not be used) 
3. Side effects (with frequencies if available) 

For each indication, contraindication, side effect, provide only one item per tag and include the exact line from the text that contains this information. Extract any and all indications, contraindications, side effects you find. 
It is important to note that these side effects, indications and contraindications can be found in sections other than the ones specifically dedicated for them so search carefully across the entire text and find all of them. 
Try to keep the indication, contraindication and side-effect names that you extract as short and straightforward as possible but accuracy is important.
For each side effect also extract the frequency information (if available). Frequency means the percentage of people who experienced the side effect (If extracting from a table ensure to chech the table header and include the % symbol after the number) or it could be phrases such as "common", "rare" etc. Only extract frequency if available else write "NA" in the frequency info.

Provide your response in the following format, remember to populate this with the information extracted from the label: 

<drug_information> 
<indications> 
<indication> 
<indication_name>INDICATION NAME</indication_name> 
</indication> 
<!-- Additional indications as needed --> 
</indications> 
<contraindications> 
<contraindication> 
<contraindication_name>CONTRAINDICATION NAME</contraindication_name>
</contraindication>
<!-- Additional contraindications as needed --> 
</contraindications> 
<side_effects> 
<side_effect> 
<side_effect_name>SIDE EFFECT NAME</side_effect_name> 
<frequency>FREQUENCY IF AVAILABLE</frequency> 
</side_effect>
<!-- Additional side-effects as needed --> 
</side_effects> 
</drug_information>

Here's the content from which you need to extract the information:
{text}"""
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # Low temperature for more consistent results
        "max_tokens": 50000
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = retry_delay * (attempt + 1)
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"API request failed: {response.status_code} - {response.text}")
                if attempt < retry_count - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        except Exception as e:
            print(f"Error calling OpenRouter API: {str(e)}")
            if attempt < retry_count - 1:
                wait_time = retry_delay * (attempt + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    return f"ERROR: Failed to get response after {retry_count} attempts"

# Function to extract XML from response
def extract_xml_from_response(response):
    """
    Extract the XML portion from the API response
    """
    xml_pattern = r"<drug_information>.*?</drug_information>"
    xml_matches = re.search(xml_pattern, response, re.DOTALL)
    
    if xml_matches:
        return xml_matches.group(0)
    else:
        return response  # Return the full response if XML can't be extracted

# Get OpenRouter API key
openrouter_api_key = getpass("Enter your OpenRouter API key (will be hidden): ")

# Configure processing parameters
spls_dir = "data/spls"  # Update this with the path to your SPL files
raw_text_dir = "data/raw_text"
extracted_dir = "data/extracted"
model = "google/gemini-2.5-flash"  # You can change this to another model
limit = None  # Set a number to limit processing for testing
batch_size = 500  # Process 5 files before sleeping
sleep_between_batches = 10  # Sleep seconds between batches
resume = True  # Skip files that already have outputs

# Create log file for tracking processed files
log_file = os.path.join(extracted_dir, "processing_log.txt")
processed_files = set()

if resume and os.path.exists(log_file):
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    processed_files.add(line.strip())
        print(f"Found {len(processed_files)} previously processed files")
    except Exception as e:
        print(f"Error reading log file: {str(e)}")

# Find all SPL XML files
xml_files = glob.glob(os.path.join(spls_dir, "*.xml"))
print(f"Found {len(xml_files)} SPL XML files in {spls_dir}")

# Filter out already processed files if resume is enabled
if resume:
    xml_files = [f for f in xml_files if os.path.basename(f) not in processed_files]
    print(f"{len(xml_files)} files remaining to process")

if limit:
    xml_files = xml_files[:limit]
    print(f"Limited to processing {limit} files")

# Shuffle files to distribute complex files across batches
random.shuffle(xml_files)

# Process files in batches
processed_count = 0
error_count = 0
log_handle = open(log_file, 'a')

try:
    for i, xml_file in enumerate(tqdm(xml_files, desc="Processing SPL files")):
        try:
            file_name = os.path.basename(xml_file)
            file_base = os.path.splitext(file_name)[0]
            
            # Check if output already exists (additional check even if we filtered above)
            output_file = os.path.join(extracted_dir, f"{file_base}.txt")
            if resume and (os.path.exists(output_file) or file_name in processed_files):
                print(f"Skipping {file_name} - output already exists")
                continue
            
            # 1. Extract raw text
            raw_text, set_id, version = extract_clean_text_from_spl(xml_file)
            
            if "ERROR:" in raw_text or not set_id or not version:
                print(f"Error extracting text from {file_name}")
                error_count += 1
                continue
            
            # 2. Save raw text
            raw_text_file = os.path.join(raw_text_dir, f"{file_base}.txt")
            with open(raw_text_file, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            
            # 3. Process with OpenRouter
            print(f"Processing {file_name} with OpenRouter API...")
            response = call_openrouter_api(raw_text, openrouter_api_key, model)
            
            if "ERROR:" in response:
                print(f"API error with {file_name}: {response}")
                error_count += 1
                continue
            
            # 4. Extract and process XML response
            xml_response = extract_xml_from_response(response)
            
            if "<drug_information>" in xml_response and "</drug_information>" in xml_response:
                # 5. Add set_id and version tags to the beginning of the XML response
                final_output = f"<set_id>{set_id}</set_id>\n<version>{version}</version>\n\n{xml_response}"
                
                # 6. Save to extracted directory
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_output)
                
                # 7. Log processed file
                log_handle.write(f"{file_name}\n")
                log_handle.flush()
                processed_files.add(file_name)
                processed_count += 1
            else:
                print(f"Failed to extract XML structure from {file_name}")
                error_count += 1
                
                # Save raw response for debugging
                debug_file = os.path.join(extracted_dir, f"debug_{file_base}.txt")
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(response)
            
            # Sleep between batches to avoid rate limits
            if (i + 1) % batch_size == 0 and i < len(xml_files) - 1:
                print(f"Completed batch. Sleeping for {sleep_between_batches} seconds...")
                time.sleep(sleep_between_batches)
        
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            error_count += 1

finally:
    # Close log file
    log_handle.close()
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed_count} files")
    print(f"  Errors: {error_count} files")
    print(f"  Raw text saved to: {raw_text_dir}")
    print(f"  Structured data saved to: {extracted_dir}")
