
# SIDEKICK: Semantically Integrated Drug Knowledge Graph

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17779317.svg)](https://doi.org/10.5281/zenodo.17779317)

SIDEKICK is a knowledge graph of drug safety information (side effects, indications, contraindications) extracted from FDA Structured Product Labels and mapped to standardized biomedical ontologies (HPO, MONDO, RxNorm). The figure below illustrates the workflow for creating the SIDEKICK knowledge graph.
<img width="2814" height="1461" alt="Sidekick_main (1) (1)" src="https://github.com/user-attachments/assets/a3a10c34-cf62-4b10-8d78-4fc635993c9b" />

## 📄 Publication

**SIDEKICK: A Semantically Integrated Resource for Drug Effects, Indications, and Contraindications**  
*Mohammad Ashhad, Olga Mashkova, Ricardo Henao, Robert Hoehndorf*  

**Abstract:** Pharmacovigilance and clinical decision support systems utilize structured drug safety data to guide medical practice. However,
  existing datasets frequently depend on terminologies such as MedDRA,
  which limits their semantic reasoning capabilities and their
  interoperability with Semantic Web ontologies and knowledge
  graphs. To address this gap, we developed SIDEKICK, a knowledge
  graph that standardizes drug indications, contraindications, and
  adverse reactions from FDA Structured Product Labels. We developed
  and used a workflow based on Large Language Model (LLM) extraction
  and Graph-Retrieval Augmented Generation (Graph RAG) for ontology
  mapping. We processed over 50,000 drug labels and mapped terms to
  the Human Phenotype Ontology (HPO), the MONDO Disease Ontology, and
  RxNorm. Our semantically integrated resource outperforms the SIDER
  and ONSIDES databases when applied to the task of drug repurposing
  by side effect similarity. We serialized the dataset as a Resource
  Description Framework (RDF) graph and employed the Semanticscience
  Integrated Ontology (SIO) as upper level ontology to further improve
  interoperability. Consequently, SIDEKICK enables automated safety
  surveillance and phenotype-based similarity analysis for drug repurposing.

[Paper Link](https://arxiv.org/abs/2602.19183v1) | [Zenodo Dataset](https://doi.org/10.5281/zenodo.17779317)

## 🚀 Quick Start

### Option 1: Use Pre-built Knowledge Graph (Recommended)

Download the ready-to-use RDF knowledge graph from Zenodo:
```bash
wget https://zenodo.org/record/17779317/files/Sidekick_v1.ttl
```

Query with SPARQL or load into a triplestore. You may also access the raw CSV files at the following [link](https://drive.google.com/drive/folders/18yubbQsGAH6KtKIXiHD3buVgPNHGBvlC?usp=sharing). The CSV files also include the unmapped terms (mapped to HP:0000001 or MONDO:0000001) that were excluded from the KG construction.

### Option 2: Reproduce from Scratch

Follow the pipeline below to rebuild the entire knowledge graph from source data.

---

## 📋 Prerequisites

- **Python**: 3.8+
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for data files
- **API Key**: [OpenRouter API key](https://openrouter.ai/) for LLM-based extraction and mapping

---

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/bio-ontology-research-group/sidekick.git
cd sidekick
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📥 Data Preparation

### Step 1: Download Required Data Files

Create the `data/` directory:
```bash
mkdir -p data
cd data
```

#### 1.1 DailyMed Human Prescription Labels (Required for Step 0)

Download the 5 DailyMed human prescription label archives:
```bash
# Download from FDA DailyMed FTP
wget https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm

# Or use direct links (example):
wget ftp://public.nlm.nih.gov/nlmdata/.dailymed/dm_spl_release_human_rx_part1.zip
wget ftp://public.nlm.nih.gov/nlmdata/.dailymed/dm_spl_release_human_rx_part2.zip
wget ftp://public.nlm.nih.gov/nlmdata/.dailymed/dm_spl_release_human_rx_part3.zip
wget ftp://public.nlm.nih.gov/nlmdata/.dailymed/dm_spl_release_human_rx_part4.zip
wget ftp://public.nlm.nih.gov/nlmdata/.dailymed/dm_spl_release_human_rx_part5.zip
```

**Place these 5 zip files in the root directory** (same level as `scripts/`).

#### 1.2 RxNorm Mappings File (Required for Step 0)

Download the FDA-RxNorm mapping file:
```bash
wget https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-mapping-files.cfm
# Extract rxnorm_mappings.txt
```

**Place `rxnorm_mappings.txt` in the root directory.**

#### 1.3 Product Ingredients CSV (Required for Steps 1-5)

After running Step 0, `human_product_ingredients.csv` will be created automatically in `data/`.

---

## 🔄 Pipeline Overview

The SIDEKICK pipeline consists of 6 steps:

```
Step 0: Preprocessing
  ↓ (human_product_ingredients.csv)
Step 1: Extract Side Effects
  ↓ (data/extracted/*.txt)
Step 2: Map Side Effects to HPO
  ↓ (data/mapped/*.txt)
Step 3: Build Side Effects CSV
  ↓ (side_effects_mapped.csv)
Step 4: Extract, Classify, Map Indications/Contraindications
  ↓ (indications_contraindications_classified.csv, 
     indications_contraindications_mapped_disease_phenotype.csv,
     indications_contraindications_mapped_drug_chemical.csv)
Step 5: Build RDF Knowledge Graph
  ↓ (Sidekick_v1.ttl)
```

---

## 🏃 Running the Pipeline

### Option A: Run All Steps Automatically

```bash
chmod +x run_all.sh
./run_all.sh
```

The script will prompt you for your OpenRouter API key once at the beginning.

### Option B: Run Steps Individually

#### **Step 0: Preprocessing (Filter Human Drugs, Extract SPL XMLs, Deduplicate)**

```bash
python scripts/step_0_preprocessing_pipeline.py
```

**Input:**
- `dm_spl_release_human_rx_part*.zip` (5 files in root directory)
- `rxnorm_mappings.txt` (in root directory)

**Output:**
- `human_setids.txt` - Cached human drug SET IDs
- `human_product_ingredients.csv` - All human products with ingredients
- `unique_human_ingredients_adverse_reactions.csv` - Deduplicated products
- `data/spls/*.xml` - Deduplicated SPL XML files (~18,500 files)

**Time:** ~30-60 minutes  
**Notes:** First run extracts SET IDs from archives (5-10 min). Subsequent runs use cache.

---

#### **Step 1: Extract Side Effects from SPL Files**

```bash
python scripts/step_1_extract_SE.py
```

**Input:**
- `data/spls/*.xml` - SPL XML files from Step 0
- OpenRouter API key (prompted)

**Output:**
- `data/raw_text/*.txt` - Preprocessed text from SPL files
- `data/extracted/*.txt` - Extracted indications, contraindications, side effects

**Time:** ~24-36 hours (for ~18,500 files)  
**Cost:** ~$300-400 in API calls  
**Notes:** Processes in batches of 500 with automatic resume on interruption.

---

#### **Step 2: Map Side Effects to HPO**

```bash
python scripts/step_2_map_SE.py
```

**Input:**
- `data/extracted/*.txt` - Extracted entities from Step 1
- OpenRouter API key (prompted)
- HPO ontology (downloads automatically)

**Output:**
- `data/mapped/*_mapped.txt` - Side effects mapped to HPO IDs with validation
- `data/hp.obo` - HPO ontology file
- `data/hpo_embeddings.pkl` - Cached HPO embeddings
- `data/term_mapping_cache.json` - Cached mappings

**Time:** ~16-24 hours  
**Cost:** ~$20-30 in API calls  
**Notes:** Uses Graph-RAG with semantic search + LLM validation. Caches results for efficiency.

---

#### **Step 3: Build Side Effects CSV**

```bash
python scripts/step_3_build_csv_SE.py
```

**Input:**
- `data/mapped/*_mapped.txt` - Mapped side effects from Step 2
- `human_product_ingredients.csv` - Product data from Step 0

**Output:**
- `side_effects_mapped.csv` - Final side effects dataset with HPO mappings

**Time:** <5 minutes  
**Notes:** No API calls. Pure data aggregation.

---

#### **Step 4: Extract, Classify, and Map Indications/Contraindications**

```bash
python scripts/step_4_extract_classify_map_CI.py
```

**Input:**
- `data/extracted/*.txt` - Extracted entities from Step 1
- `human_product_ingredients.csv` - Product data from Step 0
- OpenRouter API key (prompted)
- HPO & MONDO ontologies (downloads automatically)

**Output:**
- `indications_contraindications_classified.csv` - All I&C with classifications
- `indications_contraindications_mapped_disease_phenotype.csv` - Disease/Phenotype mapped to MONDO/HPO
- `indications_contraindications_mapped_drug_chemical.csv` - Drug interactions mapped to RxNorm (self-references cleaned)

**Time:** ~10-20 hours  
**Cost:** ~$30-40 in API calls  
**Notes:** 
- Classifies into 7 categories (Disease, Phenotype, Drug/Chemical, etc.)
- Maps using Graph-RAG with validation
- Automatically cleans self-references in drug interactions

---

#### **Step 5: Build RDF Knowledge Graph**

```bash
python scripts/step_5_build_RDF.py
```

**Input:**
- `side_effects_mapped.csv` - From Step 3
- `indications_contraindications_mapped_disease_phenotype.csv` - From Step 4
- `indications_contraindications_mapped_drug_chemical.csv` - From Step 4
- `human_product_ingredients.csv` - From Step 0

**Output:**
- `Sidekick_v1.ttl` - Final RDF knowledge graph in Turtle format

**Time:** ~10-30 minutes  
**Notes:** No API calls. Filters root terms and non-standard ontologies.

---

## 📊 Expected Outputs

After completing all steps, you should have:

| File | Description |
|------|-------------|
| `side_effects_mapped.csv` | Side effects with HPO mappings | 
| `indications_contraindications_classified.csv` | All I&C with classes | 
| `indications_contraindications_mapped_disease_phenotype.csv` | Disease/Phenotype I&C with MONDO/HPO | 
| `indications_contraindications_mapped_drug_chemical.csv` | Drug interactions with RxNorm | 
| `Sidekick_v1.ttl` | Final RDF knowledge graph | 

---

## ⚠️ Important Notes

1. **API Costs:** The full pipeline requires ~$300-500 in OpenRouter API credits for LLM calls.

2. **Processing Time:** Complete pipeline takes 24-48 hours depending on API rate limits and hardware.

3. **Resumability:** Steps 1, 2, and 4 support automatic resume via progress files. Safe to interrupt.

4. **Caching:** Embeddings and term mappings are cached to avoid recomputation.

5. **Validation:** All LLM outputs are validated against ontologies before acceptance.


---

## 🔍 Querying the Knowledge Graph

### Load into Python

```python
from rdflib import Graph

g = Graph()
g.parse("Sidekick_v1.ttl", format="turtle")

# Example query
query = """
PREFIX sk: <http://sidekick.bio2vec.net/>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT ?drug ?side_effect
WHERE {
  ?assoc a sk:SideEffect ;
         sk:refersToDrug ?drug ;
         sk:hasSideEffect ?side_effect .
}
LIMIT 10
"""

results = g.query(query)
for row in results:
    print(row)
```

### Use SPARQL Endpoint

Load into Apache Jena Fuseki, GraphDB, or Blazegraph for web-based querying or visit: http://sidekick.bio2vec.net/sparql

---

## 📝 License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 🤝 Contributing

Issues and pull requests are welcome! Please open an issue to discuss major changes.

---

## 📧 Contact

- Mohammad Ashhad: mohammad.ashhad@kaust.edu.sa
- Robert Hoehndorf: robert.hoehndorf@kaust.edu.sa

---

## 🔗 Links

- **Dataset:** [Zenodo](https://doi.org/10.5281/zenodo.17779317)
- **SPARQL Endpoint:** http://sidekick.bio2vec.net/sparql
- **Website:** http://sidekick.bio2vec.net/
