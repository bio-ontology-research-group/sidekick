# SIDEKICK Experiments

This directory contains three experiments that validate and demonstrate the capabilities of the SIDEKICK knowledge graph, as described in the paper:

---

## 📋 Overview

The experiments evaluate SIDEKICK across three dimensions:

| Experiment | Purpose | Key Validation |
|-----------|---------|----------------|
| **Drug Repurposing** | Predict drug targets via side effect similarity | HPO ontology enables better semantic similarity than MedDRA |
| **Competency Questions** | Demonstrate ontology-based reasoning | SPARQL queries leverage HPO/MONDO hierarchies for complex queries |
| **ShEx Validation** | Ensure structural integrity | All clinical associations conform to the schema |

---

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB+ recommended (16GB+ for full validation)
- **Storage**: 2GB+ for data files

### Required Python Packages

Install all dependencies:
```bash
pip install rdflib pandas numpy scikit-learn matplotlib seaborn tqdm pyshex pronto oxrdflib
```
### Optional but Recommended
- **oxrdflib**: Provides Oxigraph store for 10-100x faster RDF queries (highly recommended)
- **pronto**: Required for ontology hierarchy reasoning in competency questions

---

## 📥 Required Data Files

**Files needed for experiments:**
- `Sidekick_v1.ttl` - Main RDF knowledge graph (~3.1M triples) [Download](https://doi.org/10.5281/zenodo.17779317)
- `side_effects_mapped.csv` - Side effects with HPO mappings (~184K rows) [Download](https://drive.google.com/file/d/11KRfE4AALyRFa7dFwUVwunHlQTKLASRG/view?usp=drive_link)
- `hp.obo` - HPO ontology (for semantic similarity and reasoning) [Download](https://hpo.jax.org/data/ontology)
- `mondo.obo` - MONDO ontology (for disease reasoning) [Download](https://mondo.monarchinitiative.org/pages/download/)
- `sidekick_validation.shex` - ShEx schema for validation
- `full_database.xml` - Drugbank XML file, see below for details on how to download. 

### Option 1: Use Pre-built Files from Zenodo

Download the SIDEKICK dataset from Zenodo:
```bash
# Download the full SIDEKICK dataset 
wget https://zenodo.org/record/17779317/files/Sidekick_v1.ttl
```

### Option 2: Build from Source

If you want to reproduce the entire pipeline from scratch to build the Sidekick_v1.ttl file:

```bash
# Clone the repository
git clone https://github.com/bio-ontology-research-group/sidekick.git
cd sidekick

# Follow the pipeline instructions in the main README
# This will generate the required csv and ttl file
./run_all.sh
```

### Additional Required File: DrugBank

**DrugBank** is required for the drug repurposing experiment (ground truth for drug-target associations):

1. **Download DrugBank**:
   - Visit: https://go.drugbank.com/releases/latest
   - Sign up for a free academic account
   - Download: "All Drugs" (full database XML)

2. **Place the file**:
   ```bash
   # Rename and place in the experiments directory
   mv drugbank.xml full_database.xml
   ```

**Note**: DrugBank requires registration but is free for academic use.

---

## 🧪 Experiment 1: Drug Repurposing via Side Effect Similarity

### Purpose
Evaluate SIDEKICK's ability to predict shared drug targets using semantic similarity of side effect profiles (BMA + Resnik on HPO ontology).

### Why This Matters
- Drugs with similar side effect profiles often share molecular targets
- HPO's axiomatic structure enables better semantic similarity than flat terminologies like MedDRA
- Demonstrates SIDEKICK's utility for drug repurposing and target prediction

### What About OnSIDES/MedDRA Comparison?

**Why MedDRA is not included:**

The paper compares SIDEKICK (using HPO) against OnSIDES (using MedDRA). However, **MedDRA requires licensing** and cannot be freely distributed:

- **Official MedDRA**: Requires expensive commercial license from MSSO
- **UMLS MedDRA**: Requires UMLS license (free but requires approval + specialized extraction)

Since we cannot provide MedDRA files or guarantee users have access, **we hardcode the published OnSIDES results** from the paper (Table 2):

| Method | AUC-ROC | Mean(+) | Mean(-) | Δ |
|--------|---------|---------|---------|---|
| OnSIDES (MedDRA+BMA+Resnik) | 0.6612 | 0.4412 | 0.2700 | 0.1711 |
| SIDEKICK (HPO+BMA+Resnik) | 0.7174 | 0.5935 | 0.3978 | 0.1957 |

**The experiment computes SIDEKICK results and compares against these hardcoded OnSIDES values.**

### Running the Experiment

```bash
python drug_repurposing.py \
    --sidekick-csv side_effects_mapped.csv \
    --drugbank-xml full_database.xml \
    --hpo-obo hp.obo \
    --output-dir results/drug_repurposing
```

### Expected Output

```
results/drug_repurposing/
├── sidekick_predictions.csv      # All drug pair predictions (y_true, y_score)
├── sidekick_metrics.csv          # Performance metrics (AUC, mean scores, delta)
├── comparison_summary.csv        # SIDEKICK vs OnSIDES comparison
├── roc_comparison.png            # ROC curve visualization
└── cache/                        # Cached computations for speed
    ├── hpo_ic.pkl               # Information Content values
    ├── hpo_ancestors.pkl        # Pre-computed ancestors
    └── hpo_similarities.pkl     # Pre-computed term similarities
```

### Expected Results

You should see:
- **SIDEKICK AUC**: ~0.7174 (may vary slightly with different DrugBank versions)
- **Improvement over OnSIDES**: ~8.5% relative improvement
- **Better discrimination**: SIDEKICK's Δ (0.1957) > OnSIDES Δ (0.1711)

### Runtime
- **First run**: 30-60 minutes (computes IC, ancestors, similarities)
- **Subsequent runs**: 5-10 minutes (uses cached values)

---

## 🔍 Experiment 2: Competency Questions (Ontology Reasoning)

### Purpose
Demonstrate SIDEKICK's ontology-based reasoning capabilities through 8 competency questions (SPARQL queries) that leverage HPO and MONDO hierarchies.

### What This Demonstrates

Each competency question showcases reasoning that **MedDRA cannot perform**:

1. **CQ1 (Cardiac Side Effects)**: Find ALL drugs with ANY cardiovascular effects via hierarchy traversal
2. **CQ2 (Nervous System)**: Aggregate CNS, PNS, and autonomic effects without manual enumeration
3. **CQ3 (Renal Contraindications)**: Cross-system reasoning (phenotypes + diseases)
4. **CQ4 (Cardiac Rhythm)**: Fine-grained physiological grouping (arrhythmias)
5. **CQ5 (Metabolic Effects)**: Multi-level hierarchy reasoning
6. **CQ6 (Infectious Diseases)**: Disease taxonomy via MONDO
7. **CQ7 (Cardiac-Safe Drugs)**: Combined safety reasoning (no cardiac effects OR contraindications)
8. **CQ8 (Heart Anatomy - Optional)**: Federated query to Ubergraph for anatomical reasoning

### Running the Experiment

**Basic usage (local queries only):**
```bash
python competency_questions.py \
    --sidekick-ttl Sidekick_v1.ttl \
    --hpo-obo hp.obo \
    --mondo-obo mondo.obo \
    --output-dir results/competency_questions
```

**With federated query:**
```bash
python competency_questions.py \
    --sidekick-ttl Sidekick_v1.ttl \
    --hpo-obo hp.obo \
    --mondo-obo mondo.obo \
    --run-federated \
    --output-dir results/competency_questions
```

### Expected Output

```
results/competency_questions/
├── competency_summary.csv        # Summary table of all CQs
├── competency_results.json       # Detailed JSON results
└── queries/
    ├── cq1_cardiac_side_effects.csv
    ├── cq2_nervous_system.csv
    ├── cq3_renal_contraindications.csv
    ├── cq4_cardiac_rhythm.csv
    ├── cq5_metabolic.csv
    ├── cq6_infectious_diseases.csv
    ├── cq7_cardiac_safe.csv
    └── cq8_federated_heart.csv   # Only if --run-federated
```

### Notes
- If `pronto` is not installed, ontology loading will be skipped (queries may return fewer results)
- CQ8 (federated) queries Ubergraph - it's optional

---

## ✅ Experiment 3: ShEx Schema Validation

### Purpose
Validate the structural integrity and completeness of the SIDEKICK knowledge graph against its ShEx schema.

### What This Validates

Checks that all 205,683 clinical associations conform to the schema:

1. **Drug Collections**: Proper ingredient composition
2. **Pharmaceutical Products**: Valid RxNorm mappings
3. **SPL Documents**: Complete provenance
4. **Side Effects**: Correct HPO mappings with provenance
5. **Indications**: Valid disease/phenotype/drug mappings
6. **Contraindications**: Proper safety information structure

### Running the Experiment

**Validate all nodes (comprehensive but slow):**
```bash
python shex_validation.py \
    --sidekick-ttl Sidekick_v1.ttl \
    --shex-schema sidekick_validation.shex \
    --output-dir results/shex_validation
```

**Validate sample (faster for testing):**
```bash
python shex_validation.py \
    --sidekick-ttl Sidekick_v1.ttl \
    --shex-schema sidekick_validation.shex \
    --sample-size 10000 \
    --output-dir results/shex_validation
```

**Verbose mode for debugging:**
```bash
python shex_validation.py \
    --sidekick-ttl Sidekick_v1.ttl \
    --shex-schema sidekick_validation.shex \
    --verbose \
    --output-dir results/shex_validation
```

### Expected Output

```
results/shex_validation/
├── validation_summary.csv        # Summary table with pass/fail rates
├── validation_report.json        # Detailed JSON report
└── errors/                       # Error details for failed shapes
    ├── SideEffectShape_errors.csv
    └── ... (if any validations fail)
```

---

## 📈 Reproducing Paper Results

To reproduce the exact results from the paper:

### Table 2: Drug Target Prediction Performance
```bash
python drug_repurposing.py \
    --sidekick-csv side_effects_mapped.csv \
    --drugbank-xml full_database.xml \
    --hpo-obo hp.obo \
    --output-dir results/drug_repurposing

# Check results/drug_repurposing/comparison_summary.csv
# Expected: SIDEKICK AUC ≈ 0.7174, OnSIDES AUC = 0.6612 (hardcoded)
```

### Table 3: Competency Questions Results
```bash
python competency_questions.py \
    --sidekick-ttl Sidekick_v1.ttl \
    --hpo-obo hp.obo \
    --mondo-obo mondo.obo \
    --output-dir results/competency_questions

# Check results/competency_questions/competency_summary.csv
# Expected: Results match Table 3 (see Expected Results section above)
```

### Section 4.3: Schema Validation
```bash
python shex_validation.py \
    --sidekick-ttl Sidekick_v1.ttl \
    --shex-schema sidekick_validation.shex \
    --output-dir results/shex_validation

# Check results/shex_validation/validation_summary.csv
# Expected: All shapes PASS
```

---

## 🔗 Additional Resources

- **Paper**: [Link to paper when published]
- **Dataset**: https://doi.org/10.5281/zenodo.17779317
- **SPARQL Endpoint**: http://sidekick.bio2vec.net/sparql
- **Website**: http://sidekick.bio2vec.net/
- **GitHub**: https://github.com/bio-ontology-research-group/sidekick/

---

## 📄 License

CC BY 4.0 International - https://creativecommons.org/licenses/by/4.0/
