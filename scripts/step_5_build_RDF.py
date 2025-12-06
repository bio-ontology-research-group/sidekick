#!/usr/bin/env python3
"""
Sidekick Drug Knowledge Graph - Final Hub-and-Spoke Converter
Version: 1.0
"""

import pandas as pd
import hashlib
import subprocess
from datetime import datetime
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, BNode
from rdflib.namespace import OWL, DCTERMS
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
DATA_DIR = Path("data")
OUTPUT_FILE = "Sidekick_v1.ttl"

# Namespaces
SIO = Namespace("http://semanticscience.org/resource/")
SK = Namespace("http://sidekick.bio2vec.net/")
OBO = Namespace("http://purl.obolibrary.org/obo/")
RXNORM = Namespace("http://purl.bioontology.org/ontology/RXNORM/")

# SIO Classes
SIO_ACTIVE_INGREDIENT = SIO.SIO_010077  # active ingredient
SIO_PHARMACEUTICAL_DRUG = SIO.SIO_010039  # pharmaceutical drug
SIO_DISEASE = SIO.SIO_010299  # disease
SIO_PHENOTYPE = SIO.SIO_010056  # phenotype
SIO_ASSOCIATION = SIO.SIO_000897  # association
SIO_DOCUMENT = SIO.SIO_000148  # document
SIO_COLLECTION = SIO.SIO_000616  # collection

# SIO Properties
SIO_HAS_MEMBER = SIO.SIO_000059  # has member
SIO_HAS_PART = SIO.SIO_000028  # has part
SIO_REFERS_TO = SIO.SIO_000628  # refers to (Association -> Target)
SIO_HAS_SOURCE = SIO.SIO_000253  # has source (Association/Product -> SPL)

# Custom Types (The 7 Association Types)
SK_SIDE_EFFECT = SK.SideEffect
SK_DISEASE_INDICATION = SK.DiseaseIndication
SK_PHENOTYPE_INDICATION = SK.PhenotypeIndication
SK_DRUG_INDICATION = SK.DrugIndication
SK_DISEASE_CONTRAINDICATION = SK.DiseaseContraindication
SK_PHENOTYPE_CONTRAINDICATION = SK.PhenotypeContraindication
SK_DRUG_CONTRAINDICATION = SK.DrugContraindication
SK_DRUG_COLLECTION = SK.DrugCollection
SK_DISEASE_OR_PHENOTYPE = SK.DiseaseOrPhenotype

# Custom Properties (sub-properties of sio:refers_to)
SK_REFERS_TO_DRUG = SK.refersToDrug
SK_HAS_SIDE_EFFECT = SK.hasSideEffect
SK_IS_INDICATED_FOR_DISEASE = SK.isIndicatedForDisease
SK_IS_INDICATED_FOR_PHENOTYPE = SK.isIndicatedForPhenotype
SK_IS_INDICATED_WITH_DRUG = SK.isIndicatedWithDrug
SK_IS_CONTRAINDICATED_IN_DISEASE = SK.isContraindicatedInDisease
SK_IS_CONTRAINDICATED_IN_PHENOTYPE = SK.isContraindicatedInPhenotype
SK_IS_CONTRAINDICATED_WITH_DRUG = SK.isContraindicatedWithDrug

# Metadata Properties
SK_SET_ID = SK.setId
SK_VERSION = SK.version

# Initialize graph
g = Graph()
g.bind("sio", SIO)
g.bind("sk", SK)
g.bind("obo", OBO)
g.bind("rxnorm", RXNORM)
g.bind("owl", OWL)
g.bind("dcterms", DCTERMS)

# Add subclass axioms for custom association types
g.add((SK_SIDE_EFFECT, RDFS.subClassOf, SIO_ASSOCIATION))
g.add((SK_DISEASE_INDICATION, RDFS.subClassOf, SIO_ASSOCIATION))
g.add((SK_PHENOTYPE_INDICATION, RDFS.subClassOf, SIO_ASSOCIATION))
g.add((SK_DRUG_INDICATION, RDFS.subClassOf, SIO_ASSOCIATION))
g.add((SK_DISEASE_CONTRAINDICATION, RDFS.subClassOf, SIO_ASSOCIATION))
g.add((SK_PHENOTYPE_CONTRAINDICATION, RDFS.subClassOf, SIO_ASSOCIATION))
g.add((SK_DRUG_CONTRAINDICATION, RDFS.subClassOf, SIO_ASSOCIATION))
g.add((SK_DRUG_COLLECTION, RDFS.subClassOf, SIO_COLLECTION))

# Define DiseaseOrPhenotype as a union class
g.add((SK_DISEASE_OR_PHENOTYPE, RDF.type, OWL.Class))
disease_or_phenotype_list = BNode()
g.add((SK_DISEASE_OR_PHENOTYPE, OWL.unionOf, disease_or_phenotype_list))
g.add((disease_or_phenotype_list, RDF.first, SIO_DISEASE))
rest_list1 = BNode()
g.add((disease_or_phenotype_list, RDF.rest, rest_list1))
g.add((rest_list1, RDF.first, SIO_PHENOTYPE))
g.add((rest_list1, RDF.rest, RDF.nil))

# Add subproperty axioms for custom relations
g.add((SK_REFERS_TO_DRUG, RDFS.subPropertyOf, SIO_REFERS_TO))
g.add((SK_HAS_SIDE_EFFECT, RDFS.subPropertyOf, SIO_REFERS_TO))
g.add((SK_IS_INDICATED_FOR_DISEASE, RDFS.subPropertyOf, SIO_REFERS_TO))
g.add((SK_IS_INDICATED_FOR_PHENOTYPE, RDFS.subPropertyOf, SIO_REFERS_TO))
g.add((SK_IS_INDICATED_WITH_DRUG, RDFS.subPropertyOf, SIO_REFERS_TO))
g.add((SK_IS_CONTRAINDICATED_IN_DISEASE, RDFS.subPropertyOf, SIO_REFERS_TO))
g.add((SK_IS_CONTRAINDICATED_IN_PHENOTYPE, RDFS.subPropertyOf, SIO_REFERS_TO))
g.add((SK_IS_CONTRAINDICATED_WITH_DRUG, RDFS.subPropertyOf, SIO_REFERS_TO))

# Add property chain axiom for has_part o has_member => has_part
property_chain = BNode()
g.add((SIO_HAS_PART, OWL.propertyChainAxiom, property_chain))
g.add((property_chain, RDF.first, SIO_HAS_PART))
rest_chain = BNode()
g.add((property_chain, RDF.rest, rest_chain))
g.add((rest_chain, RDF.first, SIO_HAS_MEMBER))
g.add((rest_chain, RDF.rest, RDF.nil))

# Declare custom properties as Object Properties
g.add((SK_REFERS_TO_DRUG, RDF.type, OWL.ObjectProperty))
g.add((SK_HAS_SIDE_EFFECT, RDF.type, OWL.ObjectProperty))
g.add((SK_IS_INDICATED_FOR_DISEASE, RDF.type, OWL.ObjectProperty))
g.add((SK_IS_INDICATED_FOR_PHENOTYPE, RDF.type, OWL.ObjectProperty))
g.add((SK_IS_INDICATED_WITH_DRUG, RDF.type, OWL.ObjectProperty))
g.add((SK_IS_CONTRAINDICATED_IN_DISEASE, RDF.type, OWL.ObjectProperty))
g.add((SK_IS_CONTRAINDICATED_IN_PHENOTYPE, RDF.type, OWL.ObjectProperty))
g.add((SK_IS_CONTRAINDICATED_WITH_DRUG, RDF.type, OWL.ObjectProperty))

# Add annotations (definitions and examples) using Dublin Core
annotations = {
    # Classes
    SK_DRUG_COLLECTION: {
        "description": "A collection representing a unique set of one or more active drug ingredients.",
        "example": "sk:ingredient_set_123 a sk:DrugCollection.",
    },
    SK_DISEASE_OR_PHENOTYPE: {
        "description": "A class representing an entity that is either a disease or a phenotype. This is an owl:unionOf sio:Disease and sio:Phenotype.",
        "example": "obo:MONDO_0005071 a sk:DiseaseOrPhenotype.",
    },
    SK_SIDE_EFFECT: {
        "description": "An association indicating that a drug collection causes an adverse phenotype (side effect).",
        "example": "sk:assoc_123 a sk:SideEffect.",
    },
    SK_DISEASE_INDICATION: {
        "description": "An association indicating that a drug collection is used to treat a disease.",
        "example": "sk:assoc_123 a sk:DiseaseIndication.",
    },
    SK_PHENOTYPE_INDICATION: {
        "description": "An association indicating that a drug collection is used to treat a phenotype.",
        "example": "sk:assoc_123 a sk:PhenotypeIndication.",
    },
    SK_DRUG_INDICATION: {
        "description": "An association indicating that a drug collection is recommended for use with another drug collection.",
        "example": "sk:assoc_123 a sk:DrugIndication.",
    },
    SK_DISEASE_CONTRAINDICATION: {
        "description": "An association indicating that a drug collection should not be used in the context of a specific disease.",
        "example": "sk:assoc_123 a sk:DiseaseContraindication.",
    },
    SK_PHENOTYPE_CONTRAINDICATION: {
        "description": "An association indicating that a drug collection should not be used in the context of a specific phenotype.",
        "example": "sk:assoc_123 a sk:PhenotypeContraindication.",
    },
    SK_DRUG_CONTRAINDICATION: {
        "description": "An association indicating that a drug collection should not be used with another drug collection.",
        "example": "sk:assoc_123 a sk:DrugContraindication.",
    },
    # Properties
    SK_REFERS_TO_DRUG: {
        "description": "A property linking an association to the source drug collection that is the subject of the clinical finding.",
        "example": "sk:assoc_123 sk:refersToDrug sk:ingredient_set_456.",
    },
    SK_HAS_SIDE_EFFECT: {
        "description": "A property linking a sk:SideEffect association to the target phenotype.",
        "example": "sk:assoc_123 sk:hasSideEffect obo:HP_0001250.",
    },
    SK_IS_INDICATED_FOR_DISEASE: {
        "description": "A property linking a sk:DiseaseIndication association to the target disease.",
        "example": "sk:assoc_123 sk:isIndicatedForDisease obo:MONDO_0005071.",
    },
    SK_IS_INDICATED_FOR_PHENOTYPE: {
        "description": "A property linking a sk:PhenotypeIndication association to the target phenotype.",
        "example": "sk:assoc_123 sk:isIndicatedForPhenotype obo:HP_0002099.",
    },
    SK_IS_INDICATED_WITH_DRUG: {
        "description": "A property linking a sk:DrugIndication association to the target drug collection.",
        "example": "sk:assoc_123 sk:isIndicatedWithDrug sk:ingredient_set_789.",
    },
    SK_IS_CONTRAINDICATED_IN_DISEASE: {
        "description": "A property linking a sk:DiseaseContraindication association to the target disease.",
        "example": "sk:assoc_123 sk:isContraindicatedInDisease obo:MONDO_0005071.",
    },
    SK_IS_CONTRAINDICATED_IN_PHENOTYPE: {
        "description": "A property linking a sk:PhenotypeContraindication association to the target phenotype.",
        "example": "sk:assoc_123 sk:isContraindicatedInPhenotype obo:HP_0001250.",
    },
    SK_IS_CONTRAINDICATED_WITH_DRUG: {
        "description": "A property linking a sk:DrugContraindication association to the target drug collection.",
        "example": "sk:assoc_123 sk:isContraindicatedWithDrug sk:ingredient_set_789.",
    },
}

for term, annotation in annotations.items():
    g.add((term, DCTERMS.description, Literal(annotation["description"])))
    g.add((term, DCTERMS.instructionalMethod, Literal(annotation["example"])))

# Caches
ingredient_cache = {}
ingredient_set_cache = {}
product_cache = {}
phenotype_cache = {}
disease_cache = {}
spl_cache = {}

# Statistics tracking
stats = {
    'skipped_non_standard_ontologies': 0,
    'skipped_products_no_ingredients': 0,
    'mapped_to_mondo': 0,
    'mapped_to_hpo': 0,
    'skipped_root_terms': 0
}

# ===== HELPER FUNCTIONS =====


def get_git_commit_hash():
    """Get the current git commit hash."""
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        )
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(
            "Could not determine git commit hash. Is git installed and is this a git repository?"
        )
        return None


def parse_ingredients(ingredient_rxcuis_str):
    """Parse comma-separated RxCUIs into sorted list."""
    if pd.isna(ingredient_rxcuis_str) or str(ingredient_rxcuis_str).strip() == "":
        return []
    rxcuis = [x.strip() for x in str(ingredient_rxcuis_str).split(",") if x.strip()]
    return sorted(rxcuis, key=int)


def create_individual_ingredient(rxcui):
    """Create individual ingredient entity."""
    ingredient_uri = RXNORM[rxcui]
    if str(ingredient_uri) not in ingredient_cache:
        g.add((ingredient_uri, RDF.type, SIO_ACTIVE_INGREDIENT))
        ingredient_cache[str(ingredient_uri)] = True
    return ingredient_uri


def create_ingredient_set(rxcuis_list, ingredient_names_str):
    """Create ingredient set (The Hub)."""
    if not rxcuis_list:
        return None

    collection_id = f"ingredient_set_{'_'.join(rxcuis_list)}"
    collection_uri = SK[collection_id]

    if str(collection_uri) in ingredient_set_cache:
        return collection_uri

    g.add((collection_uri, RDF.type, SK_DRUG_COLLECTION))
    g.add(
        (
            collection_uri,
            RDFS.label,
            Literal(ingredient_names_str.strip() if ingredient_names_str else ""),
        )
    )

    for rxcui in rxcuis_list:
        individual_uri = create_individual_ingredient(rxcui)
        g.add((collection_uri, SIO_HAS_MEMBER, individual_uri))

    ingredient_set_cache[str(collection_uri)] = True
    return collection_uri


def clean_obo_id(ontology_id):
    if pd.isna(ontology_id) or str(ontology_id).strip() == "":
        return None
    return str(ontology_id).replace(":", "_")


def create_phenotype_entity(hpo_id, hpo_term):
    clean_id = clean_obo_id(hpo_id)
    if not clean_id:
        return None
    phenotype_uri = OBO[clean_id]

    if str(phenotype_uri) not in phenotype_cache:
        g.add((phenotype_uri, RDF.type, SK_DISEASE_OR_PHENOTYPE))
        if hpo_term and not pd.isna(hpo_term):
            g.add((phenotype_uri, RDFS.label, Literal(hpo_term.strip())))
        phenotype_cache[str(phenotype_uri)] = True
    return phenotype_uri


def create_disease_entity(mondo_id, mondo_term):
    clean_id = clean_obo_id(mondo_id)
    if not clean_id:
        return None
    disease_uri = OBO[clean_id]

    if str(disease_uri) not in disease_cache:
        g.add((disease_uri, RDF.type, SK_DISEASE_OR_PHENOTYPE))
        if mondo_term and not pd.isna(mondo_term):
            g.add((disease_uri, RDFS.label, Literal(mondo_term.strip())))
        disease_cache[str(disease_uri)] = True
    return disease_uri


def create_spl_entity(set_id, version):
    if pd.isna(set_id):
        return None
    full_set_id = str(set_id).strip()
    hash_id = hashlib.md5(full_set_id.encode()).hexdigest()[:12]
    spl_uri = SK[f"spl_{hash_id}"]

    if str(spl_uri) not in spl_cache:
        g.add((spl_uri, RDF.type, SIO_DOCUMENT))
        g.add((spl_uri, SK_SET_ID, Literal(full_set_id)))
        if not pd.isna(version):
            g.add((spl_uri, SK_VERSION, Literal(int(version))))
        g.add(
            (
                spl_uri,
                RDFS.seeAlso,
                URIRef(
                    f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={full_set_id}"
                ),
            )
        )
        spl_cache[str(spl_uri)] = True
    return spl_uri


def create_association(source_uri, target_uri, assoc_type, spl_uri=None):
    """
    Create an Association node.
    - source_uri: The Ingredient Set (Subject)
    - target_uri: Phenotype, Disease, or Ingredient Set (Object)
    - assoc_type: One of the 7 custom types
    - spl_uri: The source document
    """
    if not source_uri or not target_uri:
        return None

    # Deterministic hash based on the biological fact (Source + Target + Type)
    content = f"{source_uri}_{target_uri}_{assoc_type}"
    hash_id = hashlib.md5(content.encode()).hexdigest()[:12]
    assoc_uri = SK[f"assoc_{hash_id}"]

    # If this specific fact hasn't been recorded yet, create the node
    if (assoc_uri, RDF.type, SIO_ASSOCIATION) not in g:
        g.add((assoc_uri, RDF.type, SIO_ASSOCIATION))
        g.add((assoc_uri, RDF.type, assoc_type))  # The specific 7 types
        g.add((assoc_uri, SK_REFERS_TO_DRUG, source_uri))

        # Determine the specific relation based on assoc_type
        relation = SIO_REFERS_TO  # Default
        if assoc_type == SK_SIDE_EFFECT:
            relation = SK_HAS_SIDE_EFFECT
        elif assoc_type == SK_DISEASE_INDICATION:
            relation = SK_IS_INDICATED_FOR_DISEASE
        elif assoc_type == SK_PHENOTYPE_INDICATION:
            relation = SK_IS_INDICATED_FOR_PHENOTYPE
        elif assoc_type == SK_DRUG_INDICATION:
            relation = SK_IS_INDICATED_WITH_DRUG
        elif assoc_type == SK_DISEASE_CONTRAINDICATION:
            relation = SK_IS_CONTRAINDICATED_IN_DISEASE
        elif assoc_type == SK_PHENOTYPE_CONTRAINDICATION:
            relation = SK_IS_CONTRAINDICATED_IN_PHENOTYPE
        elif assoc_type == SK_DRUG_CONTRAINDICATION:
            relation = SK_IS_CONTRAINDICATED_WITH_DRUG
        g.add((assoc_uri, relation, target_uri))

        # Add a descriptive label
        source_label = g.value(source_uri, RDFS.label)
        target_label = g.value(target_uri, RDFS.label)
        source_label_str = str(source_label) if source_label else "Unknown Drug"
        target_label_str = str(target_label) if target_label else "Unknown Target"

        label_text = f"Association between {source_label_str} and {target_label_str}"
        if assoc_type == SK_SIDE_EFFECT:
            label_text = f"Side effect of {source_label_str} is {target_label_str}"
        elif assoc_type == SK_DISEASE_INDICATION:
            label_text = f"{source_label_str} is indicated for {target_label_str}"
        elif assoc_type == SK_PHENOTYPE_INDICATION:
            label_text = f"{source_label_str} is indicated for {target_label_str}"
        elif assoc_type == SK_DRUG_INDICATION:
            label_text = f"{source_label_str} is indicated for use with {target_label_str}"
        elif assoc_type == SK_DISEASE_CONTRAINDICATION:
            label_text = f"{source_label_str} is contraindicated in {target_label_str}"
        elif assoc_type == SK_PHENOTYPE_CONTRAINDICATION:
            label_text = f"{source_label_str} is contraindicated in {target_label_str}"
        elif assoc_type == SK_DRUG_CONTRAINDICATION:
            label_text = f"{source_label_str} is contraindicated with {target_label_str}"

        g.add((assoc_uri, RDFS.label, Literal(label_text)))

        # Add a detailed Dublin Core description
        assoc_type_name = str(assoc_type).split("/")[-1]
        spl_set_id_literal = g.value(spl_uri, SK_SET_ID) if spl_uri else None
        spl_set_id = str(spl_set_id_literal) if spl_set_id_literal else "N/A"

        description_text = (
            f"A '{assoc_type_name}' association between the drug collection "
            f"'{source_label_str}' and the target '{target_label_str}'. "
            f"This finding is documented in the SPL source with set_id: {spl_set_id}."
        )
        g.add((assoc_uri, DCTERMS.description, Literal(description_text)))

    # Always link the SPL source (provenance)
    if spl_uri:
        g.add((assoc_uri, SIO_HAS_SOURCE, spl_uri))

    return assoc_uri


# ===== PROCESSING FUNCTIONS =====


def process_side_effects():
    """Type 1: Side Effects -> Phenotype"""
    filepath = DATA_DIR / "side_effects_mapped.csv"
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return

    logger.info(f"Processing Side Effects...")
    df = pd.read_csv(filepath)
    
    # Filter root term
    initial_count = len(df)
    df = df[df["side_effect_hpo_id"] != "HP:0000001"]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        logger.info(f"  Filtered {filtered_count} root terms (HP:0000001)")
        stats['skipped_root_terms'] += filtered_count

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Side Effects"):
        rxcuis = parse_ingredients(row["ingredient_rxcuis"])
        if not rxcuis:
            continue

        source_set = create_ingredient_set(rxcuis, row["ingredients"])
        target_phenotype = create_phenotype_entity(
            row["side_effect_hpo_id"], row["side_effect_hpo_term"]
        )
        spl = create_spl_entity(row["set_id"], row["spl_version"])

        if source_set and target_phenotype:
            create_association(source_set, target_phenotype, SK_SIDE_EFFECT, spl)


def process_disease_phenotype_indications():
    """
    Types 2, 3, 5, 6: Indications/Contraindications -> Disease/Phenotype
    
    FIXED: Now uses actual mapped ontology (MONDO vs HPO) instead of LLM classification
    """
    filepath = DATA_DIR / "indications_contraindications_mapped_disease_phenotype.csv"
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return

    logger.info(f"Processing Indications/Contraindications (Disease/Phenotype)...")
    df = pd.read_csv(filepath)
    
    # Filter root terms
    initial_count = len(df)
    df = df[~df["ontology_id"].isin(["MONDO:0000001", "HP:0000001"])]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        logger.info(f"  Filtered {filtered_count} root terms")
        stats['skipped_root_terms'] += filtered_count

    skipped_ontologies = set()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Ind/Contra"):
        rxcuis = parse_ingredients(row["ingredient_rxcuis"])
        if not rxcuis:
            continue

        source_set = create_ingredient_set(rxcuis, row["ingredients"])
        spl = create_spl_entity(row["set_id"], row["spl_version"])

        # Get the actual mapped ontology ID
        ontology_id = str(row["ontology_id"]).strip()
        
        # CRITICAL FIX: Determine type based on ACTUAL ontology, not LLM classification
        target_uri = None
        assoc_type = None
        
        if ontology_id.startswith("MONDO:"):
            # It's a disease - use MONDO mapping
            target_uri = create_disease_entity(row["ontology_id"], row["ontology_term"])
            if row["type"] == "I":
                assoc_type = SK_DISEASE_INDICATION
            else:
                assoc_type = SK_DISEASE_CONTRAINDICATION
            stats['mapped_to_mondo'] += 1
            
        elif ontology_id.startswith("HP:"):
            # It's a phenotype - use HPO mapping
            target_uri = create_phenotype_entity(row["ontology_id"], row["ontology_term"])
            if row["type"] == "I":
                assoc_type = SK_PHENOTYPE_INDICATION
            else:
                assoc_type = SK_PHENOTYPE_CONTRAINDICATION
            stats['mapped_to_hpo'] += 1
            
        else:
            # Non-standard ontology (MAXO, ECTO, CHEBI, NCBITaxon, etc.)
            # Skip these as they don't fit our schema
            ontology_prefix = ontology_id.split(":")[0] if ":" in ontology_id else ontology_id
            skipped_ontologies.add(ontology_prefix)
            stats['skipped_non_standard_ontologies'] += 1
            continue

        if source_set and target_uri and assoc_type:
            create_association(source_set, target_uri, assoc_type, spl)
    
    if skipped_ontologies:
        logger.info(f"  Skipped {stats['skipped_non_standard_ontologies']} terms from non-standard ontologies: {sorted(skipped_ontologies)}")


def process_drug_interactions():
    """Types 4, 7: Indications/Contraindications -> Drug (Ingredient Set)"""
    filepath = DATA_DIR / "indications_contraindications_mapped_drug_chemical.csv"
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return

    logger.info(f"Processing Drug Interactions...")
    df = pd.read_csv(filepath)
    df = df[df["rxcuis"].notna() & (df["rxcuis"] != "other")]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Drug Interactions"):
        source_rxcuis = parse_ingredients(row["ingredient_rxcuis"])
        target_rxcuis = parse_ingredients(row["rxcuis"])

        if not source_rxcuis or not target_rxcuis:
            continue

        source_set = create_ingredient_set(source_rxcuis, row["ingredients"])
        # Target of a drug interaction is another Ingredient Set
        target_set = create_ingredient_set(target_rxcuis, row["ingredient_names"])
        spl = create_spl_entity(row["set_id"], row["spl_version"])

        assoc_type = None
        if row["type"] == "I":
            assoc_type = SK_DRUG_INDICATION
        else:
            assoc_type = SK_DRUG_CONTRAINDICATION

        if source_set and target_set and assoc_type:
            create_association(source_set, target_set, assoc_type, spl)


def process_products():
    """Process Products -> Link to Ingredient Set"""
    filepath = DATA_DIR / "human_product_ingredients.csv"
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return

    logger.info(f"Processing Products...")
    df = pd.read_csv(filepath)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Products"):
        if pd.isna(row["product_rxcui"]):
            continue

        # Create Ingredient Set (Hub) FIRST
        rxcuis = parse_ingredients(row["ingredient_rxcuis"])
        if not rxcuis:
            # Skip products without valid ingredients entirely
            stats['skipped_products_no_ingredients'] += 1
            continue  # Don't create the product at all

        # NOW create product (only if it has valid ingredients)
        product_uri = RXNORM[str(row["product_rxcui"]).strip()]
        if str(product_uri) not in product_cache:
            g.add((product_uri, RDF.type, SIO_PHARMACEUTICAL_DRUG))
            g.add((product_uri, RDFS.label, Literal(str(row["product_name"]).strip())))
            product_cache[str(product_uri)] = True

        ingredient_set_uri = create_ingredient_set(rxcuis, row["ingredients"])

        # LINK 1: Product has part DrugCollection (The Hub)
        g.add((product_uri, SIO_HAS_PART, ingredient_set_uri))

        # LINK 2: Product has source SPL
        spl = create_spl_entity(row["set_id"], row["spl_version"])
        if spl:
            g.add((product_uri, SIO_HAS_SOURCE, spl))


# ===== MAIN EXECUTION =====


def main():
    logger.info("=" * 80)
    logger.info("STARTING SIDEKICK GRAPH CONVERSION (HUB-AND-SPOKE MODEL)")
    logger.info("Version 1.1 - Fixed classification logic")
    logger.info("=" * 80)

    # Add ontology metadata
    ontology_uri = SK[""]  # The base namespace URI
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal("Sidekick Drug Knowledge Graph")))
    g.add(
        (
            ontology_uri,
            DCTERMS.description,
            Literal(
                "A knowledge graph of drug indications, contraindications, and side effects, derived from FDA SPL labels."
            ),
        )
    )
    g.add(
        (
            ontology_uri,
            OWL.imports,
            URIRef("http://semanticscience.org/ontology/sio.owl"),
        )
    )
    commit_hash = get_git_commit_hash()
    if commit_hash:
        g.add((ontology_uri, OWL.versionInfo, Literal(commit_hash)))

    # Add authors, license, and creation date
    g.add((ontology_uri, DCTERMS.created, Literal(datetime.now().isoformat())))
    g.add(
        (
            ontology_uri,
            DCTERMS.creator,
            Literal("Mohammad Ashhad <mohammad.ashhad@kaust.edu.sa>"),
        )
    )
    g.add(
        (
            ontology_uri,
            DCTERMS.creator,
            Literal("Olga Mashkova <olga.mashkova@kaust.edu.sa>"),
        )
    )
    g.add(
        (
            ontology_uri,
            DCTERMS.creator,
            Literal("Ricardo Henao <ricardo.henao@duke.edu>"),
        )
    )
    g.add(
        (
            ontology_uri,
            DCTERMS.creator,
            Literal("Robert Hoehndorf <robert.hoehndorf@kaust.edu.sa>"),
        )
    )
    g.add(
        (
            ontology_uri,
            DCTERMS.license,
            URIRef("https://creativecommons.org/licenses/by/4.0/"),
        )
    )

    # Process all data sources
    process_side_effects()
    process_disease_phenotype_indications()
    process_drug_interactions()
    process_products()

    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Terms mapped to MONDO: {stats['mapped_to_mondo']:,}")
    logger.info(f"Terms mapped to HPO: {stats['mapped_to_hpo']:,}")
    logger.info(f"Skipped root terms (MONDO:0000001, HP:0000001): {stats['skipped_root_terms']:,}")
    logger.info(f"Skipped non-standard ontologies (MAXO, ECTO, etc.): {stats['skipped_non_standard_ontologies']:,}")
    logger.info(f"Products without valid ingredients: {stats['skipped_products_no_ingredients']:,}")

    logger.info("\n" + "=" * 80)
    logger.info("Writing to file...")
    g.serialize(destination=OUTPUT_FILE, format="turtle")

    logger.info("=" * 80)
    logger.info(f" SUCCESS! Saved to {OUTPUT_FILE}")
    logger.info(f"Total Triples: {len(g):,}")
    logger.info(f"Unique Drug Collections: {len(ingredient_set_cache):,}")
    logger.info(f"Unique Products: {len(product_cache):,}")
    logger.info(f"Unique SPL Documents: {len(spl_cache):,}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
