#!/bin/bash

echo "======================================================================"
echo "SIDEKICK Knowledge Graph Construction Pipeline"
echo "======================================================================"
echo ""
echo "This script will run all 6 steps of the SIDEKICK pipeline."
echo "Estimated time: 24-48 hours"
echo "Estimated cost: $300-500 in API credits"
echo ""
read -p "Do you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Pipeline cancelled."
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 0: Preprocessing Pipeline"
echo "======================================================================"
python scripts/step_0_preprocessing_pipeline.py
if [ $? -ne 0 ]; then
    echo "Error in Step 0. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 1: Extract Side Effects"
echo "======================================================================"
python scripts/step_1_extract_SE.py
if [ $? -ne 0 ]; then
    echo "Error in Step 1. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 2: Map Side Effects to HPO"
echo "======================================================================"
python scripts/step_2_map_SE.py
if [ $? -ne 0 ]; then
    echo "Error in Step 2. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 3: Build Side Effects CSV"
echo "======================================================================"
python scripts/step_3_build_csv_SE.py
if [ $? -ne 0 ]; then
    echo "Error in Step 3. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 4: Extract, Classify, Map Indications/Contraindications"
echo "======================================================================"
python scripts/step_4_extract_classify_map_CI.py
if [ $? -ne 0 ]; then
    echo "Error in Step 4. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 5: Build RDF Knowledge Graph"
echo "======================================================================"
python scripts/step_5_build_RDF.py
if [ $? -ne 0 ]; then
    echo "Error in Step 5. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Final output: Sidekick_v1.ttl"
echo ""
