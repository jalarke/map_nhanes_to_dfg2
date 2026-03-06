#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src"

echo "Step 1: Building NHANES dataset — filters adults 18+, recodes demographics, and produces unique ingredient list."
python 01_build_nhanes_dataset.py

echo "Step 2: Text matching NHANES ingredients to DFG2 foods — cleans text and runs TF-IDF fuzzy matching."
python 02_text_match_nhanes_to_dfg2.py

echo "Step 3: Cleaning matches and labeling — merges accepted matches with DFG2 data and outputs final labels."
python 03_clean_matches_and_label.py

echo "Pipeline complete. Output: data/labels/nhanes_dfg2_labels.csv"
