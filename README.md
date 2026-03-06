# WWEIA Ingredients

The field of nutritional epidemiology attempts to link dietary intake data with health outcomes or other biomarkers associated with health or disease. However, dietary data from current publicly available databases is often aggregated at the level of food groups or mixed dishes, thereby limiting resolution and analysis of individual ingredients that may contribute to an outcome of interest. 

Recognizing these challenges, this project modifies the National Health And Nutrition Examination Survey (NHANES) What We Eat In America (WWEIA) data from 2001-2018 to generate an 'ingredientized' dataset by updating food codes and descriptions with the Food and Nutrition Database for Dietary Studies (FNDDS) database versions 2015-2016 and 2017-2018 and disaggregating mixed dishes into their ingredient representations from FNDDS and the Food Commodities Intake Database.

## Use and Requirements

1. clone the repository from the terminal: git clone https://github.com/JulesLarke-USDA/wweia_ingredients & cd wweia_ingredients
2. Create a conda environment from src/wweia_ingredients.yml: cd src & conda env create -f wweia_ingredients.yml
3. Activate the conda environment: conda activate wweia_ingredients
4. Run the pipeline: bash pipeline_ingredients.sh
- **03: Matching Discontinued FNDDS (WWEIA 01-13) Food Codes and Descriptions** will prompt download of nltk package data; the packages required are **wordnet** and **omw-1.4**


## Workflow

### 01: Consolidate ingredient codes
 
__Purpose__  
- Part 1: reduce the number of ingredients used by averaging commonly consumed food items. This creates a single averaged ingredient code from the multiple codes and weights found in these commonly consumed ingredients and reduces over-complexity and redundany.
- Part 2: Remaps ingredient codes represented as 8-digit foodcodes for obtaining proper ingredient weights per ingredient code.
- Part 3: Updates the ingredient codes and correponding nutrient values with ingredient descriptions from part 1. 

__Required Input Files__

  - **fndds_16** - Downloaded from: https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2015-2016%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx
  - **fndds_18** - Downloaded from: https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx
  - **nutrient_values_16** - Downloaded from: https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2015-2016%20FNDDS%20At%20A%20Glance%20-%20Ingredient%20Nutrient%20Values.xlsx
  - **nutrient_values_18** - Downloaded from: https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance%20-%20Ingredient%20Nutrient%20Values.xlsx

__Output__
- **fndds_16_consolidated_ingredient_codes.csv** (Part 1)
- **fndds_18_consolidated_ingredient_codes.csv** (Part 1)
- **fndds_16_all_ingredients.csv** (Part 2)
- **fndds_18_all_ingredients.csv** (Part 2)
- **fndds_all_ingredient_nutrient_values.csv** (Part 3)

### 02: Calculate FNDDS Ingredient Weights

__Purpose__  
This script recursively calculates the ingredient weight amounts within 8-digit foodcodes. This is needed to correctly represent the relative proportions of ingredient codes using 8-digit foodcodes

__Required Input Files__

  - **fndds_16_all_ingredients.csv** - Output from 01
  - **fndds_18_all_ingredients.csv** - Output from 01
  - **fndds_16_consolidated_ingredient_codes.csv** - Output from 01
  - **fndds_18_consolidated_ingredient_codes.csv** - Output from 01

__Workflow__
    1) Recursion: preprocessing to format dataframe for and perform calucations with get_ingredient_proportions.
    2) Combine result with the remianing ingredient codes that were properly represented as ingredient codes and not foodcodes.
    3) Combine the results for FNDDS1516 and FNDDS1718 into a single dataframe.
    
__Output__
  - **fndds_16_18_all.csv**

### 03: Matching Discontinued FNDDS (WWEIA 01-13) Food Codes and Descriptions

__Required Input Files__
  - **wweia_discontinued_foodcodes.csv** - List of discontinued foodcodes from FNDDS versions corresponding to WWEIA 01-13 
  - **fndds_16_18_all.csv** - Output from 02

__Information__  
This script prepares food descriptions in discontined foodcodes from WWEIA cycles (01 - 13) corresponding to early FNDDS versions for text similarity matching. This script achieves the following:
    
- Text cleaning: removal of punctuation and stopwords, lemmatization, etc.
- Finds matches based on main food descriptions (rather than foodcodes which were used in the crosswalk before) and exports these matches
        
__Output__
  - **string_match.csv** - This file will undergo manual curation to match discontinued foodcodes to the most appropriate foodcode in fndds_16_18_all.csv

__Note:__  The output of this manual matching is: string_match_discontinued_complete.csv and string_match_discontinued_complete_with_annotation.csv for documentation on added foodcodes.
Similarly, fndds_16_18_all.csv was updated with new codes for recipes that did not exist and were needed to match discontinued foodcodes. This updated file is: fndds_16_18_all_added_codes_for_discontinued.csv
Both of these manually curated outputs [string_match_discontinued_complete.csv and fndds_16_18_all_added_codes_for_discontinued.csv] are used in the script 04_wweia_ingredients.py

### 03b: FCID ingredientization

__Required Input Files__
- **fndds_16_18_all_added_codes_for_discontinued.csv** - manually curated foods and ingredients from script 03
- **ingredient_codes_to_ingredientization.csv** - list of ingredient codes identified to be multi-ingredient
- **fcid_data.csv** - Food Commodities Ingredient Database (Calculation performed on 6/23/2023 using FCID-WWEIA data for years 2005-2010 from https://fcid.foodrisk.org/recipes/#)

__Information__  
Many of the ingredient descriptions in FNDDS are multi-ingredient, these ingredient descriptions were manually identified to undergo further ingredientization by using the Food Commodities Intake Database which has ingredient representation of of WWEIA data.

__Output__
- **fcid_match.csv** - Text similarity matches from FCID to FNDDS
- **fcid_match_complete.csv** - Manually curated matches from FCID to FNDDS
- **fcid_match_complete_edit_0625.csv** - fcid_match_complete.csv edited on 062524 to not disaggregate chicken/beef broths or gravies. These break down into small amounts of meat and animal fats that aren't appropriate for vegetarian recipes.
- **fcid_ingredients.csv** - FCID descriptions and weights with corrected descriptions to represent ingredient codes identified as multi-ingredient in FNDDS
- **fndds_fcid_all.csv** - Combines fndds_16_18_all_added_codes_for_discontinued.csv with FCID ingredientized descriptions for multi-ingredient items. 

### 04: WWEIA Ingredient

__Required Input Files__

Demographic, Dietary day 1 and 2, and food description data for cycles 01 through 18 downloaded from https://wwwn.cdc.gov/NCHS/nhanes

  - **fndds_fcid_all.csv** - Manually curated version of fndds_16_18_all_added_codes_for_discontinued.csv with ingredient descriptions further ingredientized using the Food Commodities Ingredient Database
  - **string_match_discontinued_complete.csv** - Manually curated version of string_match.csv with added food codes and descriptions for matching discontinued foods.
  - **fndds_crosswalks** - Each of the crosswalk files in the fndds_crosswalk directory.

__Information__  
This script combines data for each of the WWEIA cycles (01 - 18), updates the food codes to those with ingredient codes and appends nutrient values corresponding to the ingredients.

__Output__
  - **wweia_all_recalls.txt** - The complete ingredientized dataset for WWEIA cycles 01-18. Diet intake with 65 nutrient estimates for each ingredient across 80275 individuals.
