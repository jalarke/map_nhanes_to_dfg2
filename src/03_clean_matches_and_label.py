import pandas as pd
from pathlib import Path

#load data for the four DFG2 foods and manually checked food mapping
dfg_foods = pd.read_csv('../data/dfg/dfg_foods.csv')
wweia_map = pd.read_csv('../data/text_match/nhanes_to_dfg2_manually_matched.csv') # manually curated list of food mapping from NHANES to DFG2

replace = wweia_map[wweia_map['replace'].notna()]

replace = replace.drop(columns=['simple_name', 'sample_id'])
replace = replace.rename(columns={'replace':'simple_name'})
replace = replace[['ingred_desc', 'ingred_code', 'simple_name']]

matched = wweia_map[wweia_map['match'].isna()]

matched = matched[['ingred_desc', 'ingred_code', 'simple_name']]

nhanes_dfg2 = pd.concat([matched, replace])

nhanes_dfg2_matches = nhanes_dfg2.merge(dfg_foods, on='simple_name', how='left')

nhanes_dfg2_matches['label'] = 1

wweia_map = wweia_map[['ingred_desc', 'ingred_code', 'simple_name', 'sample_id']] # select columns

non_matches = wweia_map[~wweia_map['ingred_code'].isin(nhanes_dfg2_matches['ingred_code'])]

non_matches['label'] = 0

nhanes_dfg2_labels = pd.concat([nhanes_dfg2_matches, non_matches])

output_dir = Path('../data/labels/')
output_dir.mkdir(exist_ok=True, parents=True)

nhanes_dfg2_labels.to_csv('../data/labels/nhanes_dfg2_labels.csv',index=None)

