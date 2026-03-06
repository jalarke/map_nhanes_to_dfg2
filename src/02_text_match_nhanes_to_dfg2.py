import pandas as pd
import string
import re
import nltk
nltk.data.path.append('/Users/local-jules/nltk_data')
wn = nltk.WordNetLemmatizer()

from polyfuzz.models import TFIDF
from polyfuzz import PolyFuzz

# all_ingred_list.csv is a list of all unique ingredients in the all NHANES participants. This was generated from 27332 participants (prior to excluding participants with <XX% of kcal consumed mappable to the DFG2)
nhanes = pd.read_csv('../data/ingredient_list/ingred_list.csv')

nhanes['ingred_desc'] = nhanes['ingred_desc'].str.replace('_', ' ')

# free_saccharide_v2.csv is a list of all DFG2 foods (same foods in each of the other DFG2 datasets)
dfg = pd.read_csv('../data/dfg/dfg_foods.csv', usecols=['sample_id', 'simple_name'])

dfg['simple_name'] = dfg['simple_name'].str.replace('_', ' ')

punct = string.punctuation[0:11] + string.punctuation[13:] # remove '-' from the list of punctuation.
stopwords = ['','and', 'to', 'not', 'no',  'bkdfrd', 'ppd', 'pkgddeli', 'pkgd', 'xtra', 'oz', 'in', 'with', 'or', 'only', 'cooking', 'as', 'food', 'distribution', 'usda', 'form', 'w', 'wo', 'ns', 'nfs', 'incl']

def clean_text(text):
    text = "".join([word for word in text if word not in punct])
    tokens = re.split('[-\W+]', text)
    text = [word for word in tokens if word not in stopwords]
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return "default" if text is [] else ' '.join(set(text))

nhanes['ingred_clean'] = nhanes['ingred_desc'].apply(lambda x: clean_text(x.lower()))
dfg['dfg_clean'] = dfg['simple_name'].apply(lambda x: clean_text(x.lower()))

nhanes_tokens = nhanes['ingred_clean'].to_list()
dfg_tokens = dfg['dfg_clean'].to_list()

tfidf = TFIDF(n_gram_range=(1, 3))
model = PolyFuzz(tfidf).match(nhanes_tokens, dfg_tokens)

match = model.get_matches()
match.rename(columns={'From':'ingred_clean', 'To':'dfg_clean'},inplace=True)

matched_1 = match.merge(nhanes, on='ingred_clean', how='left')

matched_2 = matched_1.merge(dfg, on='dfg_clean', how='left').drop_duplicates(subset='ingred_code')

matched_2 = matched_2[['ingred_clean', 'dfg_clean', 'ingred_desc', 'simple_name', 'ingred_code', 'sample_id', 'Similarity']]

matched_2.to_csv('../data/text_match/nhanes_to_dfg2_tfidf_matches.csv', index=None)

# nhanes_to_dfg2_tfidf_matches.csv will undergo manual curation to determine appropriateness of matches and edits for non-appropriate matches 
# Output file following manual matching: nhanes_to_dfg2_manually_matched.csv

