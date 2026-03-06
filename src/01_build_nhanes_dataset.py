# Import packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Load data for WWEIA_ingredients
# Dataset (wweia_all_recalls.txt) can be generated from https://github.com/JulesLarke-USDA/wweia_ingredients
# and placed in the proper directory for loading

wweia = pd.read_csv('../data/nhanes/wweia_all_recalls.txt', sep='\t', usecols=['SEQN','ingred_code','ingred_desc', 'Ingred_consumed_g', 'Energy', 'Fatty acids, total monounsaturated',	'Fatty acids, total polyunsaturated',	'Fatty acids, total saturated', 'Sodium', 'Carbohydrate', 'Fiber, total dietary', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1',	'INDFMPIR',	'DMDEDUC3',	'DMDEDUC2',	'WTINT2YR',	'WTMEC2YR',	'SDMVPSU',	'SDMVSTRA',	'CYCLE',	'diet_wts'])

# subset to include 18+ y/o
adults = wweia[wweia['RIDAGEYR'] > 17].copy()

adults.rename(columns={'RIDAGEYR': 'Age'},inplace=True)
adults.rename(columns={'RIAGENDR': 'Sex'},inplace=True)

### Education

# recode edu levels for age 19 and under
adults['DMDEDUC3'] = adults['DMDEDUC3'].replace([13, 14, 15], 'high school graduate or equivalent')
adults['DMDEDUC3'] = adults['DMDEDUC3'].replace([9, 10, 11, 12, 66, 99], 'less than high school graduate')

# recode edu levels for age 20 and older
adults['DMDEDUC2'] = adults['DMDEDUC2'].replace([1, 2], 'less than high school graduate')
adults['DMDEDUC2'] = adults['DMDEDUC2'].replace(3, 'high school graduate or equivalent')
adults['DMDEDUC2'] = adults['DMDEDUC2'].replace(4, 'some college')
adults['DMDEDUC2'] = adults['DMDEDUC2'].replace(5, 'college graduate')
adults['DMDEDUC2'] = adults['DMDEDUC2'].replace([7, 9], 'unknown')

# create single feature for education including all participants
adults['education'] = adults.filter(like='DMDEDUC').ffill(axis=1).iloc[:,-1].copy()
adults = adults[adults['education']!='unknown']
adults = adults.drop(columns=['DMDEDUC2', 'DMDEDUC3'])

### BMI

# Exam data - Body Measures (BMI: BMXBMI)
bmx_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BMX_B.XPT', format='xport', encoding='utf-8')
bmx_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/BMX_C.XPT', format='xport', encoding='utf-8')
bmx_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BMX_D.XPT', format='xport', encoding='utf-8')
bmx_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BMX_E.XPT', format='xport', encoding='utf-8')
bmx_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BMX_F.XPT', format='xport', encoding='utf-8')
bmx_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BMX_G.XPT', format='xport', encoding='utf-8')
bmx_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BMX_H.XPT', format='xport', encoding='utf-8')
bmx_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.XPT', format='xport', encoding='utf-8')
bmx_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.XPT', format='xport', encoding='utf-8')

bmx_B = bmx_B[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]
bmx_C = bmx_C[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]
bmx_D = bmx_D[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]
bmx_E = bmx_E[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]
bmx_F = bmx_F[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]
bmx_G = bmx_G[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]
bmx_H = bmx_H[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]
bmx_I = bmx_I[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]
bmx_J = bmx_J[['SEQN', 'BMXBMI', 'BMXWT', 'BMXWAIST']]

bmi = pd.concat([bmx_B, bmx_C, bmx_D, bmx_E, bmx_F, bmx_G, bmx_H, bmx_I, bmx_J])
bmi.rename(columns={'BMXBMI': 'BMI', 'BMXWT': 'body_wt', 'BMXWAIST': 'WC'}, inplace=True)

adults = adults.merge(bmi, on='SEQN', how='left')

# ### Impute BMI from WC for those who have WC measured

no_bmi = adults[adults['BMI'].isnull()]

no_bmi = no_bmi[['SEQN', 'Sex', 'BMI', 'WC']]

no_bmi = no_bmi.drop_duplicates(subset=['SEQN'])

no_bmi = no_bmi[~no_bmi['WC'].isnull()]

no_bmi_m = no_bmi[no_bmi['Sex']=='Male'].copy()
no_bmi_f = no_bmi[no_bmi['Sex']=='Female'].copy()

bmi_wc = adults[['SEQN', 'Sex', 'BMI', 'WC']]

bmi_wc = bmi_wc.drop_duplicates(subset=['SEQN'])

bmi_wc = bmi_wc.dropna(subset=['BMI', 'WC'])

bmi_wc_m = bmi_wc[bmi_wc['Sex']=='Male']

bmi_wc_f = bmi_wc[bmi_wc['Sex']=='Female']

x_m = np.array(bmi_wc_m['WC']).reshape((-1, 1))
y_m = np.array(bmi_wc_m['BMI']).reshape((-1, 1))
z_m = np.array(no_bmi_m['WC']).reshape((-1, 1))

x_f = np.array(bmi_wc_f['WC']).reshape((-1, 1))
y_f = np.array(bmi_wc_f['BMI']).reshape((-1, 1))
z_f = np.array(no_bmi_f['WC']).reshape((-1, 1))

model_m = LinearRegression().fit(x_m, y_m)

r_sq = model_m.score(x_m, y_m)
# print(f"coefficient of determination: {r_sq}")
# print(f"intercept: {model_m.intercept_}")
# print(f"slope: {model_m.coef_}")

y_pred_m = model_m.predict(z_m)
# print(f"predicted response:\n{y_pred_m}")

model_f = LinearRegression().fit(x_f, y_f)

r_sq = model_f.score(x_f, y_f)
# print(f"coefficient of determination: {r_sq}")
# print(f"intercept: {model_f.intercept_}")
# print(f"slope: {model_f.coef_}")

y_pred_f = model_m.predict(z_f)
#print(f"predicted response:\n{y_pred_f}")

no_bmi_m['BMI'] = y_pred_m.flatten()
no_bmi_f['BMI'] = y_pred_f.flatten()

no_bmi = pd.concat([no_bmi_m, no_bmi_f])

adults.set_index('SEQN', inplace=True)
no_bmi.set_index('SEQN', inplace=True)

adults['BMI'].update(no_bmi['BMI'])
adults.dropna(subset='BMI', inplace=True)
adults.reset_index(inplace=True)

# ### Smoking status

# Questionnaire data - Current or ever smoker
smq_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/SMQ_B.XPT', format='xport', encoding='utf-8')
smq_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/SMQ_C.XPT', format='xport', encoding='utf-8')
smq_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/SMQ_D.XPT', format='xport', encoding='utf-8')
smq_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/SMQ_E.XPT', format='xport', encoding='utf-8')
smq_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/SMQ_F.XPT', format='xport', encoding='utf-8')
smq_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SMQ_G.XPT', format='xport', encoding='utf-8')
smq_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SMQ_H.XPT', format='xport', encoding='utf-8')
smq_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.XPT', format='xport', encoding='utf-8')
smq_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.XPT', format='xport', encoding='utf-8')

smq_B = smq_B[['SEQN', 'SMQ020', 'SMQ040']]
smq_C = smq_C[['SEQN', 'SMQ020', 'SMQ040']]
smq_D = smq_D[['SEQN', 'SMQ020', 'SMQ040']]
smq_E = smq_E[['SEQN', 'SMQ020', 'SMQ040']]
smq_F = smq_F[['SEQN', 'SMQ020', 'SMQ040']]
smq_G = smq_G[['SEQN', 'SMQ020', 'SMQ040']]
smq_H = smq_H[['SEQN', 'SMQ020', 'SMQ040']]
smq_I = smq_I[['SEQN', 'SMQ020', 'SMQ040']]
smq_J = smq_J[['SEQN', 'SMQ020', 'SMQ040']]

smq = pd.concat([smq_B, smq_C, smq_D, smq_E, smq_F, smq_G, smq_H, smq_I, smq_J])
smq.rename(columns={'SMQ020':'ever_smoker', 'SMQ040': 'current_smoker'}, inplace=True)

adults = adults.merge(smq, on='SEQN', how='left')

# recode levels for smoking status
adults['ever_smoker'] = adults['ever_smoker'].replace(1, 'yes')
adults['ever_smoker'] = adults['ever_smoker'].replace(2, 'no')
adults['ever_smoker'] = adults['ever_smoker'].replace([7, 9], 'unknown')

adults = adults[adults['ever_smoker']!='unknown']

# ### Diabetes

#Lab data - fasting glucose

fg_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L10AM_B.XPT', format='xport', encoding='utf-8')
fg_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L10AM_C.XPT', format='xport', encoding='utf-8')
fg_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/GLU_D.XPT', format='xport', encoding='utf-8')
fg_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/GLU_E.XPT', format='xport', encoding='utf-8')
fg_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/GLU_F.XPT', format='xport', encoding='utf-8')
fg_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/GLU_G.XPT', format='xport', encoding='utf-8')
fg_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/GLU_H.XPT', format='xport', encoding='utf-8')
fg_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GLU_I.XPT', format='xport', encoding='utf-8')
fg_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.XPT', format='xport', encoding='utf-8')

fg_B = fg_B[['SEQN', 'LBXGLU']]
fg_C = fg_C[['SEQN', 'LBXGLU']]
fg_D = fg_D[['SEQN', 'LBXGLU']]
fg_E = fg_E[['SEQN', 'LBXGLU']]
fg_F = fg_F[['SEQN', 'LBXGLU']]
fg_G = fg_G[['SEQN', 'LBXGLU']]
fg_H = fg_H[['SEQN', 'LBXGLU']]
fg_I = fg_I[['SEQN', 'LBXGLU']]
fg_J = fg_J[['SEQN', 'LBXGLU']]

fg = pd.concat([fg_B, fg_C, fg_D, fg_E, fg_F, fg_G, fg_H, fg_I, fg_J])
fg.rename(columns={'LBXGLU':'fasting_glc_mg_dL'}, inplace=True)

adults = adults.merge(fg, on='SEQN', how='left')

adults['diabetes_fasting_glc'] = np.where(adults['fasting_glc_mg_dL'] >= 126, 'yes', 'no')

#Lab data - glycohemoglobic (hba1c)

gh_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L10_B.XPT', format='xport', encoding='utf-8')
gh_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L10_C.XPT', format='xport', encoding='utf-8')
gh_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/GHB_D.XPT', format='xport', encoding='utf-8')
gh_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/GHB_E.XPT', format='xport', encoding='utf-8')
gh_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/GHB_F.XPT', format='xport', encoding='utf-8')
gh_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/GHB_G.XPT', format='xport', encoding='utf-8')
gh_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/GHB_H.XPT', format='xport', encoding='utf-8')
gh_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GHB_I.XPT', format='xport', encoding='utf-8')
gh_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GHB_J.XPT', format='xport', encoding='utf-8')

gh_B = gh_B[['SEQN', 'LBXGH']]
gh_C = gh_C[['SEQN', 'LBXGH']]
gh_D = gh_D[['SEQN', 'LBXGH']]
gh_E = gh_E[['SEQN', 'LBXGH']]
gh_F = gh_F[['SEQN', 'LBXGH']]
gh_G = gh_G[['SEQN', 'LBXGH']]
gh_H = gh_H[['SEQN', 'LBXGH']]
gh_I = gh_I[['SEQN', 'LBXGH']]
gh_J = gh_J[['SEQN', 'LBXGH']]

gh = pd.concat([gh_B, gh_C, gh_D, gh_E, gh_F, gh_G, gh_H, gh_I, gh_J])
gh.rename(columns={'LBXGH':'hba1c_percent'}, inplace=True)

adults = adults.merge(gh, on='SEQN', how='left')

adults['diabetes_hba1c'] = np.where(adults['hba1c_percent'] >= 6.5, 'yes', 'no')

#Questionnaire data - taking insulin or glucose lowering meds

diq_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DIQ_B.XPT', format='xport', encoding='utf-8')
diq_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/DIQ_C.XPT', format='xport', encoding='utf-8')
diq_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/DIQ_D.XPT', format='xport', encoding='utf-8')
diq_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DIQ_E.XPT', format='xport', encoding='utf-8')
diq_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DIQ_F.XPT', format='xport', encoding='utf-8')
diq_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DIQ_G.XPT', format='xport', encoding='utf-8')
diq_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DIQ_H.XPT', format='xport', encoding='utf-8')
diq_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DIQ_I.XPT', format='xport', encoding='utf-8')
diq_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.XPT', format='xport', encoding='utf-8')

diq_B = diq_B[['SEQN', 'DIQ050', 'DIQ070']]
diq_C = diq_C[['SEQN', 'DIQ050', 'DIQ070']]
diq_D = diq_D[['SEQN', 'DIQ050', 'DID070']]
diq_E = diq_E[['SEQN', 'DIQ050', 'DID070']]
diq_F = diq_F[['SEQN', 'DIQ050', 'DIQ070']]
diq_G = diq_G[['SEQN', 'DIQ050', 'DIQ070']]
diq_H = diq_H[['SEQN', 'DIQ050', 'DIQ070']]
diq_I = diq_I[['SEQN', 'DIQ050', 'DIQ070']]
diq_J = diq_J[['SEQN', 'DIQ050', 'DIQ070']]

diq = pd.concat([diq_B, diq_C, diq_D, diq_E, diq_F, diq_G, diq_H, diq_I, diq_J])
diq.rename(columns={'DIQ050':'taking_insulin', 'DIQ070':'taking_diabetic_pills', 'DID070':'taking_diabetic_pills_D_E'}, inplace=True)

adults = adults.merge(diq, on='SEQN', how='left')

# recode levels for taking insulin / diabetes meds
adults['taking_insulin'] = adults['taking_insulin'].replace(1, 'yes')
adults['taking_insulin'] = adults['taking_insulin'].replace(2, 'no')
adults['taking_insulin'] = adults['taking_insulin'].replace([7, 9], 'unknown')

adults['taking_diabetic_pills'] = adults['taking_diabetic_pills'].replace(1, 'yes')
adults['taking_diabetic_pills'] = adults['taking_diabetic_pills'].replace(2, 'no')
adults['taking_diabetic_pills'] = adults['taking_diabetic_pills'].replace([7, 9], 'unknown')

adults['taking_diabetic_pills_D_E'] = adults['taking_diabetic_pills_D_E'].replace(1, 'yes')
adults['taking_diabetic_pills_D_E'] = adults['taking_diabetic_pills_D_E'].replace(2, 'no')
adults['taking_diabetic_pills_D_E'] = adults['taking_diabetic_pills_D_E'].replace([7, 9], 'unknown')

# create single feature for diabetes (yes/no)
adults['diabetes'] = np.where(adults['fasting_glc_mg_dL'] >= 126, 'yes', 'no')
adults['diabetes'] = np.where(adults['hba1c_percent'] >= 6.5, 'yes', adults['diabetes'])
adults['diabetes'] = np.where(adults['taking_insulin'] == 'yes', 'yes', adults['diabetes'])
adults['diabetes'] = np.where(adults['taking_diabetic_pills'] == 'yes', 'yes', adults['diabetes'])
adults['diabetes'] = np.where(adults['taking_diabetic_pills_D_E'] == 'yes', 'yes', adults['diabetes'])
adults['diabetes'] = np.where(adults['taking_insulin'] == 'unknown', 'unknown', adults['diabetes'])
adults['diabetes'] = np.where(adults['taking_diabetic_pills'] == 'unknown', 'unknown', adults['diabetes'])
adults['diabetes'] = np.where(adults['taking_diabetic_pills_D_E'] == 'unknown', 'unknown', adults['diabetes'])

adults['mets_fasting_glc'] = np.where(adults['fasting_glc_mg_dL'] >= 100, 'yes', 'no')
adults['mets_glucose'] = np.where(adults['taking_insulin'] == 'yes', 'yes', adults['mets_fasting_glc'])
adults['mets_glucose'] = np.where(adults['taking_diabetic_pills'] == 'yes', 'yes', adults['mets_glucose'])
adults['mets_glucose'] = np.where(adults['taking_diabetic_pills_D_E'] == 'yes', 'yes', adults['mets_glucose'])
adults['mets_glucose'] = np.where(adults['taking_insulin'] == 'unknown', 'unknown', adults['mets_glucose'])
adults['mets_glucose'] = np.where(adults['taking_diabetic_pills'] == 'unknown', 'unknown', adults['mets_glucose'])
adults['mets_glucose'] = np.where(adults['taking_diabetic_pills_D_E'] == 'unknown', 'unknown', adults['mets_glucose'])

adults = adults[adults['diabetes']!='unknown']

adults = adults.drop(columns=['mets_fasting_glc'])

# ### Questionnaire data - Prescription meds

rxq_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/RXQ_RX_B.XPT', format='xport', encoding='utf-8')
rxq_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/RXQ_RX_C.XPT', format='xport', encoding='utf-8')
rxq_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/RXQ_RX_D.XPT', format='xport', encoding='utf-8')
rxq_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/RXQ_RX_E.XPT', format='xport', encoding='utf-8')
rxq_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/RXQ_RX_F.XPT', format='xport', encoding='utf-8')
rxq_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/RXQ_RX_G.XPT', format='xport', encoding='utf-8')
rxq_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/RXQ_RX_H.XPT', format='xport', encoding='latin-1')
rxq_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/RXQ_RX_I.XPT', format='xport', encoding='utf-8')
rxq_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/RXQ_RX_J.XPT', format='xport', encoding='utf-8')

rxq_B.rename(columns={'RXD030':'RXDUSE', 'RXD240B':'RXDDRUG'}, inplace=True)

rxq_B = rxq_B[['SEQN', 'RXDUSE', 'RXDDRUG']]
rxq_C = rxq_C[['SEQN', 'RXDUSE', 'RXDDRUG']]
rxq_D = rxq_D[['SEQN', 'RXDUSE', 'RXDDRUG']]
rxq_E = rxq_E[['SEQN', 'RXDUSE', 'RXDDRUG']]
rxq_F = rxq_F[['SEQN', 'RXDUSE', 'RXDDRUG']]
rxq_G = rxq_G[['SEQN', 'RXDUSE', 'RXDDRUG']]
rxq_H = rxq_H[['SEQN', 'RXDUSE', 'RXDDRUG']]
rxq_I = rxq_I[['SEQN', 'RXDUSE', 'RXDDRUG']]
rxq_J = rxq_J[['SEQN', 'RXDUSE', 'RXDDRUG']]

# medication for lowering blood pressure

b_htn= rxq_B[rxq_B['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
b_htn = b_htn[b_htn['RXDUSE']==1]
c_htn= rxq_C[rxq_C['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
c_htn = c_htn[c_htn['RXDUSE']==1]
d_htn= rxq_D[rxq_D['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
d_htn = d_htn[d_htn['RXDUSE']==1]
e_htn= rxq_E[rxq_E['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
e_htn = e_htn[e_htn['RXDUSE']==1]
f_htn= rxq_F[rxq_F['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
f_htn = f_htn[f_htn['RXDUSE']==1]
g_htn= rxq_G[rxq_G['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
g_htn = g_htn[g_htn['RXDUSE']==1]
h_htn= rxq_H[rxq_H['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
h_htn = h_htn[h_htn['RXDUSE']==1]
i_htn= rxq_I[rxq_I['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
i_htn = i_htn[i_htn['RXDUSE']==1]
j_htn= rxq_J[rxq_J['RXDDRUG'].str.contains('lisinopril|amlodipine|losartan|hydrochlorothiazide|metoprolol|atenolol|Norvasc|carvedilol|Benicar|furosemide|clonidine|Cozaar|valsartan|Bystolic|Diovan|spironolactone|Toprol-XL|Lopressor|enalapril|hydralazine|Lasix|propranolol|Avapro|nifedipine|olmesartan|ramipril|Zestril|diltiazem|irbesartan|Prinivil|bisoprolol|chlorthalidone|Coreg|Katerzia|Norliqva|Vasotec|verapamil|amlodipine|benazepril|doxazosin|Maxzide|telmisartan|Aldactone|candesartan', case=False)]
j_htn =j_htn[j_htn['RXDUSE']==1]

htn_drug_use = pd.concat([b_htn, 
          c_htn,
          d_htn,
          e_htn,
          f_htn,
          g_htn,
          h_htn,
          i_htn,
          j_htn])

htn_drug_use = htn_drug_use.drop_duplicates(subset='SEQN')

htn_drug_use['htn_med'] = 'yes'

htn_drug_use = htn_drug_use[['SEQN', 'htn_med']]

# medication for lowering blood glucose

b_glu= rxq_B[rxq_B['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
b_glu = b_glu[b_glu['RXDUSE']==1]
c_glu= rxq_C[rxq_C['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
c_glu = c_glu[c_glu['RXDUSE']==1]
d_glu= rxq_D[rxq_D['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
d_glu = d_glu[d_glu['RXDUSE']==1]
e_glu= rxq_E[rxq_E['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
e_glu = e_glu[e_glu['RXDUSE']==1]
f_glu= rxq_F[rxq_F['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
f_glu = f_glu[f_glu['RXDUSE']==1]
g_glu= rxq_G[rxq_G['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
g_glu = g_glu[g_glu['RXDUSE']==1]
h_glu= rxq_H[rxq_H['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
h_glu = h_glu[h_glu['RXDUSE']==1]
i_glu= rxq_I[rxq_I['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
i_glu = i_glu[i_glu['RXDUSE']==1]
j_glu= rxq_J[rxq_J['RXDDRUG'].str.contains('Metformin|Sulfonylurea|Dipeptidyl peptidase-4 inhibitor|Meglitinide|SGLT2|Thiazolidinedione|Alpha-glucosidase inhibitor|Insulin|Glp-1|Biguanide|Pioglitazone|Acarbose|Alogliptin|Amylin|Glipizide|Incretin', case=False)]
j_glu =j_glu[j_glu['RXDUSE']==1]

glu_drug_use = pd.concat([b_glu, 
          c_glu,
          d_glu,
          e_glu,
          f_glu,
          g_glu,
          h_glu,
          i_glu,
          j_glu])

glu_drug_use = glu_drug_use.drop_duplicates(subset='SEQN')

glu_drug_use['glu_med'] = 'yes'

glu_drug_use = glu_drug_use[['SEQN', 'glu_med']]

# medication for treating low HDL

b_hdl= rxq_B[rxq_B['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
b_hdl = b_hdl[b_hdl['RXDUSE']==1]
c_hdl= rxq_C[rxq_C['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
c_hdl = c_hdl[c_hdl['RXDUSE']==1]
d_hdl= rxq_D[rxq_D['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
d_hdl = d_hdl[d_hdl['RXDUSE']==1]
e_hdl= rxq_E[rxq_E['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
e_hdl = e_hdl[e_hdl['RXDUSE']==1]
f_hdl= rxq_F[rxq_F['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
f_hdl = f_hdl[f_hdl['RXDUSE']==1]
g_hdl= rxq_G[rxq_G['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
g_hdl = g_hdl[g_hdl['RXDUSE']==1]
h_hdl= rxq_H[rxq_H['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
h_hdl = h_hdl[h_hdl['RXDUSE']==1]
i_hdl= rxq_I[rxq_I['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
i_hdl = i_hdl[i_hdl['RXDUSE']==1]
j_hdl= rxq_J[rxq_J['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
j_hdl =j_hdl[j_hdl['RXDUSE']==1]

hdl_drug_use = pd.concat([b_hdl, 
          c_hdl,
          d_hdl,
          e_hdl,
          f_hdl,
          g_hdl,
          h_hdl,
          i_hdl,
          j_hdl])

hdl_drug_use = hdl_drug_use.drop_duplicates(subset='SEQN')

hdl_drug_use['hdl_med'] = 'yes'

hdl_drug_use = hdl_drug_use[['SEQN', 'hdl_med']]

# medication for treating elevated TG (same as HDL)

b_tg= rxq_B[rxq_B['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
b_tg = b_tg[b_tg['RXDUSE']==1]
c_tg= rxq_C[rxq_C['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
c_tg = c_tg[c_tg['RXDUSE']==1]
d_tg= rxq_D[rxq_D['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
d_tg = d_tg[d_tg['RXDUSE']==1]
e_tg= rxq_E[rxq_E['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
e_tg = e_tg[e_tg['RXDUSE']==1]
f_tg= rxq_F[rxq_F['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
f_tg = f_tg[f_tg['RXDUSE']==1]
g_tg= rxq_G[rxq_G['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
g_tg = g_tg[g_tg['RXDUSE']==1]
h_tg= rxq_H[rxq_H['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
h_tg = h_tg[h_tg['RXDUSE']==1]
i_tg= rxq_I[rxq_I['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
i_tg = i_tg[i_tg['RXDUSE']==1]
j_tg= rxq_J[rxq_J['RXDDRUG'].str.contains('fibrate|fenofibric acid|gemfibrozil|statin|ezetimibe|niacin', case=False)]
j_tg =j_tg[j_tg['RXDUSE']==1]

tg_drug_use = pd.concat([b_tg, 
          c_tg,
          d_tg,
          e_tg,
          f_tg,
          g_tg,
          h_tg,
          i_tg,
          j_tg])

tg_drug_use = tg_drug_use.drop_duplicates(subset='SEQN')

tg_drug_use['tg_med'] = 'yes'

tg_drug_use = tg_drug_use[['SEQN', 'tg_med']]

# ### HDL cholesterol

#Lab data - HDL cholesterol

hdl_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L13_B.XPT', format='xport', encoding='utf-8')
hdl_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L13_C.XPT', format='xport', encoding='utf-8')
hdl_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/HDL_D.XPT', format='xport', encoding='utf-8')
hdl_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/HDL_E.XPT', format='xport', encoding='utf-8')
hdl_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/HDL_F.XPT', format='xport', encoding='utf-8')
hdl_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/HDL_G.XPT', format='xport', encoding='utf-8')
hdl_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/HDL_H.XPT', format='xport', encoding='utf-8')
hdl_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HDL_I.XPT', format='xport', encoding='utf-8')
hdl_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HDL_J.XPT', format='xport', encoding='utf-8')

hdl_B = hdl_B[['SEQN', 'LBDHDL']]
hdl_C = hdl_C[['SEQN', 'LBXHDD']]
hdl_D = hdl_D[['SEQN', 'LBDHDD']]
hdl_E = hdl_E[['SEQN', 'LBDHDD']]
hdl_F = hdl_F[['SEQN', 'LBDHDD']]
hdl_G = hdl_G[['SEQN', 'LBDHDD']]
hdl_H = hdl_H[['SEQN', 'LBDHDD']]
hdl_I = hdl_I[['SEQN', 'LBDHDD']]
hdl_J = hdl_J[['SEQN', 'LBDHDD']]

hdl_B.rename(columns={'LBDHDL':'LBDHDD'}, inplace=True)
hdl_C.rename(columns={'LBXHDD':'LBDHDD'}, inplace=True)
hdl = pd.concat([hdl_B, hdl_C, hdl_D, hdl_E, hdl_F, hdl_G, hdl_H, hdl_I, hdl_J])
hdl.rename(columns={'LBDHDD':'hdl_mg_dL'}, inplace=True)
adults = adults.merge(hdl, on='SEQN', how='left')

# ### Serum lipids

#Lab data - serum lipids (triglycerides)

tg_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L13AM_B.XPT', format='xport', encoding='utf-8')
tg_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L13AM_C.XPT', format='xport', encoding='utf-8')
tg_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/TRIGLY_D.XPT', format='xport', encoding='utf-8')
tg_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/TRIGLY_E.XPT', format='xport', encoding='utf-8')
tg_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/TRIGLY_F.XPT', format='xport', encoding='utf-8')
tg_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/TRIGLY_G.XPT', format='xport', encoding='utf-8')
tg_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/TRIGLY_H.XPT', format='xport', encoding='utf-8')
tg_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/TRIGLY_I.XPT', format='xport', encoding='utf-8')
tg_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TRIGLY_J.XPT', format='xport', encoding='utf-8')

tg_B = tg_B[['SEQN', 'LBXTR']]
tg_C = tg_C[['SEQN', 'LBXTR']]
tg_D = tg_D[['SEQN', 'LBXTR']]
tg_E = tg_E[['SEQN', 'LBXTR']]
tg_F = tg_F[['SEQN', 'LBXTR']]
tg_G = tg_G[['SEQN', 'LBXTR']]
tg_H = tg_H[['SEQN', 'LBXTR']]
tg_I = tg_I[['SEQN', 'LBXTR']]
tg_J = tg_J[['SEQN', 'LBXTR']]

tg = pd.concat([tg_B, tg_C, tg_D, tg_E, tg_F, tg_G, tg_H, tg_I, tg_J])
tg.rename(columns={'LBXTR':'tg_mg_dL'}, inplace=True)

adults = adults.merge(tg, on='SEQN', how='left')

# ### Hypertension

#Exam data - blood pressure (hypertension)

bp_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BPX_B.XPT', format='xport', encoding='utf-8')
bp_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/BPX_C.XPT', format='xport', encoding='utf-8')
bp_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BPX_D.XPT', format='xport', encoding='utf-8')
bp_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BPX_E.XPT', format='xport', encoding='utf-8')
bp_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BPX_F.XPT', format='xport', encoding='utf-8')
bp_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BPX_G.XPT', format='xport', encoding='utf-8')
bp_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPX_H.XPT', format='xport', encoding='utf-8')
bp_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.XPT', format='xport', encoding='utf-8')
bp_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.XPT', format='xport', encoding='utf-8')

bp_sys_B = bp_B[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
bp_sys_C = bp_C[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
bp_sys_D = bp_D[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
bp_sys_E = bp_E[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
bp_sys_F = bp_F[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
bp_sys_G = bp_G[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
bp_sys_H = bp_H[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
bp_sys_I = bp_I[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
bp_sys_J = bp_J[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]

bp_sys = pd.concat([bp_sys_B, bp_sys_C, bp_sys_D, bp_sys_E, bp_sys_F, bp_sys_G, bp_sys_H, bp_sys_I, bp_sys_J])
bp_sys.set_index('SEQN', inplace=True)

bp_sys['sys_mean'] = bp_sys.mean(axis=1)

bp_sys = bp_sys.reset_index()[['SEQN', 'sys_mean']]

bp_sys['sys_ht'] = np.where(bp_sys['sys_mean'] >= 140, 'yes', 'no')

#bp_sys.drop(columns='sys_mean', inplace=True)

adults = adults.merge(bp_sys, on='SEQN', how='left')

bp_di_B = bp_B[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
bp_di_C = bp_C[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
bp_di_D = bp_D[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
bp_di_E = bp_E[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
bp_di_F = bp_F[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
bp_di_G = bp_G[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
bp_di_H = bp_H[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
bp_di_I = bp_I[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
bp_di_J = bp_J[['SEQN', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]

bp_di = pd.concat([bp_di_B, bp_di_C, bp_di_D, bp_di_E, bp_di_F, bp_di_G, bp_di_H, bp_di_I, bp_di_J])
bp_di.set_index('SEQN', inplace=True)

bp_di['di_mean'] = bp_di.mean(axis=1)

bp_di = bp_di.reset_index()[['SEQN', 'di_mean']]

bp_di['di_ht'] = np.where(bp_di['di_mean'] >= 90, 'yes', 'no')

#bp_di.drop(columns='di_mean', inplace=True)

adults = adults.merge(bp_di, on='SEQN', how='left')

#Questionnaire data - blood pressure (hypertension)

bpq_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BPQ_B.XPT', format='xport', encoding='utf-8')
bpq_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/BPQ_C.XPT', format='xport', encoding='utf-8')
bpq_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BPQ_D.XPT', format='xport', encoding='utf-8')
bpq_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BPQ_E.XPT', format='xport', encoding='utf-8')
bpq_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BPQ_F.XPT', format='xport', encoding='utf-8')
bpq_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BPQ_G.XPT', format='xport', encoding='utf-8')
bpq_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPQ_H.XPT', format='xport', encoding='utf-8')
bpq_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPQ_I.XPT', format='xport', encoding='utf-8')
bpq_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPQ_J.XPT', format='xport', encoding='utf-8')

bpq_B = bpq_B[['SEQN', 'BPQ040A', 'BPQ050A']]
bpq_C = bpq_C[['SEQN', 'BPQ040A', 'BPQ050A']]
bpq_D = bpq_D[['SEQN', 'BPQ040A', 'BPQ050A']]
bpq_E = bpq_E[['SEQN', 'BPQ040A', 'BPQ050A']]
bpq_F = bpq_F[['SEQN', 'BPQ040A', 'BPQ050A']]
bpq_G = bpq_G[['SEQN', 'BPQ040A', 'BPQ050A']]
bpq_H = bpq_H[['SEQN', 'BPQ040A', 'BPQ050A']]
bpq_I = bpq_I[['SEQN', 'BPQ040A', 'BPQ050A']]
bpq_J = bpq_J[['SEQN', 'BPQ040A', 'BPQ050A']]

bpq = pd.concat([bpq_B, bpq_C, bpq_D, bpq_E, bpq_F, bpq_G, bpq_H, bpq_I, bpq_J])

adults = adults.merge(bpq, on='SEQN', how='left')

adults['hypertension'] = np.where(adults['sys_ht'] == 'yes', 'yes', 'no')
adults['hypertension'] = np.where(adults['di_ht'] == 'yes', 'yes', adults['hypertension'])
adults['hypertension'] = np.where(adults['BPQ040A'] == 1, 'yes', adults['hypertension'])
adults['hypertension'] = np.where(adults['BPQ050A'] == 1, 'yes', adults['hypertension'])
adults['hypertension'] = np.where(adults['BPQ040A'] == 7, 'unknown', adults['hypertension'])
adults['hypertension'] = np.where(adults['BPQ050A'] == 7, 'unknown', adults['hypertension'])
adults['hypertension'] = np.where(adults['BPQ040A'] == 9, 'unknown', adults['hypertension'])
adults['hypertension'] = np.where(adults['BPQ050A'] == 9, 'unknown', adults['hypertension'])

adults['hypertension'].value_counts()

adults = adults[adults['hypertension']!='unknown']

# ### Exclusion criteria - infectious diseases, CVD & Cancer

# #### infectious diseases (Hepatitis / HIV)
# * Hep A - see note
# * Hep B
# * Hep C
# * Hep D
# * HIV

# Note on Hep A - Discussion with DL suggested use of elevated liver enzymes to detect true cases of active infection rather than ABs as a result of vaccination. However, there can be several causes of elevated enzymes aside from Hepatitis: "When Alanine transaminase (ALT) rises to more than 500 IU/L, causes are usually from the liver. It can be due to hepatitis, ischemic liver injury, and toxins that causes liver damage. The ALT levels in hepatitis C rises more than in hepatitis A and B. Persistent ALT elevation more than 6 months is known as chronic hepatitis. Alcoholic liver disease, non-alcoholic fatty liver disease (NAFLD), fat accumulation in liver during childhood obesity, steatohepatitis (inflammation of fatty liver disease) are associated with a rise in ALT." https://en.wikipedia.org/wiki/Liver_function_tests
# 
# For this reason, not current excluding individuals with positive result for Hep A.

#Lab data - Hepatitis B core antibody / surface antigen

hep_BCD_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L02_B.XPT', format='xport', encoding='utf-8')
hep_BCD_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L02_C.XPT', format='xport', encoding='utf-8')

hep_B_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/HEPBD_D.XPT', format='xport', encoding='utf-8')
hep_B_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/HEPBD_E.XPT', format='xport', encoding='utf-8')
hep_B_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/HEPBD_F.XPT', format='xport', encoding='utf-8')
hep_B_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/HEPBD_G.XPT', format='xport', encoding='utf-8')
hep_B_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/HEPBD_H.XPT', format='xport', encoding='utf-8')
hep_B_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HEPBD_I.XPT', format='xport', encoding='utf-8')
hep_B_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HEPBD_J.XPT', format='xport', encoding='utf-8')

#Hepatitis B surface antibody

hep_SA_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L02HBS_B.XPT', format='xport', encoding='utf-8')
hep_SA_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L02HBS_C.XPT', format='xport', encoding='utf-8')
hep_SA_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/HEPB_S_D.XPT', format='xport', encoding='utf-8')
hep_SA_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/HEPB_S_E.XPT', format='xport', encoding='utf-8')
hep_SA_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/HEPB_S_F.XPT', format='xport', encoding='utf-8')
hep_SA_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/HEPB_S_G.XPT', format='xport', encoding='utf-8')
hep_SA_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/HEPB_S_H.XPT', format='xport', encoding='utf-8')
hep_SA_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HEPB_S_I.XPT', format='xport', encoding='utf-8')
hep_SA_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HEPB_S_J.XPT', format='xport', encoding='utf-8')

hepB_B = hep_BCD_B[['SEQN', 'LBXHBC', 'LBDHBG']]
hepB_C = hep_BCD_C[['SEQN', 'LBXHBC', 'LBDHBG']]
hepB_D = hep_B_D[['SEQN', 'LBXHBC', 'LBDHBG']]
hepB_E = hep_B_E[['SEQN', 'LBXHBC', 'LBDHBG']]
hepB_F = hep_B_F[['SEQN', 'LBXHBC', 'LBDHBG']]
hepB_G = hep_B_G[['SEQN', 'LBXHBC', 'LBDHBG']]
hepB_H = hep_B_H[['SEQN', 'LBXHBC', 'LBDHBG']]
hepB_I = hep_B_I[['SEQN', 'LBXHBC', 'LBDHBG']]
hepB_J = hep_B_J[['SEQN', 'LBXHBC', 'LBDHBG']]

hepB_SA_B = hep_SA_B[['SEQN', 'LBXHBS']]
hepB_SA_C = hep_SA_C[['SEQN', 'LBXHBS']]
hepB_SA_D = hep_SA_D[['SEQN', 'LBXHBS']]
hepB_SA_E = hep_SA_E[['SEQN', 'LBXHBS']]
hepB_SA_F = hep_SA_F[['SEQN', 'LBXHBS']]
hepB_SA_G = hep_SA_G[['SEQN', 'LBXHBS']]
hepB_SA_H = hep_SA_H[['SEQN', 'LBXHBS']]
hepB_SA_I = hep_SA_I[['SEQN', 'LBXHBS']]
hepB_SA_J = hep_SA_J[['SEQN', 'LBXHBS']]

hep_B_CSA = pd.concat([hepB_B, hepB_C, hepB_D, hepB_E, hepB_F, hepB_G, hepB_H, hepB_I, hepB_J])
hep_B_SA = pd.concat([hepB_SA_B, hepB_SA_C, hepB_SA_D, hepB_SA_E, hepB_SA_F, hepB_SA_G, hepB_SA_H, hepB_SA_I, hepB_SA_J])

adults = adults.merge(hep_B_CSA, on='SEQN', how='left')
adults = adults.merge(hep_B_SA, on='SEQN', how='left')

adults['hep_B_infection'] = np.where(adults['LBXHBC'] == 1, 'yes', 'no')
adults['hep_B_infection'] = np.where(adults['LBDHBG'] == 1, 'yes', adults['hep_B_infection'])
adults['hep_B_infection'] = np.where(adults['LBXHBS'] == 1, 'no', adults['hep_B_infection']) # testing positive for surface antibody indicates immunity from vaccination, not active infection.

adults = adults[adults['hep_B_infection'] == 'no']

#Lab data - Hepatitis C / D

hepC_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/HEPC_D.XPT', format='xport', encoding='utf-8')
hepD_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/HEPBD_D.XPT', format='xport', encoding='utf-8')

hepC_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/HEPC_E.XPT', format='xport', encoding='utf-8')
hepD_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/HEPBD_E.XPT', format='xport', encoding='utf-8')

hepC_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/HEPC_F.XPT', format='xport', encoding='utf-8')
hepD_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/HEPBD_F.XPT', format='xport', encoding='utf-8')

hepC_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/HEPC_G.XPT', format='xport', encoding='utf-8')
hepD_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/HEPBD_G.XPT', format='xport', encoding='utf-8')

hepC_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SSHEPC_H.XPT', format='xport', encoding='utf-8')
hepD_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/HEPBD_H.XPT', format='xport', encoding='utf-8')

hepC_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HEPC_I.XPT', format='xport', encoding='utf-8')
hepD_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HEPBD_I.XPT', format='xport', encoding='utf-8')

hepC_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HEPC_J.XPT', format='xport', encoding='utf-8')
hepD_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HEPBD_J.XPT', format='xport', encoding='utf-8')

hepC_B = hep_BCD_B[['SEQN', 'LBDHCV']]
hepC_C = hep_BCD_C[['SEQN', 'LBDHCV']]
hepC_D = hepC_D[['SEQN', 'LBDHCV']]
hepC_E = hepC_E[['SEQN', 'LBDHCV']]
hepC_F = hepC_F[['SEQN', 'LBDHCV']]
hepC_G = hepC_G[['SEQN', 'LBDHCV']]
hepC_H = hepC_H[['SEQN', 'LBDHCI']]
hepC_I = hepC_I[['SEQN', 'LBXHCR']] # discontinued testing kit, using 'LBXHCR' as assessment
hepC_J = hepC_J[['SEQN', 'LBDHCI']]

hepC_H.rename(columns={'LBDHCI':'LBDHCV'},inplace=True)
hepC_I.rename(columns={'LBXHCR':'LBDHCV'},inplace=True)
hepC_J.rename(columns={'LBDHCI':'LBDHCV'},inplace=True)

hepD_B = hep_BCD_B[['SEQN', 'LBDHD']]
hepD_C = hep_BCD_C[['SEQN', 'LBDHD']]
hepD_D = hepD_D[['SEQN', 'LBDHD']]
hepD_E = hepD_E[['SEQN', 'LBDHD']]
hepD_F = hepD_F[['SEQN', 'LBDHD']]
hepD_G = hepD_G[['SEQN', 'LBDHD']]
hepD_H = hepD_H[['SEQN', 'LBDHD']]
hepD_I = hepD_I[['SEQN', 'LBDHD']]
hepD_J = hepD_J[['SEQN', 'LBDHD']]

hep_C = pd.concat([hepC_B, hepC_C, hepC_D, hepC_E, hepC_F, hepC_G, hepC_H, hepC_I, hepC_J])
hep_D = pd.concat([hepD_B, hepD_C, hepD_D, hepD_E, hepD_F, hepD_G, hepD_H, hepD_I, hepD_J])

adults = adults.merge(hep_C, on='SEQN', how='left')
adults = adults.merge(hep_D, on='SEQN', how='left')

adults['hep_C_infection'] = np.where(adults['LBDHCV'] == 1, 'yes', 'no')
adults['hep_D_infection'] = np.where(adults['LBDHD'] == 1, 'yes', 'no')

adults = adults[adults['hep_C_infection'] == 'no']
adults = adults[adults['hep_D_infection'] == 'no']

#Lab data - HIV

hiv_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L03_B.XPT', format='xport', encoding='utf-8')
hiv_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L03_C.XPT', format='xport', encoding='utf-8')
hiv_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/hiv_D.XPT', format='xport', encoding='utf-8')
hiv_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/hiv_E.XPT', format='xport', encoding='utf-8')
hiv_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/hiv_F.XPT', format='xport', encoding='utf-8')
hiv_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/hiv_G.XPT', format='xport', encoding='utf-8')
hiv_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/hiv_H.XPT', format='xport', encoding='utf-8')
hiv_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/hiv_I.XPT', format='xport', encoding='utf-8')
hiv_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/hiv_J.XPT', format='xport', encoding='utf-8')

hiv_I = hiv_I[hiv_I['LBXHIVC'] == 1]
hiv_I = hiv_I[hiv_I['LBXHNAT'] != 2]
hiv_I.rename(columns={'LBXHIVC':'LBDHI'}, inplace=True)
hiv_I = hiv_I[['SEQN', 'LBDHI']]

hiv_J = hiv_J[hiv_J['LBXHIVC'] == 1]
hiv_J = hiv_J[hiv_J['LBXHNAT'] != 2]
hiv_J.rename(columns={'LBXHIVC':'LBDHI'}, inplace=True)
hiv_J = hiv_J[['SEQN', 'LBDHI']]

hiv = pd.concat([hiv_B, hiv_C, hiv_D, hiv_E, hiv_F, hiv_G, hiv_H, hiv_I, hiv_J])

hiv_pos = hiv[hiv['LBDHI']==1]

adults = adults[~adults['SEQN'].isin(hiv_pos['SEQN'])]

#Questionnaire data - CVD / Cancer

mc_B = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/MCQ_B.XPT', format='xport', encoding='utf-8')
mc_C = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/MCQ_C.XPT', format='xport', encoding='utf-8')
mc_D = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/MCQ_D.XPT', format='xport', encoding='utf-8')
mc_E = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/MCQ_E.XPT', format='xport', encoding='utf-8')
mc_F = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/MCQ_F.XPT', format='xport', encoding='utf-8')
mc_G = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/MCQ_G.XPT', format='xport', encoding='utf-8')
mc_H = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/MCQ_H.XPT', format='xport', encoding='utf-8')
mc_I = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.XPT', format='xport', encoding='utf-8')
mc_J = pd.read_sas('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/MCQ_J.XPT', format='xport', encoding='utf-8')

mc_B = mc_B[['SEQN', 'MCQ180C', 'MCQ220']]
mc_C = mc_C[['SEQN', 'MCQ180C', 'MCQ220']]
mc_D = mc_D[['SEQN', 'MCQ180C', 'MCQ220']]
mc_E = mc_E[['SEQN', 'MCQ180C', 'MCQ220']]
mc_F = mc_F[['SEQN', 'MCQ180C', 'MCQ220']]
mc_G = mc_G[['SEQN', 'MCQ180C', 'MCQ220']]
mc_H = mc_H[['SEQN', 'MCQ180C', 'MCQ220']]
mc_I = mc_I[['SEQN', 'MCQ180C', 'MCQ220']]
mc_J = mc_J[['SEQN', 'MCQ160C', 'MCQ220']]
mc_J.rename(columns={'MCQ160C':'MCQ180C'}, inplace=True)

mc = pd.concat([mc_B, mc_C, mc_D, mc_E, mc_F, mc_G, mc_H, mc_I, mc_J])

adults = adults.merge(mc, on='SEQN', how='left')

adults['MCQ180C'] = np.where(adults['MCQ180C'] < 85, 'yes', 'no')
adults['MCQ220'] = np.where(adults['MCQ220'] == 1, 'yes', 'no')

adults = adults[adults['MCQ180C'] != 'yes']
adults = adults[adults['MCQ220'] != 'yes']

adults_tg = adults.merge(tg_drug_use, on='SEQN', how='left')

adults_tg['tg_med'] = np.where(adults_tg['tg_med'] == 'yes', 1, 0)

adults_htn = adults_tg.merge(htn_drug_use, on='SEQN', how='left')

adults_htn['htn_med'] = np.where(adults_htn['htn_med'] == 'yes', 1, 0)

adults_glu = adults_htn.merge(glu_drug_use, on='SEQN', how='left')

adults_glu['glu_med'] = np.where(adults_glu['glu_med'] == 'yes', 1, 0)

adults_hdl = adults_glu.merge(hdl_drug_use, on='SEQN', how='left')

adults_hdl['hdl_med'] = np.where(adults_hdl['hdl_med'] == 'yes', 1, 0)

adults_hdl.columns

adults_hdl.SEQN.nunique()

adults_hdl.rename(columns={'RIDRETH1':'Ethnicity', 'INDFMPIR': 'family_pir'},inplace=True)

adults_hdl = adults_hdl[['SEQN', 'ingred_code', 'ingred_desc', 'Ingred_consumed_g', 'Sex', 'Age', 'Ethnicity',
                         'family_pir', 'education', 'BMI', 'Energy', 'Fatty acids, total monounsaturated',	'Fatty acids, total polyunsaturated',
                         'Fatty acids, total saturated', 'Sodium', 'Carbohydrate', 'Fiber, total dietary',
                         'ever_smoker', 'diabetes', 'hypertension', 'WTINT2YR', 'WTMEC2YR', 'SDMVPSU',
                         'SDMVSTRA', 'CYCLE', 'diet_wts']]

adults_hdl['diet_wts'] = adults_hdl['diet_wts'] / 9 # calculate sample weights based on number of included cycles

energy = adults_hdl.groupby('SEQN')['Energy'].agg("sum")

energy = energy.reset_index()

energy = energy[energy['Energy']>500]

energy = energy[energy['Energy']<4500]

e_cent = energy[energy['Energy']> np.percentile(energy['Energy'], 5)] 

e_cent = e_cent[e_cent['Energy']< np.percentile(energy['Energy'], 95)] 

adults_qc = adults_hdl[adults_hdl['SEQN'].isin(e_cent['SEQN'])] 

adults_qc = adults_qc.dropna()

output_dir = Path('../data/ingredient_list/')
output_dir.mkdir(exist_ok=True, parents=True)

# generate list of all unique ingredients to create mapping to DFG2 (src/00_map_databases/dfg2/00a and 00b)
ingredients = adults_qc[['ingred_code', 'ingred_desc']]
ingredients.drop_duplicates(subset='ingred_code').to_csv('../data/ingredient_list/ingred_list.csv', index=None)

