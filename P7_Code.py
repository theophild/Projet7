#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
datatest=pd.read_csv('application_test.csv')
datatest


# In[2]:


datatrain=pd.read_csv('application_train.csv')
datatrain


# In[3]:


list(datatrain)


# In[4]:


databureau=pd.read_csv('bureau.csv')
databureau


# In[5]:


list(databureau)


# In[6]:


databalance=pd.read_csv('bureau_balance.csv')
databalance


# In[7]:


datacc=pd.read_csv('credit_card_balance.csv')
datacc


# In[8]:


list(datacc)


# In[9]:


datahc=pd.read_csv('HomeCredit_columns_description.csv',sep=';')
datahc


# In[10]:


datapayments=pd.read_csv('installments_payments.csv')
datapayments


# In[11]:


datacash=pd.read_csv('POS_CASH_balance.csv')
datacash


# In[12]:


dataprevious=pd.read_csv('previous_application.csv')
dataprevious


# In[13]:


list(dataprevious)


# In[14]:


datasample=pd.read_csv('sample_submission.csv')
datasample


# In[15]:


# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('bureau.csv', nrows = num_rows)
    bb = pd.read_csv('bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        global clf
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
                max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


# def main(debug = False):
#     num_rows = 10000 if debug else None
#     df = application_train_test(num_rows)
#     import re
#     df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#     with timer("Process bureau and bureau_balance"):
#         bureau = bureau_and_balance(num_rows)
#         print("Bureau df shape:", bureau.shape)
#         df = df.join(bureau, how='left', on='SK_ID_CURR')
#         df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#         del bureau
#         gc.collect()
#     with timer("Process previous_applications"):
#         prev = previous_applications(num_rows)
#         print("Previous applications df shape:", prev.shape)
#         df = df.join(prev, how='left', on='SK_ID_CURR')
#         df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#         del prev
#         gc.collect()
#     with timer("Process POS-CASH balance"):
#         pos = pos_cash(num_rows)
#         print("Pos-cash balance df shape:", pos.shape)
#         df = df.join(pos, how='left', on='SK_ID_CURR')
#         df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#         del pos
#         gc.collect()
#     with timer("Process installments payments"):
#         ins = installments_payments(num_rows)
#         print("Installments payments df shape:", ins.shape)
#         df = df.join(ins, how='left', on='SK_ID_CURR')
#         df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#         del ins
#         gc.collect()
#     with timer("Process credit card balance"):
#         cc = credit_card_balance(num_rows)
#         print("Credit card balance df shape:", cc.shape)
#         df = df.join(cc, how='left', on='SK_ID_CURR')
#         df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#         del cc
#         gc.collect()
#     with timer("Run LightGBM with kfold"):
#         feat_importance = kfold_lightgbm(df, num_folds= 10, stratified= False, debug= debug)
# 
# if __name__ == "__main__":
#     submission_file_name = "submission_kernel02.csv"
#     with timer("Full model run"):
#         main()

# In[16]:


num_rows = len(datatrain)
df = application_train_test(num_rows)
import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
with timer("Process bureau and bureau_balance"):
    bureau = bureau_and_balance(num_rows)
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    del bureau
    gc.collect()
with timer("Process previous_applications"):
    prev = previous_applications(num_rows)
    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    del prev
    gc.collect()
with timer("Process POS-CASH balance"):
    pos = pos_cash(num_rows)
    print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    del pos
    gc.collect()
with timer("Process installments payments"):
    ins = installments_payments(num_rows)
    print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    del ins
    gc.collect()
with timer("Process credit card balance"):
    cc = credit_card_balance(num_rows)
    print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    del cc
    gc.collect()


# from sklearn import decomposition
# from sklearn import preprocessing
# from functions import *
# 
# # choix du nombre de composantes à calculer
# n_comp = 6
# 
# 
# # selection des colonnes à prendre en compte dans l'ACP
# data_pca=df[['SK_ID_CURR', 'PAYMENT_RATE', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_BIRTH', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'ACTIVE_DAYS_CREDIT_MAX', 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'APPROVED_CNT_PAYMENT_MEAN', 'INSTAL_DPD_MEAN', 'DAYS_EMPLOYED_PERC', 'ANNUITY_INCOME_PERC', 'DAYS_REGISTRATION', 'REGION_POPULATION_RELATIVE', 'AMT_GOODS_PRICE', 'CLOSED_DAYS_CREDIT_ENDDATE_MAX', 'AMT_CREDIT', 'CLOSED_DAYS_CREDIT_MAX', 'INSTAL_AMT_PAYMENT_SUM', 'PREV_CNT_PAYMENT_MEAN', 'APPROVED_DAYS_DECISION_MAX', 'INSTAL_DBD_SUM', 'INCOME_CREDIT_PERC', 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'OWN_CAR_AGE', 'CLOSED_AMT_CREDIT_SUM_MEAN', 'POS_MONTHS_BALANCE_SIZE', 'DAYS_LAST_PHONE_CHANGE', 'ACTIVE_DAYS_CREDIT_ENDDATE_MAX', 'ACTIVE_DAYS_CREDIT_UPDATE_MEAN', 'INSTAL_AMT_PAYMENT_MIN', 'ACTIVE_AMT_CREDIT_SUM_SUM', 'BURO_DAYS_CREDIT_ENDDATE_MAX', 'INSTAL_DBD_MAX', 'BURO_DAYS_CREDIT_MAX', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'ACTIVE_DAYS_CREDIT_ENDDATE_MEAN']]
# # préparation des données pour l'ACP
# data_pca = data_pca.fillna(data_pca.mean()) # Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable
# X = data_pca.values
# names = data_pca["SK_ID_CURR"] # ou data.index pour avoir les intitulés
# features = data_pca.columns
# 
# # Centrage et Réduction
# std_scale = preprocessing.StandardScaler().fit(X)
# X_scaled = std_scale.transform(X)
# 
# # Calcul des composantes principales
# pca = decomposition.PCA(n_components=n_comp)
# pca.fit(X_scaled)
# 
# # Eboulis des valeurs propres
# display_scree_plot(pca)
# 
# # Cercle des corrélations
# pcs = pca.components_
# display_circles(pcs, n_comp, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))
# 
# # Projection des individus
# X_projected = pca.transform(X_scaled)
# display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)])
# 
# plt.show()

# In[18]:


df2=df
df2.index=df2['SK_ID_CURR']
df2


# df2=df2.fillna(0)
# df2

# In[19]:


df2['TARGET']=df['TARGET']


# In[20]:


X=df2.drop(columns=['SK_ID_CURR'])
y=df2[['TARGET']]


# In[21]:


from sklearn.model_selection import train_test_split
past_df=df2[df2['TARGET'].notnull()]
train_df, test_df = train_test_split(df2[df2['TARGET'].notnull()], test_size=0.33)
ftest_df = df2[df2['TARGET'].isnull()]
ftest_df


# In[22]:


past_df


# In[23]:


X_past=past_df.drop(columns=[ 'TARGET'])
y_past=past_df[['TARGET']]


# In[24]:


test_df


# In[25]:


# class count
class_count_0, class_count_1 = past_df['TARGET'].value_counts()

# Separate class
class_0 = past_df[past_df['TARGET'] == 0]
class_1 = past_df[past_df['TARGET'] == 1] # print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)


# In[26]:


class_0_under = class_0.sample(class_count_1)

test_under = pd.concat([class_0_under, class_1], axis=0)

print("total class of 1 and 0:",test_under['TARGET'].value_counts())# plot the count after under-sampeling
test_under['TARGET'].value_counts().plot(kind='bar', title='count (target)')


# In[27]:


test_under = pd.concat([test_under, ftest_df], axis=0)


# # import all the required libraries
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.linear_model import LogisticRegression
# 
# 
# # define model
# model = LogisticRegression(class_weight='balanced')
# 
# # define cross-validation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 
# # evaluate model
# scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# 
# # summarize performance
# print('Mean AUROC: %.3f' % np.mean(scores))

# submission_file_name = "submission_kernel01.csv"
# with timer("Run LightGBM with kfold"):
#     feat_importance = kfold_lightgbm(test_under, num_folds= 10, stratified= False)

# In[28]:


import collections
import imblearn
from imblearn.under_sampling import RandomUnderSampler


rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
x_rus, y_rus = rus.fit_resample(X_past, y_past)

print("total class of 1 and 0:",y_past['TARGET'].value_counts())# plot the count after under-sampeling
y_past['TARGET'].value_counts().plot(kind='bar', title='count (target)')


# In[29]:


print("total class of 1 and 0:",y_rus['TARGET'].value_counts())# plot the count after under-sampeling
y_rus['TARGET'].value_counts().plot(kind='bar', title='count (target)')


# In[30]:


datarus=x_rus.copy()
datarus['TARGET']=y_rus
datarus


# In[31]:


datarus2 = pd.concat([datarus, ftest_df], axis=0)
datarus2


# submission_file_name = "submission_kernel02.csv"
# with timer("Run LightGBM with kfold"):
#     feat_importance = kfold_lightgbm(datarus2, num_folds= 10, stratified= False)

# In[32]:


datarus.index=datarus['SK_ID_CURR']
datarus


# In[33]:


train_df, test_df = train_test_split(datarus, test_size=0.33)


# In[34]:


X_train=train_df.drop(columns=[ 'TARGET'])
y_train=train_df[['TARGET']]


# In[35]:


X_test=test_df.drop(columns=[ 'TARGET'])
y_test=test_df[['TARGET']]


# X_train=X_train.drop([60477],axis=0)

# y_train=y_train.drop([60477],axis=0)

# In[36]:


X_train2=X_train.fillna(0)


# In[37]:


X_test2=X_test.fillna(0)


# In[38]:


X_train_re=X_train[['PAYMENT_RATE',  'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'ANNUITY_INCOME_PERC', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'INCOME_CREDIT_PERC', 'AMT_CREDIT', 'OWN_CAR_AGE', 'INCOME_PER_PERSON', 'POS_MONTHS_BALANCE_MEAN', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'HOUR_APPR_PROCESS_START', 'AMT_INCOME_TOTAL', 'POS_MONTHS_BALANCE_MAX', 'TOTALAREA_MODE', 'PREV_DAYS_DECISION_MAX', 'CODE_GENDER', 'PREV_APP_CREDIT_PERC_MIN', 'PREV_DAYS_DECISION_MEAN', 'YEARS_BEGINEXPLUATATION_MODE', 'LIVINGAREA_MODE', 'PREV_AMT_ANNUITY_MEAN', 'PREV_APP_CREDIT_PERC_MEAN', 'AMT_REQ_CREDIT_BUREAU_QRT', 'PREV_CNT_PAYMENT_MEAN', 'PREV_DAYS_DECISION_MIN', 'PREV_AMT_ANNUITY_MIN', 'NAME_FAMILY_STATUS_Married', 'NAME_EDUCATION_TYPE_Highereducation', 'NAME_CONTRACT_TYPE_Cashloans', 'FLAG_DOCUMENT_3']]
X_train_re=X_train_re.fillna(0)


# In[39]:


X_test_re=X_test[['PAYMENT_RATE',  'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'ANNUITY_INCOME_PERC', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'INCOME_CREDIT_PERC', 'AMT_CREDIT', 'OWN_CAR_AGE', 'INCOME_PER_PERSON', 'POS_MONTHS_BALANCE_MEAN', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'HOUR_APPR_PROCESS_START', 'AMT_INCOME_TOTAL', 'POS_MONTHS_BALANCE_MAX', 'TOTALAREA_MODE', 'PREV_DAYS_DECISION_MAX', 'CODE_GENDER', 'PREV_APP_CREDIT_PERC_MIN', 'PREV_DAYS_DECISION_MEAN', 'YEARS_BEGINEXPLUATATION_MODE', 'LIVINGAREA_MODE', 'PREV_AMT_ANNUITY_MEAN', 'PREV_APP_CREDIT_PERC_MEAN', 'AMT_REQ_CREDIT_BUREAU_QRT', 'PREV_CNT_PAYMENT_MEAN', 'PREV_DAYS_DECISION_MIN', 'PREV_AMT_ANNUITY_MIN', 'NAME_FAMILY_STATUS_Married', 'NAME_EDUCATION_TYPE_Highereducation', 'NAME_CONTRACT_TYPE_Cashloans', 'FLAG_DOCUMENT_3']]
X_test_re=X_test_re.fillna(0)



class_weights = {
  0: 1,
  1: 10,
}



# In[46]:


from sklearn.ensemble import RandomForestClassifier

for i in [10]:
    class_weights[1]=i
    # we can add class_weight='balanced' to add panalize mistake
    rf_model = RandomForestClassifier(class_weight=class_weights)
    
    rf_model.fit(X_train_re, y_train['TARGET'])
    

# In[47]:


df_re=df[['PAYMENT_RATE',  'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_EMPLOYED_PERC', 'AMT_ANNUITY', 'ANNUITY_INCOME_PERC', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'INCOME_CREDIT_PERC', 'AMT_CREDIT', 'OWN_CAR_AGE', 'INCOME_PER_PERSON', 'POS_MONTHS_BALANCE_MEAN', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'HOUR_APPR_PROCESS_START', 'AMT_INCOME_TOTAL', 'POS_MONTHS_BALANCE_MAX', 'TOTALAREA_MODE', 'PREV_DAYS_DECISION_MAX', 'CODE_GENDER', 'PREV_APP_CREDIT_PERC_MIN', 'PREV_DAYS_DECISION_MEAN', 'YEARS_BEGINEXPLUATATION_MODE', 'LIVINGAREA_MODE', 'PREV_AMT_ANNUITY_MEAN', 'PREV_APP_CREDIT_PERC_MEAN', 'AMT_REQ_CREDIT_BUREAU_QRT', 'PREV_CNT_PAYMENT_MEAN', 'PREV_DAYS_DECISION_MIN', 'PREV_AMT_ANNUITY_MIN', 'NAME_FAMILY_STATUS_Married', 'NAME_EDUCATION_TYPE_Highereducation', 'NAME_CONTRACT_TYPE_Cashloans', 'FLAG_DOCUMENT_3']]
df_re=df_re.fillna(0)
df_re


# In[48]:


df_re.to_csv('clients_list.csv')


# In[49]:


instance = df_re.iloc[[100002]]
print ("Instance 0 prediction:", rf_model.predict(instance))


# In[50]:


from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
prediction, bias, contributions = ti.predict(rf_model, instance)


# In[51]:


contributions[0][:,0]


# In[52]:


localfi = pd.DataFrame()  
localfi['col']=X_train_re.columns
localfi['val']=contributions[0][:,0]
localfi['abs']=abs(localfi['val'])
localfi=localfi.sort_values(by=['abs'],ascending=False)
localfi


# In[58]:


def localf(id):
    instance = df_re.iloc[[id]]
    print ("Instance 0 prediction:", rf_model.predict(instance))
    prediction, bias, contributions = ti.predict(rf_model, instance)
    localfi = pd.DataFrame()  
    localfi['col']=X_train_re.columns
    localfi['val']=contributions[0][:,0]
    localfi['abs']=abs(localfi['val'])
    localfi=localfi.sort_values(by=['abs'],ascending=False)
    return localfi


# In[59]:


localf(100002)


# In[ ]:




