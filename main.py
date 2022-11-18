import numpy as np
import pandas as pd
import pickle
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
 
rf_model = pickle.load(open('model.sav', 'rb'))
df=pd.read_csv('clients_list.csv')
df.index=df['SK_ID_CURR']
df=df.drop(columns=['SK_ID_CURR'])


def pred(id):
    instance = df.iloc[[id]]
    return rf_model.predict(instance)[0]

def score(id):
    instance = df.iloc[[id]]
    return rf_model.predict_proba(instance)[0][0]

def localf(id):
    instance = df.iloc[[id]]
    prediction, bias, contributions = ti.predict(rf_model, instance)
    localfi = pd.DataFrame()  
    localfi['col']=df.columns
    localfi['val']=contributions[0][:,0]
    localfi['abs']=abs(localfi['val'])
    localfi=localfi.sort_values(by=['abs'],ascending=False)
    localfi=localfi.reset_index()[['col','val']]
    return localfi

def globalfi(id):
    gfi=pd.DataFrame()
    gfi['val']=rf_model.feature_importances_
    gfi['col']=df.columns
    return gfi