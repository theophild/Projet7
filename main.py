import numpy as np
import pandas as pd
import pickle
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import boto3
 

def load_data(name, form):
    s3 = boto3.resource(
    service_name='s3',
    region_name='eu-west-3',
    aws_access_key_id='AKIAR66Z64XBX3KEK7DV',
    aws_secret_access_key='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',
    )
    if form == 'csv' :
        obj = s3.Bucket('projet7dubois').Object(name).get()
        data = pd.read_csv(obj['Body'], index_col=0)
    if form == 'pkl' :
        data = pickle.loads(s3.Bucket("projet7dubois").Object(name).get()['Body'].read())
    return data

rf_model = pickle.load(open('model.pkl', 'rb'))
df=pd.read_csv('clients_list.csv', index_col=0)

def pred(id):
    instance = df.loc[[id]]
    return rf_model.predict(instance)[0]

def score(id):
    instance = df.loc[[id]]
    return 1-rf_model.predict_proba(instance)[0][0]

def localf(id):
    instance = df.loc[[id]]
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