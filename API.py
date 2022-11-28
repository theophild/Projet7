from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import csv
import codecs
import API
import main

app = FastAPI()

@app.post('/predict/{id}')
def operate(id : int):
    result = main.pred(id)
    return result   

@app.post('/score/{id}')
def operate(id : int):
    result = main.score(id)
    return result   
        
@app.post('/localfi/{id}')
def operate(id : int):
    result = main.localf(id)
    return result

@app.post('/gfi/{id}')
def operate(id : int):
    gfi=main.globalfi(id)
    return gfi 