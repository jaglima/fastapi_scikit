from fastapi import FastAPI, File, UploadFile
import json
import numpy as np
import pandas as pd
import joblib
import json
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel

def payload_preprocessing(df):
  
  # This "best" feature set was selected through a feature selection process that was ommited here
    best_feats = ['V3',
                 'V5',
                 'V7',
                 'V9',
                 'UF_ES',
                 'UF_MG',
                 'UF_RJ',
                 'UF_SP',
                 'city_Belo_Horizonte',
                 'zip_26089250',
                 'zip_29101685',
                 'zip_8253410',
                 'zip_8420400',
                 'month_01',
                 'month_02',
                 'month_03',
                 'month_04',
                 'month_05',
                 'month_06',
                 'month_07',
                 'month_08',
                 'month_09',
                 'month_10',
                 'month_11',
                 'month_12']
    
    return df[best_feats]


app = FastAPI(
    title="Trustly Chalenge Prediction API",
    description="""An API that utilises a Logistic Regression model to predict a class based on various features.""",
    version="0.0.1",
    debug=True,
)

@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc" """
    return note


@app.post("/uploadfiles/")
def predict(upload_file: UploadFile = File(...)):
    json_data = json.load(upload_file.file)
    
    df = pd.DataFrame(json_data)
    
    payload = payload_preprocessing(df)
    
    model = joblib.load("model/model.joblib") 
    
    output = pd.DataFrame(model.predict(payload), columns=['Class']).to_json(orient='records')
    
    json_compatible_item_data = jsonable_encoder(output)
    
    return JSONResponse(content=json_compatible_item_data)
    

