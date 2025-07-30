from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os


model=joblib.load(os.path.join(os.path.dirname(__file__),"covid_diag.pkl"))

class inp(BaseModel):
    Age:int
    Gender:int
    Fever:int
    Cough:int
    Fatigue:int
    Breathlessness:int
    Comorbidity:int
    Stage:int
    Type:int
    Tumor_Size:float

app=FastAPI()

@app.get("/")
def route():
    return {"message":"Welcome"}

@app.post("/prd")
def prediction(data:inp):
    inp=pd.DataFrame([data.dict()])
    pred=model.predict(inp)
    return {"Prediction":pred[0]}
