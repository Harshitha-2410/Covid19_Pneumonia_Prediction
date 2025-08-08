import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df=pd.read_csv("feedback_dataset_100_cols.csv")

x=df["feedback_text"]
y=df["sentiment"]

vec=TfidfVectorizer(stop_words="english",max_features=1000)
xx=vec.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(xx,y,test_size=0.2)

model=RandomForestClassifier()
model.fit(x_train,y_train)
joblib.dump(model,"comment.pkl")
joblib.dump(vec,"vector.pkl")
