"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Online Shoppers Purchasing Intention.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model():
 c=["VisitorType","Weekend","OperatingSystems","Browser","Region","TrafficType"]
 n=["Administrative","Informational","ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","PageValues","SpecialDay"]
 p=ColumnTransformer([("num","passthrough",n),("cat",OneHotEncoder(handle_unknown="ignore"),c)])
 m=RandomForestClassifier(n_estimators=800,criterion="entropy",max_features=0.5,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=42)
 return Pipeline([("preprocessor",p),("model",m)])
