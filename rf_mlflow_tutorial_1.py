#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:40:19 2019

@author: ankurmanikandan
"""

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

import mlflow

np.random.seed(42)

file_name = os.path.basename(__file__)


data = pd.read_csv("../Datasets/creditcard.csv")
X = data.values[:, 1:-1]
y = data.values[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y)

artifact_loc = "rf_mlflow_artifacts"

#exp_id = mlflow.create_experiment(name="Random Forest Model Exploration",
#                                  artifact_location=artifact_loc)
exp_id = '1'

hyperparam_vals = [[10, "gini"], [20, "gini"], [10, "entropy"], 
                   [20, "entropy"]]

count = 0

for i in hyperparam_vals:
    count += 1
    n_estimators = i[0]
    criterion = i[1]
    
    with mlflow.start_run(experiment_id=exp_id):
        
        mlflow.log_param("SEED", 42)
    
        clf = RandomForestClassifier(n_estimators=i[0], max_depth=2,
                                     criterion=i[1], n_jobs=-1,
                                     class_weight="balanced")
        mlflow.log_param("n_estimators", i[0])
        mlflow.log_param("max_depth", 2)
        mlflow.log_param("criterion", i[1])
        mlflow.log_param("class_weight", "balanced")
        
        clf.fit(x_train, y_train)
        
        joblib.dump(clf, artifact_loc+"/rf_"+str(count)+".joblib")
        mlflow.log_artifact(local_path=artifact_loc+"/rf_"+str(count)+".joblib",
                            artifact_path=artifact_loc)
        
        y_pred = clf.predict(x_test)
        
        test_accuracy = clf.score(x_test, y_test)
        mlflow.log_metric("Test Accuracy", test_accuracy)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        mlflow.log_metric("True Negatives", tn)
        mlflow.log_metric("False Positives", fp)
        mlflow.log_metric("False Negatives", fn)
        mlflow.log_metric("True Positives", tp)
    
        
    
    