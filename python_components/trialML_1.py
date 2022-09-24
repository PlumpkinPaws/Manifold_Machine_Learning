#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 04:21:30 2022

@author: arunima
"""
import numpy as np
import pandas as pd
import os
os.getcwd()
getdat = pd.read_csv(r"C:/temp/mdata.csv")

def ML_50trades():

    
    
    reldat = getdat.iloc[:,[0,
                            7,8,10,11,12,
                            13,14]]


    reldat = pd.concat([reldat.drop(' Prediction', axis=1), pd.get_dummies(reldat[' Prediction'], drop_first=True)], axis=1)
    reldat.rename(columns = {" YES":"Prediction"}, 
                  inplace = True)



    reldat["repeats"] = np.tile(np.arange(1,51), len(reldat))[:len(reldat)]

    #reldat = reldat.iloc[0:50,:]

    reldat_wide = pd.pivot(reldat, index = "Market ID", columns="repeats")
    reldat_wide.columns = reldat_wide.columns.map(lambda x: ''.join([*map(str, x)]))

    getrange = [*range(0,300)]

    import joblib
    loadmodel = joblib.load(r"C:/Users/Arunima/Downloads/trialClassifier.sav")
    scaler = joblib.load(r"C:/Users/Arunima/Downloads/trialScaler.sav")

    val_cont = reldat_wide.iloc[:, getrange] 
    std_val_cont = scaler.transform(val_cont)
    std_val_cont = pd.DataFrame(std_val_cont) 
    std_val_cont.reset_index(drop=True, inplace = True)

    val_cat = reldat_wide.iloc[:, 300:] 
    val_cat.reset_index(drop=True, inplace = True)

    tot_val = pd.concat([std_val_cont, val_cat],
                          axis = 1, ignore_index = True)



    pd.set_option("display.precision", 12)

    predictions = loadmodel.predict(tot_val)
    # predictions = pd.DataFrame(predictions)

    # pred_probabilities = pd.DataFrame(loadmodel.predict_proba(tot_val), columns = loadmodel.classes_)
    proba_class1 = loadmodel.predict_proba(tot_val)[:,1]
    # output = pd.concat([predictions.rename(columns = {0:"Predicted_outcome"}), 
    #                     pred_probabilities.rename(columns = {0: "Probability_class0",
    #                                                   1: "Probability_class1"})],
    #                    axis = 1)
    
    print(proba_class1)

ML_50trades()
# output.to_csv(r"Downloads/sample_output.csv")
