#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 00:14:30 2022

@author: arunima
"""

import numpy as np
import pandas as pd
import os
os.getcwd()
getdat = pd.read_csv(r"Downloads/mdata2_correct.csv")

getdat.head()

#select relevant columns
reldat = getdat.iloc[:,[0,5,
                        8,9,11,12,13,
                        14,15]]


#check unique values
reldat.nunique()

#Dummy coding categoricals
reldat = pd.concat([reldat.drop('Prediction', axis=1), pd.get_dummies(reldat['Prediction'], drop_first=True)], axis=1)
reldat.rename(columns = {" YES":"Prediction"}, 
              inplace = True)

reldat = pd.concat([reldat.drop('Resolution', axis=1), pd.get_dummies(reldat['Resolution'], drop_first=True)], axis=1)
reldat.rename(columns = {1:"Resolution"}, 
              inplace = True)

#Create new column with trade numbers 
reldat["repeats"] = np.tile(np.arange(1,51), len(reldat))[:len(reldat)]
           

#pivot data long tp wide
reldat_wide = pd.pivot(reldat, index = ["Market.ID"],
                       columns="repeats")

reldat_wide.columns = reldat_wide.columns.map(lambda x: ''.join([*map(str, x)]))
drop_cols = [*range(351,400)]

#Exclude repeats of outcome column
reldat_wide.drop(reldat_wide.columns[drop_cols],
                 axis = 1,
                 inplace = True)

#split data into training and test sets. apply normalization
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

train, val = train_test_split(reldat_wide, test_size = 0.33,
                              random_state = 42)


getrange = [*range(0,300)]
train_cont = train.iloc[:, getrange] 
scaler = StandardScaler().fit(train_cont)
std_train_cont = scaler.transform(train_cont)
std_train_cont = pd.DataFrame(std_train_cont) 
std_train_cont.reset_index(drop=True, inplace = True)

train_cat = train.iloc[:, 300:] 
train_cat.reset_index(drop=True, inplace = True)

tot_train = pd.concat([std_train_cont, train_cat],
                      axis = 1, ignore_index = True)




val_cont = val.iloc[:, getrange] 
std_val_cont = scaler.transform(val_cont)
std_val_cont = pd.DataFrame(std_val_cont) 
std_val_cont.reset_index(drop=True, inplace = True)

val_cat = val.iloc[:, 300:] 
val_cat.reset_index(drop=True, inplace = True)

tot_val = pd.concat([std_val_cont, val_cat],
                      axis = 1, ignore_index = True)

"""
WITH NON-STANDARDIZED DATA
"""
### Apply models to non-normalized data (without any hyperparameter tuning)
#1.Logistic regression
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(random_state=42, max_iter = 50000,
                          solver = 'sag')

clf1.fit(train.iloc[:,0:350], train.iloc[:,350])

pred1 = clf1.predict(val.iloc[:,0:350])
clf1.predict_proba(val.iloc[:,0:350])

from sklearn.metrics import confusion_matrix
confusion_matrix(val.iloc[:,350], pred1)

#2. Random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(class_weight = 'balanced', random_state=0)
clf2.fit(train.iloc[:,0:350], train.iloc[:,350])
pred2 = clf2.predict(val.iloc[:,0:350])
confusion_matrix(val.iloc[:,350], pred2)

from sklearn.metrics import classification_report
print(classification_report(val.iloc[:,350], pred2))


#3. Gradient boosted machines
from sklearn.ensemble import GradientBoostingClassifier
clf3 = GradientBoostingClassifier()
clf3.fit(train.iloc[:,0:350], train.iloc[:,350])
pred3 = clf3.predict(val.iloc[:,0:350])
confusion_matrix(val.iloc[:,350], pred3)
print(classification_report(val.iloc[:,350], pred3))
clf3.predict_proba(val.iloc[:,0:350])

#4.Support vector machines
from sklearn.svm import SVC
clf4 = SVC(random_state = 42)
clf4.fit(train.iloc[:,0:350], train.iloc[:,350])
pred4 = clf4.predict(val.iloc[:,0:350])
confusion_matrix(val.iloc[:,350], pred4)
print(classification_report(val.iloc[:,350], pred4))
clf4.predict_proba(val.iloc[:,0:350])

#5. Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf5 = GaussianNB()
clf5.fit(train.iloc[:,0:350], train.iloc[:,350])
pred5 = clf5.predict(val.iloc[:,0:350])
confusion_matrix(val.iloc[:,350], pred5)
print(classification_report(val.iloc[:,350], pred5))
clf5.predict_proba(val.iloc[:,0:350])

#6.K nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
clf6 = KNeighborsClassifier()
clf6.fit(train.iloc[:,0:350], train.iloc[:,350])
pred6 = clf6.predict(val.iloc[:,0:350])
confusion_matrix(val.iloc[:,350], pred6)
print(classification_report(val.iloc[:,350], pred6))
clf6.predict_proba(val.iloc[:,0:350])

#7. Xgboost
from xgboost import XGBClassifier
clf7 = XGBClassifier(allow_nan = True)
clf7.fit(train.iloc[:,0:350], train.iloc[:,350])
pred7 = clf7.predict(val.iloc[:,0:350])
confusion_matrix(val.iloc[:,350], pred7)
print(classification_report(val.iloc[:,350], pred7))
clf7.predict_proba(val.iloc[:,0:350])


"""
WITH STANDARDIZED DATA
"""
### Apply models to normalized data (without any hyperparameter tuning)
#1.Logistic regression
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(random_state=42, max_iter = 50000,
                          solver = 'sag')
clf1.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])

pred1 = clf1.predict(tot_val.iloc[:,0:350])
clf1.predict_proba(tot_val.iloc[:,0:350])

from sklearn.metrics import confusion_matrix
confusion_matrix(tot_val.iloc[:,350], pred1)
from sklearn.metrics import classification_report
print(classification_report(tot_val.iloc[:,350], pred1))

#2. Random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(class_weight = 'balanced', random_state=0)
clf2.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])
pred2 = clf2.predict(tot_val.iloc[:,0:350])
confusion_matrix(tot_val.iloc[:,350], pred2)

from sklearn.metrics import classification_report
print(classification_report(tot_val.iloc[:,350], pred2))


#3. Gradient boosted machine
from sklearn.ensemble import GradientBoostingClassifier
clf3 = GradientBoostingClassifier()
clf3.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])
pred3 = clf3.predict(tot_val.iloc[:,0:350])
confusion_matrix(tot_val.iloc[:,350], pred3)
print(classification_report(tot_val.iloc[:,350], pred3))
clf3.predict_proba(tot_val.iloc[:,0:350])

#4. Support vector machine
from sklearn.svm import SVC
clf4 = SVC()
clf4.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])
pred4 = clf4.predict(tot_val.iloc[:,0:350])
confusion_matrix(tot_val.iloc[:,350], pred4)
print(classification_report(tot_val.iloc[:,350], pred4))
clf4.predict_proba(tot_val.iloc[:,0:350])

#5. Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf5 = GaussianNB()
clf5.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])
pred5 = clf5.predict(tot_val.iloc[:,0:350])
confusion_matrix(tot_val.iloc[:,350], pred5)
print(classification_report(tot_val.iloc[:,350], pred5))
clf5.predict_proba(tot_val.iloc[:,0:350])

#6. K Nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
clf6 = KNeighborsClassifier()
clf6.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])
pred6 = clf6.predict(tot_val.iloc[:,0:350])
confusion_matrix(tot_val.iloc[:,350], pred6)
print(classification_report(tot_val.iloc[:,350], pred6))
clf6.predict_proba(tot_val.iloc[:,0:350])

#7. XGboost
from xgboost import XGBClassifier
clf7 = XGBClassifier(allow_nan = True)
clf7.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])
pred7 = clf7.predict(tot_val.iloc[:,0:350])
confusion_matrix(tot_val.iloc[:,350], pred7)
print(classification_report(tot_val.iloc[:,350], pred7))
clf7.predict_proba(tot_val.iloc[:,0:350])

#8. Stacking (with best models from above)
from sklearn.ensemble import StackingClassifier

estimators = [('rf', RandomForestClassifier(class_weight = 'balanced', random_state=0)),
# ('svc', SVC(random_state=42)),
('gbc',GradientBoostingClassifier()), 
('log', LogisticRegression(random_state=42, max_iter = 50000,
                          solver = 'sag')) 
# ,('gb', GaussianNB())
]

clf8 = StackingClassifier(
estimators=estimators, final_estimator=LogisticRegression(random_state=42, max_iter = 50000,
                          solver = 'sag')
)

clf8.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])
pred8 = clf8.predict(tot_val.iloc[:,0:350])
confusion_matrix(tot_val.iloc[:,350], pred8)
print(classification_report(tot_val.iloc[:,350], pred8))
clf8.predict_proba(tot_val.iloc[:,0:350])


from sklearn.ensemble import VotingClassifier
estimators = [('rf', RandomForestClassifier(class_weight = 'balanced', random_state=0)),
 # ('svc', SVC(random_state=42, probability=True)), 
('gbc',GradientBoostingClassifier()),
('log', LogisticRegression(random_state=42, max_iter = 50000,
                          solver = 'sag')) 
# ,('gb', GaussianNB())
]

#9. Soft voting (with best models from above)
clf9 = VotingClassifier(
estimators=estimators, voting='soft'
)

clf9.fit(tot_train.iloc[:,0:350], tot_train.iloc[:,350])
pred9 = clf9.predict(tot_val.iloc[:,0:350])
confusion_matrix(tot_val.iloc[:,350], pred9)
print(classification_report(tot_val.iloc[:,350], pred9))
clf9.predict_proba(tot_val.iloc[:,0:350])

#Save best model and scaler
import joblib
joblib.dump(clf1, r"Downloads/trialClassifier.sav")
joblib.dump(scaler, r"Downloads/trialScaler.sav")