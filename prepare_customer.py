#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:08:00 2020

@author: parmis
"""
from scipy.sparse import lil_matrix
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, f1_score, classification_report


profile_path = 'datasets/cleaned/labeled/labeled_profile.csv'
matrices_path = 'datasets/cleaned/matrices/'
        
df = pd.read_csv(profile_path, parse_dates=True)
df = df[df['income'].notnull()]
now = pd.datetime.now()
df['became_member_on'] = pd.to_datetime(df['became_member_on'])
df = df[df['became_member_on'] <= pd.to_datetime("now")]
df = df[df['age'].astype(int) <= 95].reset_index()


incomes = df['income'].to_numpy(dtype = 'float64')
all_incomes = np.zeros((len(incomes),1))
for i in range(len(incomes)):
    all_incomes[i] = float(incomes[i])
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(all_incomes)
income_scaled = scaler.transform(all_incomes)
df['income_new'] = income_scaled


ages = df['age'].to_numpy(dtype = 'int')
all_ages = np.zeros((len(ages),1))
for i in range(len(ages)):
    all_ages[i] = float(ages[i])
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(all_ages)
ages_scaled = scaler.transform(all_ages)


gens = []
gen = df['gender'].to_numpy()
for i in range(len(gen)):
    if gen[i] == 'M':
        gens.append([1,0,0])
    elif gen[i] == 'F':
        gens.append([0,1,0])
    else:
        gens.append([0,0,1])
        
gens = np.array(gens)    
        

member_days = (pd.to_datetime("now") - df['became_member_on']) / np.timedelta64(1, 'D')


member_days = member_days.to_numpy(dtype = 'float64')
member_days = member_days.reshape(len(member_days),1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(member_days)
member_days = scaler.transform(member_days)
df['member_days'] = member_days

features = np.concatenate((gens, ages_scaled), axis =1)
features = np.concatenate((features, income_scaled), axis =1)
features = np.concatenate((features, member_days), axis =1)



with open(matrices_path + "completed_matrix.pickle", "rb") as f:
    completed_matrix = pickle.load(f).toarray()

ids = df['id'].to_numpy(dtype = 'int')
completed_offers_selcted = completed_matrix[ids, :]


customers = completed_offers_selcted.nonzero()[0]
offers = completed_offers_selcted.nonzero()[1]


all_features = []
i = 0
for customer in customers: 
    v = list(features[customer])
    v.append(offers[i])
    all_features.append(v)
    i += 1
all_features = np.array(all_features)


def populate_labels(number):
    listt = np.zeros((len(all_features), 1))
    for i in range(len(all_features)):
        if all_features[i][-1] == number:
            listt[i] = 1
    return np.array(listt)
        

offers = []
for i in range(10):
    offers.append(populate_labels(i))




 





        
 
def get_metrices(labels_test, labels_pred):
    accuracy = accuracy_score(labels_test, labels_pred)
    micro_recall = recall_score(labels_test, labels_pred, average='micro')
    macro_recall = recall_score(labels_test, labels_pred, average='macro')
    micro_precision = precision_score(labels_test, labels_pred, average='micro')
    macro_precision = precision_score(labels_test, labels_pred, average='macro')
    micro_f1 = f1_score(labels_test, labels_pred, average='micro')
    macro_f1 = f1_score(labels_test, labels_pred, average='macro')
    conf_matrix = confusion_matrix(labels_test, labels_pred)
    print("accuracy:", accuracy)
    print("macro_f1", macro_f1)
    print("micro_f1", micro_f1)
    print(classification_report(labels_test, labels_pred))
    


from sklearn.tree import DecisionTreeClassifier
X = all_features[:, :-1]  
model = []
preds = []
for i in range(10):
    print("model for offer: ", i)
    temp = DecisionTreeClassifier(max_depth=6)
    model.append(temp)
    y = offers[i]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model[i].fit(X_train, y_train)  
    # print("******************************")
    # print("DecisionTreeClassifier_train: ")
    # get_metrices(y_train, model.predict(X_train))
    print("******************************")
    print("DecisionTreeClassifier_test: ")

    get_metrices(y_test, model[i].predict(X_test))
    ll = X_test[190,:].reshape(1,-1)
    preds.append(model[i].predict(ll))
    
       
    

    



        

        