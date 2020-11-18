#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:19:46 2020

@author: parmis
"""


import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


all_customers = 17000
transactions_path = 'datasets/cleaned/labeled/labeled_transcript.csv'


df = pd.read_csv(transactions_path)
df = df[df['event']== 'transaction'].sort_values(by=['person'])
df = df.drop(['reward', 'time'], axis=1)
df = df.groupby(['person']).sum().reset_index()
all_spenders = df['person'].to_numpy()
money_spent = df.to_numpy(dtype = 'float64')


all_transactions = np.zeros((all_customers,1))
for trans in money_spent:
    all_transactions[int(trans[0])] = float(trans[1])
    

average_money_spent = np.average(money_spent[:, 1])





matrices_path = 'datasets/cleaned/matrices/'

with open(matrices_path + "completed_matrix.pickle", "rb") as f:
    completed_matrix = pickle.load(f).toarray()
    
    
with open(matrices_path + "received_matrix.pickle", "rb") as f:
    received_matrix = pickle.load(f).toarray()
    
    
with open(matrices_path + "viewed_matrix.pickle", "rb") as f:
    viewed_matrix = pickle.load(f).toarray()
    
all_completed = completed_matrix.sum(axis=1)
all_received = received_matrix.sum(axis=1)

complete_received_ratio = all_completed / all_received


for i in range(len(complete_received_ratio)):
    if complete_received_ratio[i] >= 0.66  and float(all_transactions[i]) >= average_money_spent:
        complete_received_ratio[i] = 1
    else:
        complete_received_ratio[i] = 0
complete_received_ratio = complete_received_ratio.reshape(-1,1)
        
  
    
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(all_transactions)
money_spent_scaled = scaler.transform(all_transactions)  

        
concatenated_matrix = np.concatenate((completed_matrix, received_matrix) ,axis = 1)  
concatenated_matrix = np.concatenate((concatenated_matrix, viewed_matrix) ,axis = 1)  
concatenated_matrix = np.append(concatenated_matrix, money_spent_scaled, axis = 1)
concatenated_matrix = np.append(concatenated_matrix, complete_received_ratio, axis = 1)


with open(matrices_path + "concatenated_matrix_with_passive.pickle", 'wb') as handle:
    pickle.dump(concatenated_matrix, handle, protocol= 3 )  






