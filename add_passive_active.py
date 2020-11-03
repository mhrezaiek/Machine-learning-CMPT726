#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:19:46 2020

@author: parmis
"""


import pickle
import numpy as np


matrices_path = '/Users/parmis/Desktop/Machine-learning-CMPT726/datasets/cleaned/matrices/'

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
    if complete_received_ratio[i] >= 0.66 :
        complete_received_ratio[i] = 1
    else:
        complete_received_ratio[i] = 0
complete_received_ratio = complete_received_ratio.reshape(-1,1)
        
       
concatenated_matrix = np.concatenate((completed_matrix, received_matrix) ,axis = 1)  
concatenated_matrix = np.concatenate((concatenated_matrix, viewed_matrix) ,axis = 1)  
concatenated_matrix = np.append(concatenated_matrix, complete_received_ratio, axis = 1)


with open(matrices_path + "concatenated_matrix_with_passive.pickle", 'wb') as handle:
    pickle.dump(concatenated_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)  






