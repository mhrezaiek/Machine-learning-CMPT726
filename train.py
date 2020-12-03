#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:10:55 2020

@author: parmis
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt


input_path = 'datasets/cleaned/matrices/concatenated_matrix_with_passive.pickle'

with open(input_path, "rb") as input_file:
    matrix = pickle.load(input_file)


y = matrix[:, -1] # for last column
X = matrix[:, :-1] # for all but last column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




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
    # , micro_recall, macro_recall, micro_precision, macro_precision , micro_f1, macro_f1)


    

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

print("DecisionTreeClassifier_train: ")
get_metrices(y_train, model.predict(X_train))
print("DecisionTreeClassifier_test: ")
get_metrices(y_test, model.predict(X_test))


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=20, random_state=0)
model.fit(X_train, y_train)
print("RandomForestClassifier_train: ")
get_metrices(y_train, model.predict(X_train))

print("RandomForestClassifier_test: ")
get_metrices(y_test, model.predict(X_test))




from sklearn.svm import SVC
model = SVC(kernel='linear', C=2.0)
model.fit(X_train, y_train)

print("SVC_train: ")
get_metrices(y_train, model.predict(X_train))
print("SVC_test: ")
get_metrices(y_test, model.predict(X_test))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("KNeighborsClassifier_train: ")
get_metrices(y_train, model.predict(X_train))
print("KNeighborsClassifier_test: ")
get_metrices(y_test, model.predict(X_test))



# print(classification_report(y_test, model.predict(X_test)))
