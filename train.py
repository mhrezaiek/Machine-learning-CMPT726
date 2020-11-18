#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:10:55 2020

@author: parmis
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


input_path = 'datasets/cleaned/matrices/concatenated_matrix_with_passive.pickle'

with open(input_path, "rb") as input_file:
    matrix = pickle.load(input_file)


y = matrix[:, -1] # for last column
X = matrix[:, :-1] # for all but last column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
print("DecisionTreeClassifier_train: ",model.score(X_train, y_train))
print("DecisionTreeClassifier_test: ", model.score(X_test, y_test))



from sklearn.svm import SVC
model = SVC(kernel='linear', C=2.0)
model.fit(X_train, y_train)
print("SVC_train: ",model.score(X_train, y_train))
print("SVC_test: ", model.score(X_test, y_test))


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("KNeighborsClassifier_train: ", model.score(X_train, y_train))
print("KNeighborsClassifier_test: ", model.score(X_test, y_test))



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=20, random_state=0)
model.fit(X_train, y_train)
print("RandomForestClassifier_train: ", model.score(X_train, y_train))
print("RandomForestClassifier_test: ", model.score(X_test, y_test))


# print(classification_report(y_test, model.predict(X_test)))
