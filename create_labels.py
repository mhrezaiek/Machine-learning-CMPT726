#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 00:24:13 2020

@author: parmis
"""
import numpy as np
import pandas as pd
import csv

profile_path = 'datasets/cleaned/profile.csv'
offer_path = 'datasets/cleaned/portfolio.csv'
transactions_path = 'datasets/cleaned/transcript.csv'
portfolio_path = 'datasets/cleaned/portfolio.csv'
output_path = 'datasets/cleaned/labeled/'


with open(profile_path, 'r') as file:
    reader = csv.reader(file)
    people = []
    for row in reader:
        people.append(row[2])
    people.pop(0) # remove the header


people_dict = {}
i = 0
for person in people:
    if person not in people_dict:
        people_dict[person] = i
        i+= 1
        
with open(offer_path, 'r') as file:
    reader = csv.reader(file)
    offers = []
    for row in reader:
        offers.append(row[5])
    offers.pop(0) # remove the header
    
offers_dict = {}
i = 0
for offer in offers:
    if offer not in offers_dict:
        offers_dict[offer] = i
        i+= 1
        
        
df = pd.read_csv(transactions_path)
df['value'] = np.where(df.event != 'transaction', df['value'].replace(offers_dict) , df['value'])

df['value'] = df['value'].replace(offers_dict)
df['person'] = df['person'].replace(people_dict)
df.to_csv (output_path + 'labeled_transcript.csv', index = None)

df = pd.read_csv(profile_path)
df['id'] = df['id'].replace(people_dict)
df.to_csv (output_path + 'labeled_profile.csv', index = None)


df = pd.read_csv(portfolio_path)
df['id'] = df['id'].replace(offers_dict)
df.to_csv (output_path + 'labeled_portfolio.csv', index = None)






