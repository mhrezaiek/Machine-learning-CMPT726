#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:15:13 2020

@author: parmis
"""


from scipy.sparse import lil_matrix
import pickle



profile_path = 'datasets/cleaned/labeled/labeled_profile.csv'
transcript_path = 'datasets/cleaned/labeled/labeled_transcript.csv'
portfolio_path = 'datasets/cleaned/labeled/labeled_portfolio.csv'
ouput_path = 'datasets/cleaned/matrices/'

with open(profile_path) as f:
    content = f.readlines()
content = [x.strip() for x in content]
content.pop(0)
people_num = len(content)


with open(portfolio_path) as f:
    content = f.readlines()
content = [x.strip() for x in content]
content.pop(0)
offer_num = len(content)


with open(transcript_path) as f:
    content = f.readlines()
transcript = [x.strip() for x in content]
transcript = [x.split(',') for x in content]
transcript.pop(0)


received_matrix = lil_matrix((people_num, offer_num))
for row in transcript:   
    if row[1] == 'offer received':
        received_matrix[int(row[0]),int(row[2])] = 1

with open(ouput_path + 'received_matrix.pickle', 'wb') as handle:
    pickle.dump(received_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)




viewed_matrix = lil_matrix((people_num, offer_num))
for row in transcript:   
    if row[1] == 'offer viewed':
        viewed_matrix[int(row[0]),int(row[2])] = 1

with open(ouput_path + 'viewed_matrix.pickle', 'wb') as handle:
    pickle.dump(viewed_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)        
 



       
completed_matrix = lil_matrix((people_num, offer_num))
for row in transcript:   
    if row[1] == 'offer completed':
        completed_matrix[int(row[0]),int(row[2])] = 1
        
with open(ouput_path + 'completed_matrix.pickle', 'wb') as handle:
    pickle.dump(completed_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)      




