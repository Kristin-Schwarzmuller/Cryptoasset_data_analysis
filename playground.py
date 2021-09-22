import csv
import requests
import pandas as pd 
import io
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
#import datapackage
#import sklearn.linear_model as lm

path = 'cy_data'
pathToSave = 'diagrams/{}.png'
dir_list = sorted(os.listdir(path))
data  = list() # a list of dataframes containing all data from all cryp

digital_currency_list = 'digital_currency_list.csv'
currencies = pd.read_csv(digital_currency_list) 

#____________________________________________________________________________________
data, currencies_list, errors, notes, shit = list(), list(), list(), list(), list()
for d in dir_list: 
    #df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    df = pd.read_csv('{}/{}'.format(path, d))
    #print(d)
    if df.size == 0:
        print('Empty: {}'.format(d))
    elif '20' in df.iat[0, 1]:
        data.append(df)
        currencies_list.append(d.split(".")[0])
        #print('Yeah data')
    elif 'Error' in df.iat[0, 1]:
        errors.append(d.split(".")[0])
    elif 'Note' in df.iat[0, 1]:
        notes.append(d.split(".")[0])
    else:
        print('Crazy shit is happening with {}'.format(d))
        shit.append(df)
        
data[0].head()
print('--------Number of data: {}'.format(len(data)))   
#data[0].head()
print('--------Number of errors: {}'.format(len(errors)))  
#print(errors)
print('--------Number of notes: {}'.format(len(notes)))  
#print(notes)
print('--------Number of shit: {}'.format(len(shit)))  
#print(shit)

rel_data = data.copy() 
for d in rel_data: 
    start = d.iat[0, 2]
    #df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    d['open (USD)'] /= start
    d['high (USD)'] /= start
    d['low (USD)'] /= start
    d['close (USD)'] /= start
    d['spread'] = d['high (USD)'] - d['low (USD)']
rel_data[0]

#____________________________________________________________________________________
sp500 = pd.read_csv('HistoricalData_SP500_2.csv')
print(sp500)
for idx, d in enumerate(sp500['Date']):
        tmp = d.split('-')
        sp500['Date'][idx] = tmp[2] + '-' + tmp[0] + '-' + tmp[1]
print(sp500)

#____________________________________________________________________________________
def time_corr_SP500(a, b, timestamp_a = 'timestamp', timestamp_b='Date', open_a = 'open (USD)', open_b='Open'):
    
    # get the time frame where data for both dfs 
    mi = max(a[timestamp_a].min(), b[timestamp_b].min())
    ma = min(a[timestamp_a].max(), b[timestamp_b].max())

    if(mi == ma):
        return float('nan')

    # get the data from the time frame where there is data of both dataframes
    a_start = a.index[a[timestamp_a] == ma].tolist()[0]
    a_end = a.index[a[timestamp_a] == mi].tolist()[0]
    b_start = b.index[b[timestamp_b] == ma].tolist()[0]
    b_end = b.index[b[timestamp_b] == mi].tolist()[0]

    aa, bb = list(), list() 

    a_idx = a_start

    for b_idx in range(b_start, b_end):
        a_w = a[timestamp_a][a_idx]
        b_w = b[timestamp_b][b_idx]
        while((a_w != b_w) or (a_idx == a_end) or (b_idx == b_end)):
            a_idx += 1
            a_w = a[timestamp_a][a_idx]
        aa.append(a[open_a][a_idx])
        bb.append(b[open_b][b_idx])
        a_idx += 1
        b_idx += 1
    
    #aa, bb = pd.DataFrame(aa), pd.DataFrame(bb)
    return np.corrcoef(aa, bb)[0,1]

print("Wooo")
aa = time_corr_SP500(rel_data[0], sp500)
print(aa)