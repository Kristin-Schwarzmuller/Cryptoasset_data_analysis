# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#pip install datapackage


# %%
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


# %%
path = 'cy_data'
pathToSave = 'diagrams/{}.png'
dir_list = sorted(os.listdir(path))
data  = list() # a list of dataframes containing all data from all cryp

digital_currency_list = 'digital_currency_list.csv'
currencies = pd.read_csv(digital_currency_list) 


# %%
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


# %%
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


# %%
ouput_df = pd.DataFrame(data=currencies_list)
ouput_df.columns =['currency']


# %%
def time_corr(a, b, timestamp_b='timestamp', open_b='open (USD)'):

    # get the time frame where data for both dfs 
    mi = max(a['timestamp'].min(), b[timestamp_b].min())
    ma = min(a['timestamp'].max(), b[timestamp_b].max())

    # get the data from the time frame where there is data of both dataframes
    a_start = a.index[a['timestamp'] == ma].tolist()
    a_end = a.index[a['timestamp'] == mi].tolist()
    b_start = b.index[b[timestamp_b] == ma].tolist()
    b_end = b.index[b[timestamp_b] == mi].tolist()
    
    if(a_start and b_start and a_end and b_end):
        a_use = a.iloc[a_start[0]: a_end[0], :]
        b_use = b.iloc[b_start[0]: b_end[0], :]

        return a_use['open (USD)'].corr(b_use[open_b])
    else:
        return float('nan')


# %%
def time_corr_SP500(a, b, timestamp_a = 'timestamp', timestamp_b='Date', open_a = 'open (USD)', open_b='Open'):
    
    mi_a = a[timestamp_a].min()
    mi_b = b[timestamp_b].min()
    ma_a = a[timestamp_a].max()
    ma_b = b[timestamp_b].max()

    # get the time frame where data for both dfs 
    mi = max(a[timestamp_a].min(), b[timestamp_b].min())
    ma = min(a[timestamp_a].max(), b[timestamp_b].max())

    d_b_mi, d_b_ma = 0, 0
    while(not(mi in b[timestamp_b].tolist())):
        d_b_mi += 1
    while(not(ma in b[timestamp_b].tolist())):
        d_b_ma += 1

    if(mi == ma):
        return float('nan')

    # get the data from the time frame where there is data of both dataframes
    a_start = a.index[a[timestamp_a] == ma].tolist()[0]
    a_end = a.index[a[timestamp_a] == mi].tolist()[0]
    b_start = b.index[b[timestamp_b] == ma + d_b_ma].tolist()[0]
    b_end = b.index[b[timestamp_b] == mi + d_b_mi].tolist()[0]

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


# %%
sp500 = pd.read_csv('HistoricalData_SP500_2.csv')
print(sp500)
for idx, d in enumerate(sp500['Date']):
        tmp = d.split('-')
        sp500['Date'][idx] = tmp[2] + '-' + tmp[0] + '-' + tmp[1]
print(sp500)


# %%
aa = time_corr_SP500(rel_data[1], sp500)


# %%
fig, ax = plt.subplots()
ax.plot('Open', data=sp500)
#ax.plot('Open', data=sp500.tail(1000))
plt.show()


# %%
l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11 = list(),  list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
corr_btc_cur, corr_btc_df = list(),list()
corr_sp500_cur, corr_sp500_df = list(),list()
x = 2
for cnt, d in enumerate(rel_data): 
    #l0.append(d['market cap (USD)'].mean())
    l1.append(d.count(axis=0)[1])
    l2.append(d['open (USD)'].std())
    l3.append(d['open (USD)'].mean())
    
    std = d['spread'].std()
    mu = d['spread'].mean()
    l4.append(std)
    l5.append(mu)
    ss = 0
    for e in d['spread']:
        if (e > mu + x*std) or (e < mu - x*std):
            ss += 1
    l6.append(ss)
    l7.append(d.iat[0, 3] - d.iat[-1, 3])
    l8.append((d.iat[0, 3] - d.iat[-1, 3]) / d.count(axis=0)[1])
    c = time_corr(d, rel_data[currencies_list.index('BTC')])
    l9.append(c)
    if c < -0.5:
        corr_btc_cur.append(ouput_df.iloc[cnt, 0])
        corr_btc_df.append(d)
    
    c = time_corr_SP500(d, sp500)
    l10.append(c)
    if c < -0.5:
        corr_sp500_cur.append(ouput_df.iloc[cnt, 0])
        corr_sp500_df.append(d)
    l11 = d['timestamp'][0]

for cnt, d in enumerate(data): 
    l0.append(d['market cap (USD)'].mean())
    
ouput_df['avg market cap'] = l0    
ouput_df['observations'] = l1
ouput_df['std'] = l2
ouput_df['mean'] = l3
ouput_df['spread std'] = l4
ouput_df['spread mean'] = l5
ouput_df['outlier (3std)'] = l6
ouput_df['profit'] = l7
ouput_df['avg profit per day'] = l8
ouput_df['corr to BTC'] = l9
ouput_df['corr to S&P500'] = l10
ouput_df['last timestamp'] = l11

ouput_df


# %%
print('min obersations: {}'.format(ouput_df['observations'].min()))
print('max obersations: {}'.format(ouput_df['observations'].max()))

ouput_sorted = ouput_df.sort_values(['avg market cap', 'currency'], ascending=False)
ouput_sorted.head(10)


# %%
plt.scatter(ouput_df['outlier (3std)'], ouput_df['profit'], color='k')
plt.xlabel('Outlier ')
plt.ylabel('Profit')


# %%
print(currencies.loc[currencies['currency code'] == currencies_list[ouput_df['profit'].idxmin()]])
ouput_df_wo_ol = ouput_df.drop(ouput_df['profit'].idxmin(), axis=0)

print(currencies.loc[currencies['currency code'] == currencies_list[ouput_df_wo_ol['profit'].idxmin()]])
ouput_df_wo_ol = ouput_df_wo_ol.drop(ouput_df_wo_ol['profit'].idxmin(), axis=0)

print(currencies.loc[currencies['currency code'] == currencies_list[ouput_df_wo_ol['profit'].idxmin()]])
ouput_df_wo_ol = ouput_df_wo_ol.drop(ouput_df_wo_ol['profit'].idxmin(), axis=0)

print(currencies.loc[currencies['currency code'] == currencies_list[ouput_df_wo_ol['profit'].idxmin()]])
ouput_df_wo_ol = ouput_df_wo_ol.drop(ouput_df_wo_ol['profit'].idxmin(), axis=0)

print(currencies.loc[currencies['currency code'] == currencies_list[ouput_df_wo_ol['profit'].idxmin()]])
ouput_df_wo_ol = ouput_df_wo_ol.drop(ouput_df_wo_ol['profit'].idxmin(), axis=0)

print('average profit: {}'.format(ouput_df_wo_ol['profit'].mean()))


# %%
#data = pd.read_csv('data.csv')  # load data set
x = ouput_df_wo_ol['outlier (3std)'].values.reshape(-1, 1)
y = ouput_df_wo_ol['profit'].values.reshape(-1, 1)

linear_regressor = lm.LinearRegression()  # create object for the class
linear_regressor.fit(x, y)  # perform linear regression
y_pred = linear_regressor.predict(x)  # make predictions

print('Coefficients: \n', linear_regressor.coef_)
plt.scatter(x, y, color='k')
plt.plot(x, y_pred, color='b')
plt.xlabel('number of spread outliers')
plt.ylabel('profit')
plt.savefig(pathToSave.format('outlier_profit'))
plt.show()


# %%
#data = pd.read_csv('data.csv')  # load data set
x = ouput_df_wo_ol['outlier (3std)'].values.reshape(-1, 1)
y = ouput_df_wo_ol['avg profit per day'].values.reshape(-1, 1)

linear_regressor = lm.LinearRegression()  # create object for the class
linear_regressor.fit(x, y)  # perform linear regression
y_pred = linear_regressor.predict(x)  # make predictions

print('Coefficients: \n', linear_regressor.coef_)
plt.scatter(x, y, color='k')
plt.plot(x, y_pred, color='b')
plt.xlabel('Number of spread outliers')
plt.ylabel('avg profit per day')
plt.savefig(pathToSave.format('outlier_profit_per_day'))
plt.show()


# %%
ouput_df_wo_olx = ouput_df_wo_ol.drop(ouput_df_wo_ol['avg market cap'].idxmax(), axis=0)
ouput_df_wo_olx = ouput_df_wo_olx.drop(ouput_df_wo_olx['avg market cap'].idxmax(), axis=0)
ouput_df_wo_olx = ouput_df_wo_olx.drop(ouput_df_wo_olx['avg market cap'].idxmax(), axis=0)
#ouput_df_wo_olx = ouput_df_wo_ol.drop(ouput_df_wo_olx['profit'].idxmax(), axis=0)
#ouput_df_wo_olx = ouput_df_wo_ol.drop(ouput_df_wo_olx['profit'].idxmax(), axis=0)
#ouput_df_wo_olx = ouput_df_wo_ol.drop(ouput_df_wo_olx['profit'].idxmax(), axis=0)
#plt.scatter(ouput_df_wo_olx['avg market cap'], ouput_df_wo_olx['std'], color='k')
#plt.scatter(ouput_df['avg market cap'], ouput_df['std'], color='k')
#plt.xlabel('Average market capitalisation')
#plt.ylabel('Profit')

#data = pd.read_csv('data.csv')  # load data set
x = ouput_df_wo_olx['avg market cap'].values.reshape(-1, 1)
y = ouput_df_wo_olx['std'].values.reshape(-1, 1)

linear_regressor = lm.LinearRegression()  # create object for the class
linear_regressor.fit(x, y)  # perform linear regression
y_pred = linear_regressor.predict(x)  # make predictions

print('Coefficients: \n', linear_regressor.coef_)
plt.scatter(x, y, color='k')
plt.plot(x, y_pred, color='b')
plt.xlabel('average market capitalization')
plt.ylabel('std')
plt.savefig(pathToSave.format('std_market_cap'))
plt.show()


# %%
plt.scatter(ouput_df['std'], ouput_df['profit'], color='k')
plt.xlabel('Standard deviation ')
plt.ylabel('Profit')


# %%
#data = pd.read_csv('data.csv')  # load data set
x = ouput_df_wo_ol['std'].values.reshape(-1, 1)
y = ouput_df_wo_ol['profit'].values.reshape(-1, 1)

linear_regressor = lm.LinearRegression()  # create object for the class
linear_regressor.fit(x, y)  # perform linear regression
y_pred = linear_regressor.predict(x)  # make predictions

print('Coefficients: \n', linear_regressor.coef_)
plt.scatter(x, y, color='k')
plt.plot(x, y_pred, color='b')
plt.xlabel('std')
plt.ylabel('profit')
plt.savefig(pathToSave.format('std_profit'))
plt.show()


# %%
#data = pd.read_csv('data.csv')  # load data set
x = ouput_df_wo_ol['std'].values.reshape(-1, 1)
y = ouput_df_wo_ol['avg profit per day'].values.reshape(-1, 1)

linear_regressor = lm.LinearRegression()  # create object for the class
linear_regressor.fit(x, y)  # perform linear regression
y_pred = linear_regressor.predict(x)  # make predictions

print('Coefficients: \n', linear_regressor.coef_)
plt.scatter(x, y, color='k')
plt.plot(x, y_pred, color='b')
plt.xlabel('std')
plt.ylabel('avg profit per day')
plt.savefig(pathToSave.format('std_profit_per_day'))
plt.show()


# %%
plt.xcorr(ouput_df['outlier (3std)'], ouput_df['profit'])
plt.xlabel('Outlier ')
plt.ylabel('Profit')


# %%
plt.hist(ouput_df['outlier (3std)'], bins=30, color='k')
plt.xlabel('outliers >/< 3*std')
plt.ylabel('observations')
plt.savefig(pathToSave.format('hist_outlier'))
plt.show()


# %%
plt.hist(ouput_df['corr to BTC'],bins=50, color='k')
plt.xlabel('Pearson correlation coefficient to BTC')
plt.ylabel('observations')
plt.savefig(pathToSave.format('hist_corr_BTC'))
plt.show()


# %%
print(corr_btc_cur)
ouput_df.loc[ouput_df['currency'] == corr_btc_cur[0]]


# %%
data_STRAT = pd.read_csv('cy_data/STRAT.csv')
data_STRAT.head(10)


# %%
plt.hist(ouput_df['corr to S&P500'],bins=50, color='k') 
plt.xlabel('Pearson correlation coefficient to S&P500')
plt.ylabel('observations')
#plt.savefig(pathToSave.format('hist_corr_SP500'))
plt.show()


# %%
print(corr_sp500_cur)
ouput_df.loc[ouput_df['currency'].isin(corr_sp500_cur)]


# %%
used_currencies = currencies.loc[currencies['currency code'].isin(currencies_list)]
used_currencies.to_csv('{}.csv'.format('used_currencies'), encoding='utf-8')
ouput_df.to_csv('{}.csv'.format('output'), encoding='utf-8')


# %%
# https://www.alphavantage.co/documentation/#currency-daily
CSV_URL = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=GRT&datatype=csv&market=USD&apikey=E7BF5XXE8WRMMDOD'
download = requests.Session().get(CSV_URL).content
grt = pd.read_csv(io.StringIO(download.decode('utf-8')))
grt


# %%
rel_sp500 = sp500.copy() 
print(sp500)
for d in rel_sp500: 
    start = d.iat[0, 3]
    #df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    d['Open'] /= start
rel_data[0]


# %%
btc_strat = pd.read_csv('BTC_STRAT.csv') 
btc_strat.head()


# %%
ss = btc_strat['STRAT'][0]
btc_strat['STRAT'] /= ss

bs = btc_strat['BTC'][0]
btc_strat['BTC'] /= bs
btc_strat.head()


# %%
#btc_strat = btc_strat.iloc[::-1]
ax =  btc_strat.plot(color=['k','b'])
ax.set_xlabel("time")
#ax.set_xticklabels(['a', 'b'])
ax.set_xticklabels([btc_strat['timestamp'][0], btc_strat['timestamp'][2]])
ax.set_ylabel("normalized opening price")
ax.figure.savefig('diagrams/btc_stat.png')


# %%



