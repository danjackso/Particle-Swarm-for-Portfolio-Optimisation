import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import numpy as np
import pyswarms as pso
from itertools import combinations

#List of Dow Jones 30 Companies
company_list = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DOW', 'XOM', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
                'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'VZ', 'V', 'WMT', 'WBA', 'DIS']

#Downloading 10 year daily stock data
stocks = yf.download(company_list, start=datetime.datetime(2010, 1, 1),
                     end=datetime.datetime(2020, 1, 1))['Adj Close'].astype('float16')
stocks.dropna(inplace=True,axis=1,how='all')

#Finding 15 companies with the heighest sharpe ratio's
stocks_list = (stocks / stocks.shift(1)) - 1
SR_stocks = (stocks_list.mean() / stocks_list.std() * np.sqrt(252))
sorted_arr = np.array(SR_stocks.values).argsort()[-10:][::-1]
new_stocks = stocks.iloc[:, sorted_arr]

#Creating array of all combinations of companies and portfolio size
# Create all combinations
comb = []
for item in np.arange(4, 9, 1):
    comb_obj = combinations(list(np.arange(0, 10, 1)), item)
    for i in list(comb_obj):
        comb.append(i)

#Arrays to store results of porfolio optimisation
SR_results=np.zeros((len(comb),1))
ExpRet_results=np.zeros((len(comb),1))
Vol_results=np.zeros((len(comb),1))

for idx,tuple in enumerate(comb):
    print('Progress Percentage: {:.2f} %'.format((idx*100)/len(comb)))
    log_ret = np.log(new_stocks.iloc[:,np.array(tuple)]/new_stocks.iloc[:,np.array(tuple)].shift(1))

    def get_ret_vol_sr(weights):
        ret = np.sum(log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])

    def neg_sharpe(weights):
        weights = weights / np.sum(weights)
        return get_ret_vol_sr(weights)[2] * -1

    def f(x):
        n_particles = x.shape[0]
        j = [neg_sharpe(x[i]) for i in range(n_particles)]
        return np.array(j)
    
    #Initiating Particle Swarm Global Minimum Optimisation
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    optimizer = pso.single.GlobalBestPSO(n_particles=100,
                                         dimensions=len(tuple),
                                         options=options,
                                         bounds=(np.zeros((1, len(tuple))), np.ones((1, len(tuple)))),
                                         ftol=-1e-03)
                                   
    cost, pos = optimizer.optimize(f, iters=25,n_processes=4)
    SR_results[idx]=(cost*-1)
    ExpRet_results[idx]=get_ret_vol_sr(pos/np.sum(pos))[0]
    Vol_results[idx]=get_ret_vol_sr(pos/np.sum(pos))[1]

#Plotting Results
plt.scatter(Vol_results,ExpRet_results,c=SR_results,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Stock Portfolios')

#Plotting Portfolio with Heighest Sharpe Ratio
index_SR_MAX=SR_results.argmax()
plt.scatter(Vol_results[index_SR_MAX],ExpRet_results[index_SR_MAX],c='red',s=50,edgecolors='black')
plt.show()

