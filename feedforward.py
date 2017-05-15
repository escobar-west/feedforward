import datetime
import numpy as np
from pandas_datareader import data
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

symbol  = 'SPY'
start   = datetime.datetime(2013, 1, 1)
end     = datetime.datetime.today()
alpha   = 0.9
horizon = 1
hidden_layer_sizes = (150, 200)
split_ratio = 0.8

stock = data.DataReader(symbol, 'google', start, end)[['Close', 'High', 'Low', 'Volume']]

stock.Close = np.log(stock.Close)
stock.High = np.log(stock.High)
stock.Low = np.log(stock.Low)

stock['Diff'] = stock.Close.diff()

########## MACD ##########
period = 12
EMA12 = [np.NaN] * (period-1)
EMA12.append(stock.Close[:period].mean())

multiplier = 2 / (period+1)

for val in stock.Close[period:]:
    EMA12.append( multiplier * val + (1-multiplier) * EMA12[-1] )

period = 26
EMA26 = [np.NaN] * (period-1)
EMA26.append(stock.Close[:period].mean())

multiplier = 2 / (period+1)

for val in stock.Close[period:]:
    EMA26.append( multiplier * val + (1-multiplier) * EMA26[-1] )

stock['MACD_Line'] = np.array(EMA12) - np.array(EMA26)


period = 9

MACD = stock.MACD_Line.dropna()
MACD_EMA = [np.NaN] * (25 + (period-1))
MACD_EMA.append(MACD[:period].mean())

multiplier = 2 / (period+1)

for val in MACD[period:]:
    MACD_EMA.append( multiplier * val + (1-multiplier) * MACD_EMA[-1] )

stock['MACD_EMA'] = MACD_EMA

stock['MACD_Signal'] = stock.MACD_Line - stock.MACD_EMA

######### CMF #########
stock['MFM'] = (2*stock.Close - stock.High - stock.Low) / (stock.High-stock.Low)

stock['MFV'] = stock.MFM * stock.Volume

CMF = [np.NaN] * 19

for i in range(20, len(stock)+1):
    CMF.append( stock.MFV[i-20:i].sum() / stock.Volume[i-20:i].sum() )
    
stock['CMF'] = CMF

######### Target #########
weights = [alpha] ** np.arange(0, horizon)
weights /= weights.sum()

target = []
for i in range(1, len(stock)-horizon+1):
    target.append( (weights*stock.Diff[i:i+horizon]).sum() )
    
target = target + [np.NaN] * (horizon)
stock['Target'] = target

stock.drop(['High', 'Low', 'MACD_EMA', 'MFM', 'MFV'],inplace=True,axis=1)

######### MLP #########
data = stock[['Diff', 'Volume', 'MACD_Line', 'MACD_Signal', 'CMF', 'Target']]
for name in data.columns:
    data.loc[:, name] /= data[name].std()
    
data.dropna(inplace=True)

split = int(np.floor(split_ratio*len(data)))

input_train = data[['Diff', 'Volume', 'MACD_Line', 'MACD_Signal', 'CMF']][:split]
target_train = data.Target[:split]

input_val = data[['Diff', 'Volume', 'MACD_Line', 'MACD_Signal', 'CMF']][split:]
target_val = data.Target[split:]

mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver='lbfgs')
mlp.fit(input_train, target_train)

pred_train = pd.Series( mlp.predict(input_train), index = input_train.index )
results_train = pd.concat({'target': target_train, 'pred': pred_train}, axis=1)

pred_val = pd.Series( mlp.predict(input_val), index = input_val.index )
results_val = pd.concat({'target': target_val, 'pred': pred_val}, axis=1)

######### Paper Trade #########
data['Pred'] = results_val.pred
data['Buy'] = data.loc[:,'Pred'] > 0
stock['Buy'] = data.loc[:,'Pred'] > 0

print(stock.Buy[stock.Buy == True].first_valid_index())
print(stock.loc[stock.Buy[stock.Buy == True].first_valid_index(), 'Close'])
portfolio = [np.NaN]
portfolio.append( stock.loc[stock.Buy[stock.Buy == True].first_valid_index(), 'Close'] )

for i in range(1, len(stock)-1):
    if stock.Buy[i] == np.NaN:
        pass
    elif stock.Buy[i] == True:
        portfolio.append( portfolio[-1] + stock.Diff[i+1] )
    else:
        portfolio.append( portfolio[-1] )
        
stock['Portfolio'] = portfolio

print(stock[['Close', 'Diff', 'Buy', 'Portfolio']])

print(stock.corr())
print(results_train.corr())
print(results_val.corr())

stock.Close.plot(rot=45)
stock.Portfolio.plot(rot=45)
plt.show()
