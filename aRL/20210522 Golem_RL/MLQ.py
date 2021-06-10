__author__ = 'LumberJack (Jyss)'
__copyright__ = 'D.A.G. 26 - 5781'
__version__ = 'v0.1'

####################################################################
####################################################################
############################### GOLEM FX ###########################
####################################################################
####################################################################

'''
The goal of this version is to make a motor for our system using DQN
'''


import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas_datareader as data_reader
import time
from tqdm import tqdm_notebook, tqdm
from natsort import natsorted
from collections import deque
import joblib
import colorama as col

x = 'EUR/USD'
_period = 'm5'
_period2 = 'H1'
_ticker = x.replace('/','')
_start = '2010-01-01' # start the train there '2010-01-01'
_mid = '2016-06-30' # stop the train and begin the test there '2016-08-31'
_stop = '2017-12-31' # stop the test there. After that, it is kept for oos '2017-12-31'
_last = '2021-04-29' # '2020-12-31'

window_size = 10
episodes = 1000
batch_size = 32
#data_samples = 500 #len(df_train) - 1


def dataset_loader():

    data = joblib.load('BASES/EURUSD_m5')

    start_date = str(data.index[0]).split()[0]
    end_date = str(data.index[1]).split()[0]

    _sl = 0.001
    _target = 0.002

    # scaler = Normalizer()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()

    def strategy(df):
        ##### CONDITIONS LONG
        _condition_1 = (df.slow_K5 < 20) & (df.slow_K5.shift(1) < df.slow_D5.shift(1)) & (df.slow_K5 > df.slow_D5)

        ##### CONDITIONS SHORT
        _condition_1_bar = (df.slow_K5 > 80) & (df.slow_K5.shift(1) > df.slow_D5.shift(1)) & (df.slow_K5 < df.slow_D5)

        ##### 1 condition
        df['Signal'] = np.where(_condition_1,1,np.where(_condition_1_bar,-1,0))
        df = df.drop(['Symbol','Date','DateIndex','SB_Gamma'], axis=1)
        return(df.sort_index(axis=0)) 

    def strategy5(df,_window=40):

        #df['RSI_2'] = TA.RSI(ohlc=df,int=2,str='Close')

        df['Window_High_Ask'] = df.HighAsk.iloc[::-1].rolling(_window).max().iloc[::-1] # Limite SL Short
        df['Window_High_Bid'] = df.HighBid.iloc[::-1].rolling(_window).max().iloc[::-1] # Limite Target Long
        df['Window_Low_Ask'] = df.LowAsk.iloc[::-1].rolling(_window).min().iloc[::-1] # Limite Target Short
        df['Window_Low_Bid'] = df.LowBid.iloc[::-1].rolling(_window).min().iloc[::-1] # Limite SL Long
        df['Window_sl_Short'] = df.CloseBid + (df.CloseBid * _sl) # Short pour SL
        df['Window_sl_Long'] = df.CloseAsk - (df.CloseAsk * _sl) # Long pour SL
        df['Window_tp_Short'] = df.CloseBid - (df.CloseBid * _target) # Short pour TP
        df['Window_tp_Long'] = df.CloseAsk + (df.CloseAsk * _target) # Long pour TP

        ##### CONDITIONS LONG
        _condition_1 = (df['Window_tp_Long'] <= df['Window_High_Bid']) & (df['Window_sl_Long'] <= df['Window_Low_Bid'])

        ##### CONDITIONS SHORT
        _condition_1_bar = (df['Window_tp_Short'] >= df['Window_Low_Ask']) & (df['Window_sl_Short'] >= df['Window_High_Ask'])

        ##### 1 condition
        df['Signal'] = np.where(_condition_1,1,np.where(_condition_1_bar,-1,0))
        df = df.drop(['Symbol','Date','DateIndex','Window_High_Ask','Window_High_Bid','Window_Low_Ask','Window_Low_Bid','Window_sl_Short','Window_sl_Long','Window_tp_Short','Window_tp_Long'], axis=1)
        return(df.sort_index(axis=0))


    data = klines(data)

    data = strategy(data)

    data = data[['Body','Color','UpperShadow','LowerShadow','Signal']]

    data['Body1'] = data['Body'] - data['Body'].shift(1)
    data['Body2'] = data['Body'] - data['Body'].shift(2)
    data['Body3'] = data['Body'] - data['Body'].shift(3)
    data['Body4'] = data['Body'] - data['Body'].shift(4)
    data['Body5'] = data['Body'] - data['Body'].shift(5)

    data['UpperShadow1'] = data['UpperShadow'] - data['UpperShadow'].shift(1)
    data['UpperShadow2'] = data['UpperShadow'] - data['UpperShadow'].shift(2)
    data['UpperShadow3'] = data['UpperShadow'] - data['UpperShadow'].shift(3)
    data['UpperShadow4'] = data['UpperShadow'] - data['UpperShadow'].shift(4)
    data['UpperShadow5'] = data['UpperShadow'] - data['UpperShadow'].shift(5)

    data['LowerShadow1'] = data['LowerShadow'] - data['LowerShadow'].shift(1)
    data['LowerShadow2'] = data['LowerShadow'] - data['LowerShadow'].shift(2)
    data['LowerShadow3'] = data['LowerShadow'] - data['LowerShadow'].shift(3)
    data['LowerShadow4'] = data['LowerShadow'] - data['LowerShadow'].shift(4)
    data['LowerShadow5'] = data['LowerShadow'] - data['LowerShadow'].shift(5)


    df_train, df_test, df_oos = split_df(data)

    df_oos_raw = df_oos.copy().dropna()

    df_train.sort_index(inplace=True)
    
    
    for i in df_train.columns.unique():
        if i != 'Signal' and i != 'Color':
            df_train[i] = scaler.fit_transform(df_train[i].values.reshape(-1, 1))
            df_test[i] = scaler.fit_transform(df_test[i].values.reshape(-1, 1))
            df_oos[i] = scaler.fit_transform(df_oos[i].values.reshape(-1, 1))

    df_train = df_train.dropna()
    df_test = df_test.dropna()
    df_oos = df_oos.dropna()

    signal_train = df_train['Signal']
    signal_test = df_test['Signal']
    signal_oos = df_oos['Signal']

    df_train = df_train.drop(['Signal'],axis=1)
    df_test = df_test.drop(['Signal'],axis=1)
    df_oos = df_oos.drop(['Signal'],axis=1)

    df_oos = df_oos.reindex(natsorted(df_oos.columns), axis=1)
    df_test = df_test.reindex(natsorted(df_test.columns), axis=1)
    df_train = df_train.reindex(natsorted(df_train.columns), axis=1)
    df_oos_raw = df_oos_raw.reindex(natsorted(df_oos_raw.columns), axis=1)

    return df_train, df_test, df_oos, df_oos_raw, signal_train, signal_test, signal_oos

def state_creator(data, timestep): #, window_size):

    state = []
    state.append(data.iloc[timestep,:])
        
    return np.array([state])

def klines(df):
    _condition1 = df.Close >= df.Open
    df['Color'] = np.where(_condition1,1,-1)
    _condition2 = df.Color = 1
    df['UpperShadow'] = np.where(_condition2,(df.High-df.Close),(df.High-df.Open))
    df['LowerShadow'] = np.where(_condition2,(df.Open-df.Low),(df.Close-df.Low))
    df['Body'] = abs(df.Close-df.Open)
    return (df)

def split_df(df):
    """[Split the dtaframe in train/test/oos and reduce train and test to dataframe with signal!=0 only]

    Args:
        df ([pandas]): [the datframe to split already featured]
        _start ([date]): [beginning of the df]
        _mid ([date]): [stop of the train and beginning of the test]
        _stop ([date]): [stop of the test and beginning of the oos]
        _last ([date]): [end of the oos]
    """    
    df = df.dropna()
    df['Date'] = pd.to_datetime(df.index)
    
    df_train = df[(df.Date>=_start)&(df.Date<=_mid)]
    df_test = df[(df.Date>_mid)&(df.Date<=_stop)]
    df_oos = df[(df.Date>_stop)&(df.Date <= _last)]
    df_train = df_train.drop(['Date'],axis=1)
    df_test = df_test.drop(['Date'],axis=1)
    df_oos = df_oos.drop(['Date'],axis=1)
    return(df_train, df_test,df_oos)

df_train, df_test, df_oos, df_oos_raw, signal_train, signal_test, signal_oos = dataset_loader()

state_size = df_train.shape[1]



class Golem():
  
    def __init__(self, state_size=state_size, action_space=3, model_name="AITrader"):

        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=5000)
        self.inventory = []
        self.model_name = model_name
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.model = self.model_builder()

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(df_train.shape[1]))
        model.add(tf.keras.layers.Dense(units=64, activation='relu')) # , input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=3, activation='relu'))
        #model = tf.model({inputs: input, outputs: dense2})
        model.output_shape
        model.compile(optimizers.Adam(lr=0.001),loss='mse')
        return model 

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        
        action = self.model.predict(state) #### NOT VALID
        action = np.amax(action)
        return action

    def batch_train(self, batch_size):

        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

            for state, action, reward, next_state, done in batch:
                reward = reward
            
                if not done:
                    reward = reward + self.gamma * np.amax(self.model.predict(next_state))

                target = self.model.predict(state)
                target[0][action] = reward

                self.model.fit(state, target, epochs=150, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":

    trader = Golem(df_train,window_size,state_size)
    print(trader.model.summary())
    data_samples = len(df_train - 1)
    for episode in range(1, episodes + 1):
  
        print(col.Fore.LIGHTRED_EX+"Episode: {}/{}".format(episode, episodes)+col.Style.RESET_ALL)
        
        # state = state_creator(df_train, 0)
        state = np.array([df_train.iloc[0:11,:]])
        action = np.array(trader.trade([state]))
        
        total_profit = 0
        trader.inventory = []
        
        for t in tqdm(range(data_samples)):
            
            # next_state = state_creator(data, t+1, window_size + 1)
            next_state = df_train.iloc[t+1:12,:]
            next_state = np.array([next_state])
            
            reward = 0

            if action == 1 and signal_train[t] == 1: #Buying
                reward = 1
                total_profit += reward
                
            elif action == 2 and signal_train[t] == -1: #Selling
                reward = 1
                total_profit += reward

            elif action == 0 and signal_train[t] == 0: #Selling
                reward = 0
                total_profit += reward

            elif action == 0 and signal_train[t] != 0: #Selling
                reward = -1
                total_profit += reward

            elif action != 0 and signal_train[t] == 0: #Selling
                reward = -1
                total_profit += reward
                
            if t == data_samples - 1:
                done = True
            else:
                done = False
                
            trader.memory.append((state, action, reward, next_state, done))

            state = next_state

            if done:
                time.sleep(0.2)
                print("\n"+col.Fore.CYAN+"########################"+col.Fore.GREEN)
                print("TOTAL PROFIT:  {} ".format(total_profit))
                print(col.Fore.CYAN+"########################"+col.Style.RESET_ALL)
                print()

            if len(trader.memory) > batch_size:
                trader.batch_train(batch_size)
                
            if episode % 10 == 0:
                trader.model.save("ai_trader_{}.h5".format(episode))

