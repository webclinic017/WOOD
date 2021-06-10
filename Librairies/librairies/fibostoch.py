import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

__author__ = 'LumberJack'

##### DAG26 5781 (c) #####
#####
##### df : dataframe with OHLC => Need Close, Low & High
#####


def Fibo_Stoch(df):
    scaler = MinMaxScaler((0,100))
    FIBO = []
    ##### Determine la std() du Close avec un rolling window de 20
    df['Std'] = scaler.fit_transform(df.Close.rolling(20).std().values.reshape(-1,1))
    for i in tqdm(range(len(df))):

        #####  Etabli une fenetre pour le calcul de Stoch selon suite Fibonacci
        if 90 < df.Std[i] <= 100 :
            FIBO.append(2)
        elif 80 < df.Std[i] <= 90 :
            FIBO.append(3)
        elif 70 < df.Std[i] <= 80 :
            FIBO.append(5)
        elif 60 < df.Std[i] <= 70 :
            FIBO.append(8)
        elif 50 < df.Std[i] <= 60 :
            FIBO.append(13)
        elif 40 < df.Std[i] <= 50 :
            FIBO.append(21)
        elif 30 < df.Std[i] <= 40 :
            FIBO.append(34)
        elif 20 < df.Std[i] <= 30 :
            FIBO.append(55)
        elif 10 < df.Std[i] <= 20 :
            FIBO.append(89)
        elif 0 <= df.Std[i] <= 10 :
            FIBO.append(144)
        else:
            FIBO.append(999)
        
    df['Fibo'] = FIBO
    
    ##### Calcul du %K Stochastic ajusté selon la std 
    df['LowRolling'] = [df.Low.rolling(df.Fibo[i]).min()[i] for i in tqdm(range(len(df)))]
    df['HighRolling'] = [df.High.rolling(df.Fibo[i]).max()[i] for i in tqdm(range(len(df)))]
    df['FiboStoch'] = (df.Close - df.LowRolling) / (df.HighRolling - df.LowRolling) * 100

    #df['FiboStoch'] = [((df.Close[i] - df.Low.rolling(df.Fibo[i]).min()[i])/(df.High.rolling(df.Fibo[i]).max()[i] - df.Low.rolling(df.Fibo[i]).min()[i]))*100 \
     #               for i in tqdm(range(len(df)))]

    return(df)