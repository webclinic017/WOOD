import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import datetime as dt
from tqdm import tqdm

__author__ = 'LumberJack'

##### DAG26 5781 (c) #####
#####
##### _deviation : seuil de return entre le High/Low actuel et le min/max du Low/High _backstep nougies avant
#####
##### _backstep : fenêtre de n bougies en arrière
#####
##### _depth : fenêtre de n bougies après
#####



def zigzag(df, _deviation , _backstep, _depth):
    ##### calcul du return sur backstep bougies : (actuel - initial) / initial

    ##### Calcul des max et des min sur les RetHiLo et les RetLoHi sur la fenêtre _backstep et répondant à l'amplitude _deviation
    try:
        ##### On fixe la date en index sous forme de Timestamp
        df.set_index(pd.to_datetime(df.Date),drop=True,inplace=True)

        ###### On drop les colonnes inutiles car drop=True bug
        df = df.drop(['Date','Total'],axis=1)
    except:
        pass

    df['Peak'] = np.where(df.High == df.rolling(_backstep).High.max(),1,0)
    df['Valley'] = np.where(df.Low == df.rolling(_backstep).Low.min(),1,0)

    PEAK = []
    VALLEY = []
    _pic = 0
    _val = 1
    _idx = 0

    for i in tqdm(range(0,len(df))):


        if df.Valley[i] == 1 and _val == 1 and df.Low[i] <= df.Low[i:i+_depth].min() and _idx==0:
            VALLEY.append(1)
            PEAK.append(0)
            _val = 0
            _pic = 1
            _idx = i
        
        elif df.Peak[i] == 1 and _pic == 1 and df.High[i] >= df.High[i:i+_depth].max() and _idx==0:
            VALLEY.append(0)
            PEAK.append(1)
            _pic = 0
            _val = 1
            _idx = i

        elif df.Valley[i] == 1 and _val == 1 and df.Low[i] <= df.Low[i:i+_depth].min() and ((df.Low[i] - df.High[_idx]) / df.High[_idx]) <= - _deviation:#*100
            VALLEY.append(1)
            PEAK.append(0)
            _val = 0
            _pic = 1
            _idx = i
        
        elif df.Peak[i] == 1 and _pic == 1 and df.High[i] >= df.High[i:i+_depth].max() and ((df.High[i] - df.Low[_idx]) / df.Low[_idx]) >= _deviation:#*100
            VALLEY.append(0)
            PEAK.append(1)
            _pic = 0
            _val = 1
            _idx = i

        else :
            VALLEY.append(0)
            PEAK.append(0)


    df['Peak'] = PEAK
    df['Valley'] = VALLEY
    
    df_zigzag = pd.DataFrame()
    
    print('Il y a ',sum(VALLEY),'signaux VALLEY, ',sum(PEAK),'signaux PEAK, et ',len(df)-sum(PEAK)-sum(VALLEY),' signaux sans rien' )

    df_zigzag['PeakValley'] = pd.concat([df.Peak, df.Valley], axis=0, ignore_index=False, sort=True)
    df_zigzag['Close'] = pd.concat([df.High, df.Low], axis=0, ignore_index=False, sort=True)
    df_zigzag = df_zigzag[df_zigzag.PeakValley==1]
    # Sort peak and valley datapoints by date.
    df_zigzag = df_zigzag.sort_index()
    
    

    plt.figure(figsize=(22,4))
    plt.title('ZigZag')
    plt.plot(df.Close,color='black',alpha=0.4,label="Close")
    #plt.plot(df.High,color='orange',alpha=0.3,label="High")
    #plt.plot(df.Low,color='blue',alpha=0.3,label="Low")
    plt.scatter(x=df[df.Peak==1].index,y=df[df.Peak==1].High,color='red',marker='v',label="Sell")
    plt.scatter(x=df[df.Valley==1].index,y=df[df.Valley==1].Low,color='green',marker='^',label="Buy")
    # Plot zigzag trendline.
    plt.plot(df_zigzag.index.values, df_zigzag['Close'].values,color='purple', label="zigzag",alpha=0.8)
    plt.legend()

    return(PEAK,VALLEY)
