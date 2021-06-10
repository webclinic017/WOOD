import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import datetime as dt
from tqdm import tqdm

__author__ = 'LumberJack'

##### DAG26 5781 (c) #####
#####
##### _fibo : seuil du fibonacci exprimé en fraction. Ex : pour 61.8% => 0.618
#####
##### _err : Marge d'erreur du déclenchement du signal => (1 + _err) * _fibo
#####
##### _verbose : 1 Si on veut afficher les print, 2 si on veut afficher les graphes et 3 si on veut afficher les deux. Sinon 0
#####
##### _save_pic : 1 si on veut sauvegarder une image du graphe. /!\ _verbose doit être à 2 ou 3 /!\

def ret_hilo(df,_fibo,_err,_verbose,_save_pic):
    LASTP1 = []
    LASTV1 = []
    LASTP2 = []
    LASTV2 = []
    lastv1 = 0
    lastp1 = 0
    lastv2 = 0
    lastp2 = 0
    lastp = 0
    lastv = 0

    for i in tqdm(range(len(df))):
        if df.BackLo_val[i] == 1:
            lastv2 = lastv1
            lastv1 = df.Low[i]
            LASTV1.append(lastv1)
            LASTP1.append(lastp1)
            LASTV2.append(lastv2)
            LASTP2.append(lastp2)
        elif df.BackHi_pic[i] == 1:
            lastp2 = lastp1
            lastp1 = df.High[i]
            LASTP1.append(lastp1)
            LASTV1.append(lastv1)
            LASTP2.append(lastp2)
            LASTV2.append(lastv2)
        else:
            LASTP1.append(lastp1)
            LASTV1.append(lastv1)
            LASTP2.append(lastp2)
            LASTV2.append(lastv2)

    df['LastP1'] = LASTP1
    df['LastV1'] = LASTV1
    df['LastP2'] = LASTP2
    df['LastV2'] = LASTV2
    df['XAp'] =  df['LastV2'] -  df['LastP2']
    df['XAv'] = df['LastP2'] -  df['LastV2']
    df['ABp'] = df['LastP1'] -  df['LastV2']
    df['ABv'] = df['LastV1'] -  df['LastP2']
    df['BCp'] = df['LastV1'] -  df['LastP1']
    df['BCv'] = df['LastP1'] -  df['LastV1']
    df['CDp'] = df['High'] - df['LastV1']
    df['CDv'] = df['Low'] - df['LastP1']

    BUY = [] 
    SELL = [] 
    INDEXv = []
    VALUEv = []
    INDEXp = []
    VALUEp = []
    _pic = 0
    _val = 0
    _picklock = 0
    _locked = 1

    for i in tqdm(range(0,len(df))):

        if df.BackLo_val[i] == 1:
            if _verbose == 1 or _verbose == 3:
                print('\nVAL a 1, Initiation de recherche PIC en',i,'\n')
                print('\nAffichage CDp',df.CDp[i],' CDv',df.CDv[i],'\n')
            _pic = 1
            _val = 0
            _locked = 0
            _picklock += 1
        elif df.BackHi_pic[i] == 1:
            if _verbose == 1 or _verbose == 3:
                print('\nPIC a 1, Initiation de recherche VAL en',i,'\n')
                print('\nAffichage CDp',df.CDp[i],' CDv',df.CDv[i],'\n')
            _val = 1
            _pic = 0
            _locked = 0
            _picklock += 1
        if _picklock < 4:
            _locked = 1


        if _val == 1 and _locked == 0 and abs(df.CDv[i]/df.BCv[i]) > _fibo * (1+_err) :
            if _verbose == 1 or _verbose == 3:
                print('\n[Mode_VALLEY = ON]\nDate  :',df.index[i],'i : ',i,'\nLow : ',df.Low[i],'\nLastP : ',df.LastP1[i],'\nLastV : ',df.LastV1[i],'\nfibo*(1+err : ',_fibo * (1+_err))
                print('\nho : ',abs(df.Low[i] - df.LastP1[i]),'\n ba : ',abs(df.LastP1[i] - df.LastV1[i]),'\nletou quessé de lunsurlotr: ',abs((df.Low[i] - df.LastP1[i]) / (df.LastP1[i] - df.LastV1[i])),'\n_____________________\n\n')
            VALUEv.append(df.Low[i])
            INDEXv.append(df.index[i])
            BUY.append(1)
            SELL.append(0)
            _val = 0
            _locked = 1

        elif _pic == 1 and _locked == 0 and abs(df.CDp[i]/df.BCp[i]) > _fibo * (1+_err): 
            if _verbose == 1 or _verbose == 3:
                print('\n[Mode_PEAK = ON]\nDate  :',df.index[i],'i : ',i,'\nHigh : ',df.High[i],'\nLastV : ',df.LastV1[i],'\nLastP : ',df.LastP1[i],'\nfibo*(1+err : ',_fibo * (1+_err))
                print('\nho : ',abs(df.CDp[i]),'\n ba : ',abs(df.BCp[i]),'\nletou quessé de lunsurlotr: ',abs(df.CDp[i]/df.BCp[i]),'\n_____________________\n\n')
            VALUEp.append(df.High[i])
            INDEXp.append(df.index[i])
            BUY.append(0)
            SELL.append(1)
            _pic = 0
            _locked = 1
            
        else:
            BUY.append(0)
            SELL.append(0)
        
    df['BuyRet'] = BUY
    df['SellRet'] = SELL

    if _verbose == 2 or _verbose == 3:

        df_zigzag = pd.DataFrame()
        df_zigzag['PeakValley'] = pd.concat([df.BackHi_pic, df.BackLo_val], axis=0, ignore_index=False, sort=True)
        df_zigzag['Close'] = pd.concat([df.High, df.Low], axis=0, ignore_index=False, sort=True)
        df_zigzag = df_zigzag[df_zigzag.PeakValley==1]
        # Sort peak and valley datapoints by date.
        df_zigzag = df_zigzag.sort_index()

        plt.figure(figsize=(22,4))
        plt.title('Base Retracement du '+str(df.index[0])+' au '+str(df.index[-1]))
        plt.plot(df.Close,color='black',alpha=0.4,label="Close")
        #plt.plot(df.High,color='orange',alpha=0.3,label="High")
        #plt.plot(df.Low,color='blue',alpha=0.3,label="Low")
        #plt.plot(talib.SMA(df.Close, 200),color='blue',alpha=0.6,label='SMA')


        plt.scatter(x=INDEXp,y=VALUEp,color='red',marker='v',label="Sell")
        plt.scatter(x=INDEXv,y=VALUEv,color='green',marker='^',label="Buy")

        plt.scatter(x=df[df.BackHi_pic==1].index,y=df[df.BackHi_pic==1].High,color='purple',marker='x',label="Peak",alpha=0.6)
        plt.scatter(x=df[df.BackLo_val==1].index,y=df[df.BackLo_val==1].Low,color='purple',marker='o',label="Valley",alpha=0.6)
        plt.plot(df_zigzag.index.values, df_zigzag['Close'].values,color='purple', label="zigzag",alpha=0.6)
        plt.legend()
        if _save_pic == 1:
            plt.savefig('Retracements sur Hi Lo.png',dpi=1000)
    return(df)

    def ret_close(f,_fibo,_err,_verbose,_save_pic):
        LASTP1 = []
        LASTV1 = []
        LASTP2 = []
        LASTV2 = []
        lastv1 = 0
        lastp1 = 0
        lastv2 = 0
        lastp2 = 0
        lastp = 0
        lastv = 0

        for i in tqdm(range(len(df))):
            if df.BackLo_val[i] == 1:
                lastv2 = lastv1
                lastv1 = df.Close[i]
                LASTV1.append(lastv1)
                LASTP1.append(lastp1)
                LASTV2.append(lastv2)
                LASTP2.append(lastp2)
            elif df.BackHi_pic[i] == 1:
                lastp2 = lastp1
                lastp1 = df.Close[i]
                LASTP1.append(lastp1)
                LASTV1.append(lastv1)
                LASTP2.append(lastp2)
                LASTV2.append(lastv2)
            else:
                LASTP1.append(lastp1)
                LASTV1.append(lastv1)
                LASTP2.append(lastp2)
                LASTV2.append(lastv2)

        df['LastP1'] = LASTP1
        df['LastV1'] = LASTV1
        df['LastP2'] = LASTP2
        df['LastV2'] = LASTV2
        df['XAp'] =  df['LastV2'] -  df['LastP2']
        df['XAv'] = df['LastP2'] -  df['LastV2']
        df['ABp'] = df['LastP1'] -  df['LastV2']
        df['ABv'] = df['LastV1'] -  df['LastP2']
        df['BCp'] = df['LastV1'] -  df['LastP1']
        df['BCv'] = df['LastP1'] -  df['LastV1']
        df['CDp'] = df['Close'] - df['LastV1']
        df['CDv'] = df['Close'] - df['LastP1']

        BUY = [] 
        SELL = [] 
        INDEXv = []
        VALUEv = []
        INDEXp = []
        VALUEp = []
        _pic = 0
        _val = 0
        _picklock = 0
        _locked = 1

        for i in tqdm(range(0,len(df))):

            if df.BackLo_val[i] == 1:
                if _verbose == 1 or _verbose == 3:
                    print('\nVAL a 1, Initiation de recherche PIC en',i,'\n')
                    print('\nAffichage CDp',df.CDp[i],' CDv',df.CDv[i],'\n')
                _pic = 1
                _val = 0
                _locked = 0
                _picklock += 1
            elif df.BackHi_pic[i] == 1:
                if _verbose == 1 or _verbose == 3:
                    print('\nPIC a 1, Initiation de recherche VAL en',i,'\n')
                    print('\nAffichage CDp',df.CDp[i],' CDv',df.CDv[i],'\n')
                _val = 1
                _pic = 0
                _locked = 0
                _picklock += 1
            if _picklock < 4:
                _locked = 1


            if _val == 1 and _locked == 0 and abs(df.CDv[i]/df.BCv[i]) > _fibo * (1+_err) :
                if _verbose == 1 or _verbose == 3:
                    print('\n[Mode_VALLEY = ON]\nDate  :',df.index[i],'i : ',i,'\nClose : ',df.Close[i],'\nLastP : ',df.LastP1[i],'\nLastV : ',df.LastV1[i],'\nfibo*(1+err : ',_fibo * (1+_err))
                    print('\nho : ',abs(df.Close[i] - df.LastP1[i]),'\n ba : ',abs(df.LastP1[i] - df.LastV1[i]),'\nletou quessé de lunsurlotr: ',abs((df.Close[i] - df.LastP1[i]) / (df.LastP1[i] - df.LastV1[i])),'\n_____________________\n\n')
                VALUEv.append(df.Close[i])
                INDEXv.append(df.index[i])
                BUY.append(1)
                SELL.append(0)
                _val = 0
                _locked = 1

            elif _pic == 1 and _locked == 0 and abs(df.CDp[i]/df.BCp[i]) > _fibo * (1+_err): 
                if _verbose == 1 or _verbose == 3:
                    print('\n[Mode_PEAK = ON]\nDate  :',df.index[i],'i : ',i,'\nClose : ',df.Close[i],'\nLastV : ',df.LastV1[i],'\nLastP : ',df.LastP1[i],'\nfibo*(1+err : ',_fibo * (1+_err))
                    print('\nho : ',abs(df.CDp[i]),'\n ba : ',abs(df.BCp[i]),'\nletou quessé de lunsurlotr: ',abs(df.CDp[i]/df.BCp[i]),'\n_____________________\n\n')
                VALUEp.append(df.Close[i])
                INDEXp.append(df.index[i])
                BUY.append(0)
                SELL.append(1)
                _pic = 0
                _locked = 1
                
            else:
                BUY.append(0)
                SELL.append(0)
            
        df['BuyRet'] = BUY
        df['SellRet'] = SELL

        if _verbose == 2 or _verbose == 3:

            df_zigzag = pd.DataFrame()
            df_zigzag['PeakValley'] = pd.concat([df.BackHi_pic, df.BackLo_val], axis=0, ignore_index=False, sort=True)
            df_zigzag['Close'] = pd.concat([df.High, df.Low], axis=0, ignore_index=False, sort=True)
            df_zigzag = df_zigzag[df_zigzag.PeakValley==1]
            # Sort peak and valley datapoints by date.
            df_zigzag = df_zigzag.sort_index()

            plt.figure(figsize=(22,4))
            plt.title('Base Retracement du '+str(df.index[0])+' au '+str(df.index[-1]))
            plt.plot(df.Close,color='black',alpha=0.4,label="Close")
            #plt.plot(df.High,color='orange',alpha=0.3,label="High")
            #plt.plot(df.Low,color='blue',alpha=0.3,label="Low")
            #plt.plot(talib.SMA(df.Close, 200),color='blue',alpha=0.6,label='SMA')


            plt.scatter(x=INDEXp,y=VALUEp,color='red',marker='v',label="Sell")
            plt.scatter(x=INDEXv,y=VALUEv,color='green',marker='^',label="Buy")

            plt.scatter(x=df[df.BackHi_pic==1].index,y=df[df.BackHi_pic==1].High,color='purple',marker='x',label="Peak",alpha=0.6)
            plt.scatter(x=df[df.BackLo_val==1].index,y=df[df.BackLo_val==1].Low,color='purple',marker='o',label="Valley",alpha=0.6)
            plt.plot(df_zigzag.index.values, df_zigzag['Close'].values,color='purple', label="zigzag",alpha=0.6)
            plt.legend()
            if _save_pic == 1:
                plt.savefig('Retracements sur Hi Lo.png',dpi=1000)
        return(df)