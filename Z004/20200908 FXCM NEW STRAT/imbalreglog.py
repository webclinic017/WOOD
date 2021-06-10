###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
#################################      I M B A L A N C E D      R E G L O G        ########################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'


import configparser
config = configparser.ConfigParser()
from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from numpy import sqrt
from numpy import argmax
import colorama as col

import pyttsx3
engine = pyttsx3.init()

engine.say("Imbalanced, Logistic Regression")
engine.runAndWait()

TICKERS = ['AUD/CHF']
x = TICKERS[0]
config.read('config.ini')
_period1 = config.get('TIMEFRAME','_period1')
_path4 = config.get('PATH','_path4')
_path = _path4
config['PATH']['_path'] = _path
with open('config.ini', 'w') as configfile: # save
    config.write(configfile)

def imbal_reglog(x):
    globals()['df1_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period1+'.csv')

    df = globals()['df1_%s' %x.replace('/','')].copy()

    print(col.Fore.BLUE,'Regression Logistique Imbalanced pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)

    try:
        df = df.drop(['Unnamed: 0'],axis=1)
    except:
        pass

    BUY = []
    SELL = []
    for i in range(len(df)):
        if df.Signal[i] == 1:
            BUY.append(1)
            SELL.append(0)
        elif df.Signal[i] == -1:
            BUY.append(0)
            SELL.append(1)
        else:
            BUY.append(0)
            SELL.append(0)
    df['Buy'] = BUY
    df['Sell'] = SELL
    df = df.drop(['Date','Total','Signal'],axis=1)

    X = df.iloc[:,:-2]

    yb = df['Buy']
    yv = df['Sell']
    # summarize class distribution
    counterb = Counter(yb)
    counterv = Counter(yv)
    print('Signaux achat avant oversampling',counterb)
    print('Signaux vente avant oversampling',counterv)

    oversample = SMOTE(sampling_strategy=0.5)

    Xb_over, yb_over = oversample.fit_resample(X, yb)
    Xv_over, yv_over = oversample.fit_resample(X, yv)

    print('Signaux achat après oversampling',Counter(yb_over))
    print('Signaux vente après oversampling',Counter(yv_over))

    # split into train/test sets with same class ratio
    trainXb, testXb, trainyb, testyb = train_test_split(Xb_over, yb_over, test_size=0.5, stratify=yb_over)
    trainXv, testXv, trainyv, testyv = train_test_split(Xv_over, yv_over, test_size=0.5, stratify=yv_over)
    # define model
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
    # fit model
    model.fit(trainXb, trainyb)
    # predict on test set
    yhatb = model.predict(testXb)
    # evaluate predictions
    print('Achat - Accuracy: %.3f' % accuracy_score(testyb, yhatb))
    print('Achat - Precision: %.3f' % precision_score(testyb, yhatb))
    print('Achat - Recall: %.3f' % recall_score(testyb, yhatb))
    print('Achat - F-measure: %.3f' % f1_score(testyb, yhatb))

    model.fit(trainXv, trainyv)
    # predict on test set
    yhatv = model.predict(testXv)
    # evaluate predictions
    print('Vente - Accuracy: %.3f' % accuracy_score(testyv, yhatv))
    print('Vente - Precision: %.3f' % precision_score(testyv, yhatv))
    print('Vente - Recall: %.3f' % recall_score(testyv, yhatv))
    print('Vente - F-measure: %.3f' % f1_score(testyv, yhatv))

    fprv, tprv, thresholdsv = roc_curve(testyv, yhatv)
    # calculate the g-mean for each threshold
    gmeansv = sqrt(tprv * (1-fprv))
    # locate the index of the largest g-mean
    ixv = argmax(gmeansv)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholdsv[ixv], gmeansv[ixv]))
    # plot the roc curve for the model
    plt.figure(figsize=(12,4))
    plt.suptitle('ROC Curve pour les ventes')
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fprv, tprv, marker='.', label='Logistic')
    plt.scatter(fprv[ixv], tprv[ixv], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()

    precisionb, recallb, thresholdsb = precision_recall_curve(testyb, yhatb)
    # plot the roc curve for the model
    no_skillb= len(testyb[testyb==1]) / len(testyb)

    # convert to f score
    fscoreb = (2 * precisionb * recallb) / (precisionb + recallb)
    # locate the index of the largest f score
    ixb = argmax(fscoreb)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholdsb[ixb], fscoreb[ixb]))

    plt.figure(figsize=(12,4))
    plt.suptitle('Recall_Precision Curve pour les achats')
    plt.plot([0,1], [no_skillb,no_skillb], linestyle='--', label='No Skill')
    plt.plot(recallb, precisionb, marker='.', label='Logistic')
    plt.scatter(recallb[ixb], precisionb[ixb], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # show the plot
    plt.show()

    precisionv, recallv, thresholdsv = precision_recall_curve(testyv, yhatv)
    # plot the roc curve for the model
    no_skillv= len(testyv[testyv==1]) / len(testyv)

    # convert to f score
    fscorev = (2 * precisionv * recallv) / (precisionv + recallv)
    # locate the index of the largest f score
    ixv = argmax(fscorev)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholdsv[ixv], fscorev[ixv]))

    plt.figure(figsize=(12,4))
    plt.suptitle('Recall_Precision Curve pour les ventes')
    plt.plot([0,1], [no_skillv,no_skillv], linestyle='--', label='No Skill')
    plt.plot(recallv, precisionv, marker='.', label='Logistic')
    plt.scatter(recallv[ixv], precisionv[ixv], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # show the plot
    plt.show()

    return()
