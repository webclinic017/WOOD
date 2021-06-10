import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from numpy import sqrt
from numpy import argmax
import colorama as col
from collections import Counter
import joblib

import pyttsx3
engine = pyttsx3.init()

import configparser
config = configparser.ConfigParser()

engine.say("Double M L P Classifier")
engine.runAndWait()

def modelize(df,x,_model):
    
    print(col.Fore.BLUE,'Double MLPClassifier pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)
    print('Modèle :',_model)
    ### Isolation de la partie out of sample ####
    
    print('SIGNAL BUY')
    #print(df.head())
    
    Xb = df.copy()
    Xb['Signal'] = np.where(Xb.Signal<=0,0,1)
    #Xb = Xb.set_index(Xb.Date, drop =True)
    yb = Xb['Signal']
    Xb = Xb[['Momentum','ATR','Bolup','Bolmid','Bollow']]

    # summarize class distribution
    counterb = Counter(yb)
    
    print('Signaux achat',counterb)
    #print(counterb[-1],counterb[1])

    if counterb[0] > 2 and counterb[1] > 2:
    
        # split into train/test sets with same class ratio
        trainXb, testXb, trainyb, testyb = train_test_split(Xb, yb, test_size=0.7, stratify=yb)
        # define model
        model = MLPClassifier(solver='lbfgs', activation = 'relu', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1) # lbfgs
        # fit model
        model.fit(trainXb, trainyb)
        # predict on test set
        yhatb = model.predict(testXb)
        # evaluate predictions
        accub = round(accuracy_score(testyb, yhatb) * 100,2)
        precb = round(precision_score(testyb, yhatb) * 100,2) # labels=[-1,1], average='micro' 'weighted' 'macro' 'None'
        recallb = round(recall_score(testyb, yhatb) * 100,2) # labels=[-1,1], average='micro'  'weighted' 'macro' 'None'
        f1b = round(f1_score(testyb, yhatb) * 100,2) # labels=[-1,1], average='micro'   'weighted' 'macro''None'

        print(col.Fore.BLUE,'Achat pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)
        if accub > 69 and precb > 69 :
            print(col.Fore.GREEN)
        elif accub < 51 or precb < 51 :
            print(col.Fore.RED)
        else:
            print(col.Fore.YELLOW)

        print('Achat - Accuracy :' ,accub,'%')
        print('Achat - Precision :',precb,'%')
        print('Achat - Recall :', recallb,'%')
        print('Achat - F-measure: :' ,f1b,'%',col.Style.RESET_ALL)
        print('\n')
        
        print(classification_report(testyb, yhatb))
        conf_matrix = pd.DataFrame(index = ['vrais_réels','Faux_réels'])
        conf_matrix['Vrais_estimés'] = ['Vrais_positifs','Faux_positifs']
        conf_matrix['Faux_estimés'] = ['Faux_négatif','Vrais-négatifs']
        print(confusion_matrix(testyb, yhatb))
        print('\n')

        savename = 'JOBLIB/'+_model+'/Save'+x.replace('/','')+'m5_buy.sav'
        
        # some time later...
        
        # load the model from disk
        #loaded_model = joblib.load(filename)
        #result = loaded_model.score(X_test, Y_test)
        #print(result)

        # serialize model to JOBLIB
        joblib.dump(model, savename)
        print(col.Fore.BLUE,"Joblib Model ",savename," dumped to disk",col.Style.RESET_ALL)

    else:
        print(col.Fore.RED,'Not enough populated for',col.Fore.YELLOW,x,col.Style.RESET_ALL)
    
    print('SIGNAL SELL')
    #print(df.head())
    
    Xv = df.copy()
    Xv['Signal'] = np.where(Xv.Signal>=0,0,1)
    
    #Xv = Xv.set_index(Xv.Date, drop =True)
    yv = Xv['Signal']
    Xv = Xv[['Momentum','ATR','Bolup','Bolmid','Bollow']]

    # summarize class distribution
    counterv = Counter(yv)
    
    print('Signaux vente',counterv)
    #print(counterv[-1],counterv[1])

    if counterv[1] > 2 and counterv[0] > 2:
    
        # split into train/test sets with same class ratio
        trainXv, testXv, trainyv, testyv = train_test_split(Xv, yv, test_size=0.7, stratify=yv)
        # define model
        model = MLPClassifier(solver='lbfgs', activation = 'relu', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=770) # lbfgs
        # fit model
        model.fit(trainXv, trainyv)
        # predict on test set
        yhatv = model.predict(testXv)
        # evaluate predictions
        accuv = round(accuracy_score(testyv, yhatv) * 100,2)
        precv = round(precision_score(testyv, yhatv) * 100,2) # labels=[-1,1], average='micro' 'weighted' 'macro' 'None'
        recallv = round(recall_score(testyv, yhatv) * 100,2) # labels=[-1,1], average='micro'  'weighted' 'macro' 'None'
        f1v = round(f1_score(testyv, yhatv) * 100,2) # labels=[-1,1], average='micro'   'weighted' 'macro''None'

        print(col.Fore.BLUE,'Achat pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)
        if accuv > 69 and precv > 69 :
            print(col.Fore.GREEN)
        elif accuv < 51 or precv < 51 :
            print(col.Fore.RED)
        else:
            print(col.Fore.YELLOW)

        print('Vente - Accuracy :' ,accuv,'%')
        print('Vente - Precision :',precv,'%')
        print('Vente - Recall :', recallv,'%')
        print('Vente - F-measure: :' ,f1v,'%',col.Style.RESET_ALL)
        print('\n')
        
        print(classification_report(testyv, yhatv))
        conf_matrix = pd.DataFrame(index = ['vrais_réels','Faux_réels'])
        conf_matrix['Vrais_estimés'] = ['Vrais_positifs','Faux_positifs']
        conf_matrix['Faux_estimés'] = ['Faux_négatif','Vrais-négatifs']
        print(confusion_matrix(testyv, yhatv))
        print('\n')

        savename = 'JOBLIB/'+_model+'/Save'+x.replace('/','')+'m5_sell.sav'
        
        # some time later...
        
        # load the model from disk
        #loaded_model = joblib.load(filename)
        #result = loaded_model.score(X_test, Y_test)
        #print(result)

        # serialize model to JOBLIB
        joblib.dump(model, savename)
        print(col.Fore.BLUE,"Joblib Model ",savename," dumped to disk",col.Style.RESET_ALL)
        
    else:
        print(col.Fore.RED,'Not enough populated for',col.Fore.YELLOW,x,col.Style.RESET_ALL)

    return()
