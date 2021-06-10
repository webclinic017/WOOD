import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
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

engine.say("Logistic Regression")
engine.runAndWait()


def modelize(df,x,_model):
    
    print(col.Fore.BLUE,'Regression Logistique pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)
    print(_model)

    ### Isolation de la partie out of sample ####

    #df.iloc[-int(len(df) * 0.2):,:].to_csv(_path5+x.replace('/','')+_period1+'.csv')
    
    df = df.iloc[:-int(len(df) * 0.2),:]
    print('DF (Les 20% de la fin ont été enlevé pour un OOST ultérieur')

    #print(df.head())
    
    Xb = df.copy()
    Xb = Xb.set_index(Xb.Date, drop =True)
    yb = Xb['Signal']
    Xb = Xb[['Return3','ATR']]

    # summarize class distribution
    counterb = Counter(yb)
    
    print('Signaux achat',counterb)
    #print(counterb[-1],counterb[1])

    if counterb[-1] > 2 and counterb[1] > 2:
    
        # split into train/test sets with same class ratio
        trainXb, testXb, trainyb, testyb = train_test_split(Xb, yb, test_size=0.7, stratify=yb)
        # define model
        model = LogisticRegression(solver='liblinear', class_weight='balanced')
        # fit model
        model.fit(trainXb, trainyb)
        # predict on test set
        yhatb = model.predict(testXb)
        # evaluate predictions
        accub = round(accuracy_score(testyb, yhatb) * 100,2)
        precb = round(precision_score(testyb, yhatb) * 100,2)
        recallb = round(recall_score(testyb, yhatb) * 100,2)
        f1b = round(f1_score(testyb, yhatb) * 100,2)

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

        savename = 'JOBLIB/'+_model+'/Save'+x.replace('/','')+'m5.sav'
        
        # some time later...
        
        # load the model from disk
        #loaded_model = joblib.load(filename)
        #result = loaded_model.score(X_test, Y_test)
        #print(result)

        # serialize model to JOBLIB
        joblib.dump(model, savename)
        print(col.Fore.BLUE,"Joblib Model ",savename," dumped to disk",col.Style.RESET_ALL)
        
        precisionb, recallb, thresholdsb = precision_recall_curve(testyb, yhatb)
        # plot the roc curve for the model
        no_skillb= len(testyb[testyb==1]) / len(testyb)

        # convert to f score
        fscoreb = (2 * precisionb * recallb) / (precisionb + recallb)
        # locate the index of the largest f score
        ixb = argmax(fscoreb)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholdsb[ixb], fscoreb[ixb]))

        plt.ion()
        plt.figure(figsize=(12,4))
        plt.suptitle('Recall_Precision Curve la dataframe')
        plt.plot([0,1], [no_skillb,no_skillb], linestyle='--', label='No Skill')
        plt.plot(recallb, precisionb, marker='.', label='Logistic')
        plt.scatter(recallb[ixb], precisionb[ixb], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        # show the plot
        plt.show()

    else:
        print(col.Fore.RED,'Not enough populated for',col.Fore.YELLOW,x,col.Style.RESET_ALL)
    return()
