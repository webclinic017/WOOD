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
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from keras.models import Sequential
from keras.layers import Dense
from numpy import sqrt
from numpy import argmax
import colorama as col
from collections import Counter
import tensorflow as tf
import joblib

import pyttsx3
engine = pyttsx3.init()

import configparser
config = configparser.ConfigParser()

tf.keras.backend.set_floatx('float64')

engine.say("processing M L P, Multi Layer Perceptron.")
engine.runAndWait()

config.read('config.ini')
_period1 = config.get('TIMEFRAME','_period1')
_path4 = config.get('PATH','_path4')
_path5 = config.get('PATH','_path5')
_path = _path4
config['PATH']['_path'] = _path
with open('config.ini', 'w') as configfile: # save
    config.write(configfile)

def modelize(x):
    
    globals()['df1_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period1+'.csv')

    df = globals()['df1_%s' %x.replace('/','')].copy()

    print(col.Fore.BLUE,'Multi Layer Perceptron pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)

    try:
        df = df.drop(['Unnamed: 0'],axis=1)
    except:
        pass
    try:
        df = df.drop(['Unnamed: 0.1'],axis=1)
    except:
        pass
    try:
        df = df.drop(['Unnamed: 0.2'],axis=1)
    except:
        pass
    ### Isolation de la partie out of sample ####
    
    df = df.iloc[:-int(len(df) * 0.2),:]
    
    Xb = df[df.Signal == 1]
    Xb = Xb.set_index(Xb.Date, drop =True)
    yb = Xb['BUY']
    Xb = Xb.drop(['CloseBid','HigMax','LowMin','SELL','Date','Signal','BUY'],axis=1)
    
    Xv = df[df.Signal == -1]
    Xv = Xv.set_index(Xv.Date, drop =True)
    yv = Xv['SELL']
    Xv = Xv.drop(['CloseBid','HigMax','LowMin','BUY','Date','Signal','SELL'],axis=1)

    counterb = Counter(yb)
    counterv = Counter(yv)
    print('Signaux achat ',counterb)
    print('Signaux vente ',counterv)


    if counterb[0] > 2 and counterb[1] > 2 and counterv[0] > 2 and counterv[1] > 2:

        '''oversample = SMOTEENN(sampling_strategy=0.7)

        Xb_over, yb_over = oversample.fit_resample(Xb, yb)
        Xv_over, yv_over = oversample.fit_resample(Xv, yv)

        print('Signaux achat après oversampling',Counter(yb_over))
        print('Signaux vente après oversampling',Counter(yv_over))'''
    
        # split into train/test sets with same class ratio
        '''trainXb, testXb, trainyb, testyb = train_test_split(Xb_over, yb_over, test_size=0.7, stratify=yb_over)
        trainXv, testXv, trainyv, testyv = train_test_split(Xv_over, yv_over, test_size=0.7, stratify=yv_over)'''

        trainXb, testXb, trainyb, testyb = train_test_split(Xb, yb, test_size=0.7, stratify=yb)
        trainXv, testXv, trainyv, testyv = train_test_split(Xv, yv, test_size=0.7, stratify=yv)

        seed = 770
        np.random.seed(seed)

        #trainyb = trainyb.reshape(trainyb.shape[0],)
        #trainyv = trainyv.reshape(trainyv.shape[0],)

        #trainXb = trainXb.reshape(trainXb.shape[0],trainXb.shape[1])
        #trainXv = trainXv.reshape(trainXv.shape[0],trainXv.shape[1])

        modelb = Sequential()
        # Add an input layer 
        modelb.add(Dense(50, activation='sigmoid'))
        # Add one hidden layer 
        modelb.add(Dense(23, activation='sigmoid'))
        # Add an output layer 
        modelb.add(Dense(1, activation='sigmoid'))

        print(col.Fore.BLUE,'Processing Deep Learning',col.Style.RESET_ALL)
        modelb.compile(loss='binary_crossentropy',
                    optimizer='adam', #rmsprop # adam
                    metrics=['accuracy','mse'])
        
                
        historyb = modelb.fit(trainXb, trainyb,epochs=280, batch_size=8, verbose=0)

        yhatb = modelb.predict(testXb)
        
        # evaluate predictions

       
        yhatb = modelb.predict_classes(testXb) * 100
       
        # La précision permet de mesurer la capacité du modèle à refuser résultats non-pertinents : vrais_positifs/(vrais_positifs+faux_positifs)
        precb = precision_score(testyb, yhatb,average='macro') * 100 # modelb.predict(testXb)[-1] # weighted
        
        accub = accuracy_score(testyb, yhatb) * 100
        # Recall : (vrai_positifs/(vrais_positifs+faux_négatifs))
        recallb = recall_score(testyb, yhatb,average='macro') * 100 # weighted

        f1b = f1_score(testyb, yhatb,average='macro') * 100 # weighted
    
        print(col.Fore.BLUE,'Achat pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)
        if accub > 69 and precb > 69 :
            print(col.Fore.GREEN)
        elif accub < 51 or precb < 51 :
            print(col.Fore.RED)
        else:
            print(col.Fore.YELLOW)

        #print('Achat - Evaluation :' ,evalub,'%')
        print('Achat - Accuracy :' ,accub,'%')
        print('Achat - Precision :',precb,'%')
        print('Achat - F1_Score :',f1b,'%')
        print('Achat - Recall :', recallb,'%',col.Style.RESET_ALL)
        #print('Achat - F-measure: :' ,f1b,'%',col.Style.RESET_ALL)
        print('\n')
        
        #print(classification_report(testyb, yhatb))
        #conf_matrix = pd.DataFrame(index = ['vrais_réels','Faux_réels'])
        #conf_matrix['Vrais_estimés'] = ['Vrais_positifs','Faux_positifs']
        #conf_matrix['Faux_estimés'] = ['Faux_négatif','Vrais-négatifs']
        #print(confusion_matrix(testyb, yhatb))
        print('\n')

        savename = 'JOBLIB/mlp/Save'+x.replace('/','')+'m5_buy'

        # serialize model to YAML
        modelb_yaml = modelb.to_yaml()
        
        # serialize weights to HDF5
        modelb.save_weights(savename+".h5")
        print(savename,".h5 saved to disk")
    
        modelv = Sequential()
        # Add an input layer 
        modelv.add(Dense(50, activation='relu'))
        # Add one hidden layer 
        modelv.add(Dense(23, activation='relu'))
        # Add an output layer 
        modelv.add(Dense(1, activation='sigmoid'))

        print(col.Fore.BLUE,'Processing Deep Learning',col.Style.RESET_ALL)
        modelv.compile(loss='binary_crossentropy',
                    optimizer='adam', #rmsprop # adam
                    metrics=['accuracy','mse'])
        
                
        historyv = modelb.fit(trainXv, trainyv,epochs=280, batch_size=8, verbose=0)

        yhatv = modelv.predict(testXv)
        
        # evaluate predictions

       
        yhatv = modelv.predict_classes(testXv) * 100
       
        # La précision permet de mesurer la capacité du modèle à refuser résultats non-pertinents : vrais_positifs/(vrais_positifs+faux_positifs)
        precv = precision_score(testyv, yhatv,average='macro') * 100 # modelb.predict(testXb)[-1] # weighted
        
        accuv = accuracy_score(testyv, yhatv) * 100
        # Recall : (vrai_positifs/(vrais_positifs+faux_négatifs))
        recallv = recall_score(testyv, yhatv,average='macro') * 100 # weighted

        f1v = f1_score(testyv, yhatv,average='macro') * 100 # weighted
    
        print(col.Fore.BLUE,'Vente pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)
        if accuv > 69 and precv > 69 :
            print(col.Fore.GREEN)
        elif accuv < 51 or precv < 51 :
            print(col.Fore.RED)
        else:
            print(col.Fore.YELLOW)

        #print('Achat - Evaluation :' ,evalub,'%')
        print('Vente - Accuracy :' ,accuv,'%')
        print('Vente - Precision :',precv,'%')
        print('Vente - F1_Score :',f1v,'%')
        print('Vente - Recall :', recallv,'%',col.Style.RESET_ALL)
        #print('Achat - F-measure: :' ,f1b,'%',col.Style.RESET_ALL)
        print('\n')
        
        #print(classification_report(testyb, yhatb))
        #conf_matrix = pd.DataFrame(index = ['vrais_réels','Faux_réels'])
        #conf_matrix['Vrais_estimés'] = ['Vrais_positifs','Faux_positifs']
        #conf_matrix['Faux_estimés'] = ['Faux_négatif','Vrais-négatifs']
        #print(confusion_matrix(testyb, yhatb))
        print('\n')

        savename = 'JOBLIB/mlp/Save'+x.replace('/','')+'m5_sell'

        # serialize model to YAML
        modelb_yaml = modelb.to_yaml()
        
        # serialize weights to HDF5
        modelb.save_weights(savename+".h5")
        print(savename,".h5 saved to disk")

        print(confusion_matrix(testyv, yhatv))
        print('\n')

        fprv, tprv, thresholdsv = roc_curve(testyv, yhatv)
        # calculate the g-mean for each threshold
        gmeansv = sqrt(tprv * (1-fprv))
        # locate the index of the largest g-mean
        ixv = argmax(gmeansv)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholdsv[ixv], gmeansv[ixv]))
        # plot the roc curve for the model
        plt.ion()
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

        plt.ion()
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

    else:
        print(col.Fore.RED,'Not enough populated for',col.Fore.YELLOW,x,col.Style.RESET_ALL)
    return()