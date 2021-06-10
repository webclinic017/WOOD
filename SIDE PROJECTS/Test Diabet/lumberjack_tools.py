__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'
__version__ = 'v0.1'

####################################################################
####################################################################
############################### TOOLS ##############################
####################################################################
####################################################################

'''
tools for datasciences using tensorflow
'''

# Math Stuff
import numpy as np
import pandas as pd
import scipy.stats as stat

# Proceesing librairies
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # Disable eager execution
import keras
import sklearn
from natsort import natsorted
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.layers import Input, Dense, Activation,Dropout, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Recall, Precision, Metric
from keras.utils import to_categorical
import imblearn

# Side Stuff
import joblib
import warnings
import colorama as col
import pyttsx3
from tensorflow.python.ops.gen_dataset_ops import model_dataset
engine = pyttsx3.init()
from tqdm import tqdm, tqdm_notebook, tqdm_pandas

# Random Seed
seed_value = 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)


# Technical analysis
from finta import TA
import talib

# Plotting stuff
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# Time Stuff
import time
import datetime as dt

print('Imblearn version ',imblearn.__version__)
print('Pandas version ',pd.__version__)
print('Numpy version ',np.__version__)
print('Tensorflow version ',tf.__version__)
print('Joblib version ',joblib.__version__)
print('TaLib version ',talib.__version__)
print('FinTA version ',TA.__version__)

warnings.filterwarnings('ignore')

# Undersampling pour 2 classes
def undersample2(df,target,_coef=1):
    
    print('Avant Resampling :')
    print('Classe 0',df[df[target]==0].shape[0])
    print('Classe 1',df[df[target]==1].shape[0])

    # Class count
    count_class_0, count_class_1 = df[target].value_counts()
    # Divide by class
    df_class_0 = df[df[target]== 0]
    df_class_1 = df[df[target] == 1]

    df_class_0_under = df_class_0.sample(int((count_class_1)*_coef))
    df = pd.concat([df_class_0_under, df_class_1], axis=0)
    df = df.sort_index()
    print('Random under-sampling:')
    print(df[target].value_counts())

    # Classify and report the results
    print('\nAprès resample:')
    print('Classe 0',df[df[target]==0].shape[0])
    print('Classe 1',df[df[target]==1].shape[0])
    return df

# Oversampling pour deux clases
def oversample2(df,target):
    from imblearn.over_sampling import SMOTE
    print('Avant Resampling :')
    print('Classe 0',df[df[target]==0].shape[0])
    print('Classe 1',df[df[target]==1].shape[0])
    
    
    # Class count
    count_class_0, count_class_1 = df[target].value_counts()
    # Divide by class
    df_class_0 = df[df[target]== 0]
    df_class_1 = df[df[target] == 1]

    oversample = SMOTE()
    df, y = oversample.fit_resample(df, df[target])
    df = df.sort_index()
    print('Random under-sampling:')
    print(df[target].value_counts())

    # Classify and report the results
    print('\nAprès resample:')
    print('Classe 0',df[df[target]==0].shape[0])
    print('Classe 1',df[df[target]==1].shape[0])
    return df

# Création de colonnes basées sur la différence avec la ligne précédente
def feature_diff(df,COLUMNS=[]):
    # Diff
    COLUMNS = ['Valid','TRACKER','Signal']
    for _column in df.columns:
        if _column not in COLUMNS:
            df['D_'+_column] = df[_column] - df[_column].shift(1)
            df['D2_'+_column] = df[_column] - df[_column].shift(2)
            df['D3_'+_column] = df[_column] - df[_column].shift(3)
            df['D4_'+_column] = df[_column] - df[_column].shift(4)
            df['D5_'+_column] = df[_column] - df[_column].shift(5)
    df = df.dropna()
    return df

# Différents calculs pour featurer les colonnes exitentes
def feature_transformation(df,COLUMNS=[]):
    for _column in df.columns:
        if _column not in COLUMNS:
            df['Log_'+_column] = np.log(df[_column])
            df['Sqrt_'+_column] = np.sqrt(df[_column])
            df['Recip_'+_column] = 1 / (df[_column])
            #df['Sq_'+_column] = df[_column] ** df[_column]
            #df['Box_'+_column] = stat.boxcox(df[_column])
            df['Cos_'+_column] = np.cos(df[_column])
            df['Sin_'+_column] = np.sin(df[_column])
            df['Tan_'+_column] = np.tan(df[_column])
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    df = df.T.dropna().T
    return df

# Scaling de chaque colonne en gérant les exceptions de colonne
def feature_scaling(df,scaler = MinMaxScaler(),EXCEPTION=[]):
    scaler = MinMaxScaler()
    df.sort_index(inplace=True)
    
    for _column in df.columns.unique():
        if _column not in EXCEPTION:
            df[_column] = scaler.fit_transform(df[_column].values.reshape(-1, 1))

    df = df.dropna()
    df = df.reindex(natsorted(df.columns), axis=1)
    return df

# Feature Selection en utilisant l'univariate
def univariate_feature_selection(X,Z,y,_k):
    from sklearn.feature_selection import SelectKBest, f_classif
    # Keep _k features
    selector = SelectKBest(f_classif, k=_k)

    X_new = selector.fit_transform(X, y)

    # Get back the features we've kept, zero out all other features
    selected_features = pd.DataFrame(selector.inverse_transform(X_new), index=X.index, columns=X.columns)

    # Dropped columns have values of all 0s, so var is 0, drop them
    selected_columns = selected_features.columns[selected_features.var() != 0]

    X = X[selected_columns]
    Z = Z[selected_columns]
    return X,Z

# Feature Selection en utilisant le l1
def l1_feature_selection(X,Z,y,_C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    
    # Set the regularization parameter C=1
    logistic = LogisticRegression(C=_C, penalty="l1", solver='liblinear', random_state=42).fit(X, y)
    model = SelectFromModel(logistic, prefit=True)

    X_new = model.transform(X)

    # Get back the kept features as a DataFrame with dropped columns as all 0s
    selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                    index=X.index,
                                    columns=X.columns)

    # Dropped columns have values of all 0s, keep other columns 
    selected_columns = selected_features.columns[selected_features.var() != 0]

    X = X[selected_columns]
    Z = Z[selected_columns]
    return X,Z

# Créatio d'une pondération si necessaire
def affect_weight(y_train):
    bool_train_labels = y_train != 0
    _pos, _neg = y_train.value_counts()
    _total = _pos + _neg
    initial_bias = np.log([_pos/_neg])
    #initial_bias = _pos/(_pos+_neg)
    print('Initial bias',initial_bias)
    print('')

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / _neg) * (_total / 2.0) 
    weight_for_1 = (1 / _pos) * (_total / 2.0)

    # class_weight = {0: weight_for_0, 1: weight_for_1}
    class_weight = [weight_for_0, weight_for_1]
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight

# Création du modele de classification TensorFlow 2
def make_model(X_train,_first_a = 'relu',_a='relu',_last_a='sigmoid',BATCH_SIZE = 32):
    
    model  = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=63, activation=_first_a,input_shape=(X_train.shape[1],)))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=128, activation=_a))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=256, activation=_a))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=128, activation=_a))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(units=63, activation=_a))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=32, activation=_a))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=16, activation=_a))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=8, activation=_a))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=4, activation=_a))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=1, activation=_last_a))

    print(model.summary())

    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
            ]

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=20,
    mode='max',
    restore_best_weights=True)

    return model

######
def compile_model(model, METRICS =['binary_accuracy']):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True, name='Adam'),
        metrics=METRICS, 
        loss='binary_crossentropy'
        #,metrics=['mean_squared_error',
        # tf.keras.metrics.Precision(),
        # tf.keras.metrics.Recall()])
                )
    return model

#######
def evaluate_batch(model, X_train, y_train,_borne,_step):
    _best_result = 99999999999999
    _best_batch = 0
    for i in tqdm(range(10,_borne,_step)):
        results = model.evaluate(X_train, y_train, batch_size=i, verbose=0)
        if results[0] < _best_result:
            _best_result = results[0]
            _best_batch = i
    print("Meilleur Loss: {:0.4f}".format(_best_result))
    print("Meilleur Batch: {:0.4f}".format(_best_batch))
    return _best_batch

#######
def fit_model(X_train,y_train,X_test,y_test,model,epochs=20,BATCH_SIZE=32,callbacks=None,class_weight=None):
    validation_data=(X_test, y_test)
    history = model.fit(X_train, y_train,epochs=epochs,batch_size=BATCH_SIZE,callbacks=callbacks,validation_data=validation_data,class_weight=class_weight)   
    mpl.rcParams['figure.figsize'] = (24, 6)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_metrics(history):
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric],
                    color=colors[1], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.ylim([0, plt.ylim()[1]])
            if metric == 'auc':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])
    plot_metrics(history)

    
    train_predictions_weighted = model.predict(X_train, batch_size=BATCH_SIZE)
    test_predictions_weighted = model.predict(X_test, batch_size=BATCH_SIZE)

    results = model.evaluate(X_test, y_test,
                                            batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(model.metrics_names, results):
        print(name, ': ', value)
        print()

    def plot_cm(labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
        print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
        print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
        print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
        print('Total Fraudulent Transactions: ', np.sum(cm[1]))


    def plot_roc(name, labels, predictions, **kwargs):
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

        plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.xlim([-0.5,20])
        plt.ylim([80,100.5])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        plot_roc("Train", y_train, train_predictions_weighted, color=colors[0])
        plot_roc("Test", y_test, test_predictions_weighted, color=colors[1], linestyle='--')


        plt.legend(loc='lower right')

        print('Extrèmes des autres',min(model.predict(X_test))[0],max(model.predict(X_test))[0])
        print()
        return history

# On remet à 0 ou 1 les prdict.proba
def norm_target(model,X_test,y_test):
    _edge = (min(model.predict(X_test))[0]+max(model.predict(X_test))[0])/2
    yhat = pd.DataFrame()
    PREDICTAT = []
    PREDICT = model.predict(X_test)
    for i in tqdm(range(len(PREDICT))):
        if PREDICT[i] <= _edge:
            PREDICTAT.append(0)
        else:
            PREDICTAT.append(1)
    yhat['Predict'] = PREDICTAT

    TEST = []
    for j in tqdm(range(len(y_test))):
        if y_test[j] == 0.:
            TEST.append(0)
        else:
            TEST.append(1)
    y_test = TEST
    return(y_test,yhat)

#####
def get_results(y_test,yhat):
    #y_test = y_test.reshape(-1,)
    yhat = yhat.Predict

    accu = round(accuracy_score(y_test, yhat) * 100,2)
    prec = round(precision_score(y_test, yhat, labels=[0,1],pos_label=1) * 100,2)
    recall = round(recall_score(y_test, yhat) * 100,2)
    f1 = round(f1_score(y_test, yhat) * 100,2)


    print('Signaux - Accuracy :' ,accu,'%')
    print('Signaux - Precision :',prec,'%')
    print('Signaux - Recall :', recall,'%')
    print('Achat - F-measure: :' ,f1,'%')
    print('\n')
    print(classification_report(y_test, yhat))
    conf_matrix = pd.DataFrame(columns=['Positifs','Négatifs'])
    _tn, _fp, _fn, _tp = confusion_matrix(y_test, yhat).ravel()    #_model.classes_)

    conf_matrix.loc['Positifs'] = [_tp,_fn]
    conf_matrix.loc['Négatifs'] = [_fp,_tn]

    print(conf_matrix)
    print()
    #print(col.Fore.BLUE,'Signaux pour',col.Fore.YELLOW,x,col.Style.RESET_ALL)

    _prec = round((_tp/(_tp+_fp))*100,2)
    _rec = round((_tp/(_tp+_fn))*100,2)
    _f1 = round((2 * _prec * _rec) / (_prec + _rec),2)

    print('Vrais signaux positifs trouvés    : ',_tp)
    print('Vrais signaux positifs non trouvé :',_fn)
    print('Total des signaux posistifs :',_tp+_fn)

    if _prec > 69 :
        print(col.Fore.GREEN,'Précision :',_prec,'%',col.Style.RESET_ALL)
    elif _prec < 51 :
        print(col.Fore.RED,'Précision :',_prec,'%',col.Style.RESET_ALL)
    else:
        print(col.Fore.YELLOW,'Precision : ',_prec,'%')
    if _rec > 69  :
        print(col.Fore.GREEN,'Recall :',_rec,'%',col.Style.RESET_ALL)
    elif _rec < 51  :
        print(col.Fore.RED,'Recall :',_rec,'%',col.Style.RESET_ALL)
    else:
        print(col.Fore.YELLOW,'Recall',_rec,'%',col.Style.RESET_ALL)
    if _f1 > 69  :
        print(col.Fore.GREEN,'F-Score :',_f1,'%',col.Style.RESET_ALL)
    elif _f1 < 51  :
        print(col.Fore.RED,'F-Score :',_f1,'%',col.Style.RESET_ALL)
    else:
        print(col.Fore.YELLOW,'F-Score',_f1,'%',col.Style.RESET_ALL)

    print('\n')
    print('Vrais signaux négatifs trouvés    : ',_tn)
    print('Vrais signaux négatifs non trouvé :',_fp)
    print('Total des signaux négatifs :',_tn+_fp)

    _prec = round((_tn/(_tn+_fn))*100,2)
    _rec = round((_tn/(_tn+_fp))*100,2)
    _f1 = round((2 * _prec * _rec) / (_prec + _rec),2)

    if _prec > 69 :
        print(col.Fore.GREEN,'Précision :',_prec,'%',col.Style.RESET_ALL)
    elif _prec < 51 :
        print(col.Fore.RED,'Précision :',_prec,'%',col.Style.RESET_ALL)
    else:
        print(col.Fore.YELLOW,'Precision : ',_prec,'%')
    if _rec > 69  :
        print(col.Fore.GREEN,'Recall :',_rec,'%',col.Style.RESET_ALL)
    elif _rec < 51  :
        print(col.Fore.RED,'Recall :',_rec,'%',col.Style.RESET_ALL)
    else:
        print(col.Fore.YELLOW,'Recall',_rec,'%',col.Style.RESET_ALL)
    if _f1 > 69  :
        print(col.Fore.GREEN,'F-Score :',_f1,'%',col.Style.RESET_ALL)
    elif _f1 < 51  :
        print(col.Fore.RED,'F-Score :',_f1,'%',col.Style.RESET_ALL)
    else:
        print(col.Fore.YELLOW,'F-Score',_f1,'%',col.Style.RESET_ALL)

    fpr , tpr , thresholds = roc_curve ( y_test, yhat)
    auc_score = roc_auc_score(y_test, yhat) * 100
    
    def plot_roc_curve(fpr,tpr):
        plt.figure(figsize=(24,6))
        plt.title("Aera Under Curve : "+str(round(auc_score,2))+' %')
        plt.plot(fpr,tpr,c='orange')
        plt.axis([0,1,0,1]) 
        plt.plot(y_test,y_test)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        plt.show()    
    
    plot_roc_curve (fpr,tpr) 



if __name__ == '__main__':
    pass
