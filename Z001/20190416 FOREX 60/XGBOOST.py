___Author___='LumberJack Jyss'
print('LumberJack BRUTAL AtidotCom XGBOOST\nLumberJack Jyss (c)')
print('Importation des librairies...')
import statistics
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from pandas.plotting import register_matplotlib_converters
import pandas_datareader as web
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,roc_curve
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

print('Librairies importées')
df = pd.read_csv('dataset_ready_to_use.csv')
df = df.dropna()
df.set_index('Local time', inplace=True)
df = df.iloc[:,1:]
df.head()
delta = df.shape[0]
bloc1 = round(delta*0.8)
bloc2 = delta - bloc1
print("Période d'étude : ",delta,'périodes')
print('Sur un découpage 80% - 20% de la période : ')
print('Bloc 1 : ',bloc1,' périodes \nBloc 2 :',bloc2,' périodes')
print(df['long'].sum(),df['short'].sum())
Xtrain_long = df.iloc[:bloc1,:-1]
Xtrain_long = Xtrain_long.drop(['short'],axis=1)
Xtest_long = df.iloc[bloc1:,:-1]
Xtest_long = Xtest_long.drop(['short'],axis=1)
ytrain_long = df.iloc[:bloc1,-1]
ytest_long = df.iloc[bloc1:,-1]

Xtrain_short = df.iloc[:bloc1,:-2]
Xtest_short = df.iloc[bloc1:,-2]
ytrain_short = df.iloc[:bloc1,:-2]
ytest_short = df.iloc[bloc1:,-2]
print('Split effectué')
model_long = XGBClassifier(silent=False,objective='binary:logistic',n_estimators=200)
eval_set_long=[(Xtrain_long, ytrain_long), (Xtest_long, ytest_long)]
model_long.fit(Xtrain_long, ytrain_long, eval_metric=['auc','error','logloss'], eval_set=eval_set_long, verbose=1)
predictions_long = model_long.predict(Xtest_long)
accuracy_long = accuracy_score(ytest_long, predictions_long)
precision_long=precision_score(ytest_long, predictions_long) # vrais_positifs/(vrais_positifs+faux_positifs)
# La précision permet de mesurer la capacité du modèle à refuser résultats non-pertinents.
recall_long=recall_score(ytest_long, predictions_long) # (vrai_positifs/(vrais_positifs+faux_négatifs))
roc_long=roc_auc_score(ytest_long,predictions_long)
print('Accuracy: %.2f%%' % (accuracy_long * 100.0))
print("Precision: %.2f%% " % (precision_long *100))
print("Recall: %.2f%% " % (recall_long * 100))
print("AUC: %.2f%% " % (roc_long *100))
# get probabilities for positive class
prediction_long = model_long.predict_proba(Xtest_long)
roc_long2 = roc_auc_score(ytest_long, prediction_long[:,1])
print("AUC_proba: %.2f%% " % (roc_long2 * 100))
print(classification_report(ytest_long, predictions_long))
print(confusion_matrix(ytest_long, predictions_long))
plt.figure(figsize=(16,6))
plt.plot(predictions_long,)
backtest = pd.DataFrame()
backtest['Close_eurusd'] = df['Close_eurusd']
backtest['long'] = df['long']
backtest['short'] = df['short']
backtest.head()
backtest = backtest.iloc[bloc1:,:]
