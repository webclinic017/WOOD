
___author___ = 'LumberJack Jyss'

#### LOAD BIBAL ####
import pandas as pd
import numpy as np
import re
import csv
import datetime

#### OUVERTURE DES FICHIERS ####
df=pd.read_csv ('statistiques.csv', encoding='ISO-8859-1', delimiter=';') #, low_memory=False)

#df['column']=pd.to_numeric(df['column'])
#df['column'].replace('[A-Za-z]','',inplace=True,regex=True)
#df['column'].replace('  ','0',inplace=True)
#df['column']=pd.to_numeric(df['Size'])

#df['Installs'].replace(',','',inplace=True,regex=True)
#df['Installs'].replace('\+','',inplace=True,regex=True)
#df['Installs']=pd.to_numeric(df['Installs'])
#df['Price'].replace('\$','',inplace=True,regex=True)
#df['Price']=pd.to_numeric(df['Price'])
#df['Last Updated']=pd.to_datetime(df['Last Updated'])#, format='%Y-%m-%d')

#df['Column']=str(df['Column'])
#df['Column']=str(df['Column'])

print('head\n',df.head(5),'\n')
print('Info\n',df.info(),'\n')
print('Describe\n',df.describe(),'\n')

for Column in df.head(0):
	print('\nLes 3 val min et max pour ',Column,' :\n')
	df_temp=df.sort_values(by=Column)
	print(df_temp[Column].head(3))
	print('\n')
	df_temp=df.sort_values(by=Column,ascending=False)
	print(df_temp[Column].head(3))
	print('\n')

print('Count\n',df.count(),'\n')
# ==> 

print('Types\n',df.dtypes,'\n')
# ==> 

# Verifiaction de l'unicit� des ColumnID :

print('Classement des doublons : \n')
#try:
#	df_temp=pd.concat(g for _, g in df.groupby("columnID") if len(g) > 1)
#	print(df_temp)
#	df_temp.to_csv('doublons.csv', sep=',')
#	print('Longueur du dataframe doublon :',len(df_temp))

#except:
#	print('Aucun doublon\n')
#	print('Nombre de valeurs uniques dans App :',df['App'].nunique())

# ==> 

print('Nombre de NaN au total dans le dataframe : ',df.isnull().sum().sum())

# On v�rifie s'il n'y a pas de date illogiques.
#print(df.sort_values(by='column').head(5))
#print(df.sort_values(by='column',ascending=False).head(5))
# ==> 



#df_cleaned.to_csv('Cleaned_googleplaystore.csv')

