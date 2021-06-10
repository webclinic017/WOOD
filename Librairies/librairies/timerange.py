
import pandas as pd
from tqdm import tqdm

__author__ = 'LumberJack'

##### DAG26 5781 (c) #####
#####
##### df : dataframe with OHLC => Need index in pd.datetime()
#####


def timerange1D(df):
    df['TimeRange'] = [df.index[i].strftime(format='%Y-%m-%d') for i in tqdm(range(len(df)))]
    return(df)