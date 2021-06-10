__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'

####################################################################
####################################################################
####### RECUPERATION DONNEES ET PREPARATION DES DATA FX ############
####################################################################
####################################################################

import pandas as pd
import numpy as np
import colorama as col
import pyttsx3
engine = pyttsx3.init()

def stochastic(df):
    """[Determine the stochastic strategy based on _window=5. Enter dataframe with column slow_K5 and slow_D5. Return dataframe with column Signal.]

    Args:
        df ([dataframe]): [Must been already computed with slow_D5 and slow_K5 column]
    """    
    ##### CONDITIONS LONG
    _condition_1 = (df.slow_K5 < 20) & (df.slow_K5.shift(1) < df.slow_D5.shift(1)) & (df.slow_K5 > df.slow_D5)

    ##### CONDITIONS SHORT
    _condition_1_bar = (df.slow_K5 > 80) & (df.slow_K5.shift(1) > df.slow_D5.shift(1)) & (df.slow_K5 < df.slow_D5)

    ##### 1 condition
    df['Signal'] = np.where(_condition_1,1,np.where(_condition_1_bar,-1,0))

    return(df) 

