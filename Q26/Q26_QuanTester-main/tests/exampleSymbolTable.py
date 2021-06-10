#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"..")
sys.path.append(filename)

import numpy as np 
import pandas as pd 
import datetime as dt 
import matplotlib.pyplot as plt 
import pprint

from quanTest.symbol    import SYMBOL
from quanTest.symbol    import SYMBOL_TABLE 

# We initialize some assets 
eurusd = SYMBOL(symbolName              = "EUR.USD",
                contractSize            = 100000, 
                marginCurrency          = "USD", # Can be any existing currency 
                profitCalculationMethod = "Forex", # "CFD", "Forex", "Stock", "CFD-Index"
                marginRequestMethod     = "Forex", # "CFD", "Forex", "Stock", "CFD-Index"
                marginPercentage        = 100, 
                execution               = "Market", 
                minimalVolume           = 0.01, 
                maximalVolume           = 100.0, 
                volumeStep              = 0.01, 
                precision               = 5,        # Price precision (3 means 1 point = 0.001)
                exchangeType            = "Point", # "Point", "Percentage"
                exchangeLong            = 6.88, 
                exchangeShort           = 0.63)

btcusd = SYMBOL(symbolName              = "BTC.USD",
                contractSize            = 1, 
                marginCurrency          = "USD", # Can be any existing currency 
                profitCalculationMethod = "CFD", # "CFD", "Forex", "Stock", "CFD-Index"
                marginRequestMethod     = "CFD", # "CFD", "Forex", "Stock", "CFD-Index"
                marginPercentage        = 100, 
                execution               = "Market", 
                minimalVolume           = 0.10, 
                maximalVolume           = 5.00, 
                volumeStep              = 0.10, 
                precision               = 2,        # Price precision (3 means 1 point = 0.001)
                exchangeType            = "Percentage", # "Point", "Percentage"
                exchangeLong            = -30, 
                exchangeShort           = 10)


table = SYMBOL_TABLE([eurusd, btcusd])



print (table.symbol("EUR.USD").__dict__)

table.symbol("EUR.USD").contractSize = 75 

print (table.symbol("EUR.USD").__dict__)












