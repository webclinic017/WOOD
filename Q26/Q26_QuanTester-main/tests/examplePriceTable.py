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

from quanTest.data import PRICE 
from quanTest.data import PRICE_TABLE



path  = "/home/loann/Travail/Quantums/Travaux/Données/Finnhub/Stocks/"
path += "GOOG_112019-102020_M1.csv"

price1 = PRICE("GOOG2019-2020") 


price1.setColumnsTitle(askOpen    ="askopen", 
                      askHigh    ="askhigh",
                      askLow     ="asklow",
                      askClose   ="askclose", 
                      bidOpen    ="askopen",
                      bidHigh    ="askhigh",
                      bidLow     ="asklow",
                      bidClose   ="askclose",
                      date       ="date",
                      dateFormat = "%Y-%m-%d %H:%M:%S", 
                      volume     ="volume")

price1.read(path)
price1.setBaseTimeframe(timeframe = dt.timedelta(minutes = 1))
price1.fillMissingData()
price1.shiftMarketTime(timeshift = 2)

price1.dataTimeZone   = 2 
price1.marketTimeZone = -6 

price1.marketOpeningHour = "09:30"
price1.marketClosingHour = "17:30"
price1.marketLunch = "12:30-13:30"

price1.setMarketState()


path  = "/home/loann/Travail/Quantums/Travaux/Données/Finnhub/Stocks/"
path += "AAPL_112019-102020_M1.csv"

price2 = PRICE("AAPL2019-2020") 


price2.setColumnsTitle(askOpen    ="askopen", 
                      askHigh    ="askhigh",
                      askLow     ="asklow",
                      askClose   ="askclose", 
                      bidOpen    ="askopen",
                      bidHigh    ="askhigh",
                      bidLow     ="asklow",
                      bidClose   ="askclose",
                      date       ="date",
                      dateFormat = "%Y-%m-%d %H:%M:%S", 
                      volume     ="volume")

price2.read(path)
price2.setBaseTimeframe(timeframe = dt.timedelta(minutes = 1))
price2.fillMissingData()
price2.shiftMarketTime(timeshift = 0)

price2.dataTimeZone   = 0
price2.marketTimeZone = 0

price2.marketOpeningHour = "09:30"
price2.marketClosingHour = "17:30"
price2.marketLunch = "12:30-13:30"

price2.setMarketState()




table = PRICE_TABLE([price1, price2]) 
table.synchronize()



df = table.array("AAPL2019-2020", dt.datetime(2019, 11, 5, 12, 28), dt.datetime(2019, 11, 5, 13, 32), format = "dataframe")
print(df)

















