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



path  = "/home/loann/Travail/Quantums/Travaux/Données/Finnhub/Stocks/"
path += "GOOG_112019-102020_M1.csv"

price = PRICE("GOOG2019-2020") 


price.setColumnsTitle(askOpen    ="askopen", 
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

price.read(path)
price.setBaseTimeframe(timeframe = dt.timedelta(minutes = 1))
price.fillMissingData()
price.shiftMarketTime(timeshift = 2)

price.dataTimeZone   = 2 
price.marketTimeZone = -6 

price.marketOpeningHour = "09:30"
price.marketClosingHour = "17:30"
price.marketLunch = "12:30-13:30"

price.setMarketState()









