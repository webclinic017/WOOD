#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 07:16:04 2021

@author: loann
"""

import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"..")
sys.path.append(filename)

import numpy as np 
import pandas as pd 
import datetime as dt 
import matplotlib.pyplot as plt 
import pprint
import copy 

from quanTest.symbol     import SYMBOL
from quanTest.portfolio  import PORTFOLIO 
from quanTest.data import PRICE 
from quanTest.data import PRICE_TABLE
from quanTest.simulation import SIMULATION


""" 
===============================================================================
INITIALIZATION STEP
===============================================================================
"""

path  = "/home/loann/Travail/Quantums/Travaux/Données/Other/HDD/"
path += "AUDCAD_m5_BidAndAsk.csv"

price = PRICE("AUD.CAD") 

price.setColumnsTitle(askOpen        ="OpenAsk", 
                      askHigh        ="HighAsk",
                      askLow         ="LowAsk",
                      askClose       ="CloseAsk", 
                      bidOpen        ="OpenBid",
                      bidHigh        ="HighBid",
                      bidLow         ="LowBid",
                      bidClose       ="CloseBid",
                      dateFormat     ="%m-%d-%Y %H:%M:%S", 
                      volume         ="Total Ticks",
                      splitDaysHours =True, 
                      days           ="Date", 
                      hours          ="Time")

price.read(path)
price.setBaseTimeframe(timeframe = dt.timedelta(minutes = 5))
price.fillMissingData()

price.shiftMarketTime(timeshift = 0)
price.dataTimeZone   = 0
price.marketTimeZone = 0
price.marketOpeningHour = "07:00"
price.marketClosingHour = "24:00"
price.marketLunch = "12:30-13:30"
price.daysOfWeek = [0, 1, 2, 3, 4]

price.setMarketState() 


#price.resampleData("02:00")

"""

# We generate a day time sampler hours list 
dayCandleList = price.timeDaySampler("00:05", "01:00")
print(dayCandleList)



# We create a pandas dataframe from our data and pass date as index 
df = pd.DataFrame({"askOpen" : price.askOpen, 
                   "askHigh" : price.askHigh, 
                   "askLow"  : price.askLow, 
                   "askClose": price.askClose, 
                   "bidOpen" : price.bidOpen, 
                   "bidHigh" : price.bidHigh, 
                   "bidLow"  : price.bidLow, 
                   "bidClose": price.bidClose, 
                   "date"    : price.date, 
                   "volume"  : price.volume, 
                   "market status" : price.marketStatus})


df.set_index("date", inplace = True)


dfList = list()

# We iterate over old sampled data over every day 
currentDay = df.index[0].date()

lastDay = df.index[-1].date()
while currentDay <= lastDay : 
    subDf = df[dt.datetime.combine(currentDay, dt.time(hour = 0, minute = 0)) : dt.datetime.combine(currentDay, dt.time(hour = 23, minute = 59))]
    dfList.append(subDf)
    currentDay += dt.timedelta(days = 1)


sampledData = list()
for i in range(len(dfList)) : 
    isMarketOpen = False 
    # 1. We check if it exists an open phase of the market 
    if "open" in list(dfList[i]["market status"]) : 
        isMarketOpen = True 
    
    if isMarketOpen : 
        # 2. We aggregate the data 
        currentDay = dfList[i].index[0].date()
        for j in range(len(dayCandleList)) : 
            timeIni, timeEnd = dayCandleList[j].split("-")[0], dayCandleList[j].split("-")[1]
            t_ini = dt.time(hour = int(timeIni.split(":")[0]), minute = int(timeIni.split(":")[1]))
            t_end = dt.time(hour = int(timeEnd.split(":")[0]), minute = int(timeEnd.split(":")[1]))
            
            sampledData.append(dfList[i][dt.datetime.combine(currentDay, t_ini) : dt.datetime.combine(currentDay, t_end)])


# We create new data lists which will contain future data sampling 
askOpen  = list()
askHigh  = list()
askLow   = list() 
askClose = list()

bidOpen  = list()
bidHigh  = list()
bidLow   = list() 
bidClose = list()

date_     = list() 
volume   = list()
for i in range(len(sampledData)) : 
    askOpen.append(sampledData[i]["askOpen"].iloc[0])
    askHigh.append(max(sampledData[i]["askHigh"]))
    askLow.append(min(sampledData[i]["askLow"])) 
    askClose.append(sampledData[i]["askClose"].iloc[-1])
    
    bidOpen.append(sampledData[i]["bidOpen"].iloc[0])
    bidHigh.append(max(sampledData[i]["bidHigh"]))
    bidLow.append(min(sampledData[i]["bidLow"])) 
    bidClose.append(sampledData[i]["bidClose"].iloc[-1])
    
    date_.append(sampledData[i].index[0].to_pydatetime())
    volume.append(sum(sampledData[i]["volume"]))
    


    
price.askOpen = askOpen 
price.askHigh = askHigh 
price.askLow  = askLow 
price.askClose= askClose

price.bidOpen = bidOpen 
price.bidHigh = bidHigh 
price.bidLow  = bidLow 
price.bidCLose= bidClose 

price.date    = date_ 
price.volume  = volume  

price.sampled = True 


price.marketStatus = list()

price.setBaseTimeframe(timeframe = dt.timedelta(hours = 2))

# Only for base data ...
#price.fillMissingData()

#price.setMarketState() 

"""
price_H1 = copy.deepcopy(price)
price_H1.resampleData("02:00", name = "AUD.CAD_H1")




table = PRICE_TABLE([price, price_H1]) 
table.synchronize()

name = "AUD.CAD_H1" 
indexIni = dt.datetime(2010, 1, 4, 11)#0
indexEnd = dt.datetime(2010, 1, 4, 23, 30)#10

a = table.array(name, indexIni, indexEnd, format = "dictionnary")

print (a)