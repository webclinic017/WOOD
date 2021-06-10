#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"../../")
sys.path.append(filename)

import pandas as pd 
import datetime as dt 

class STRATEGY : 
    
    def __init__(self) : 
        
        self.symbolName   = "EUR.USD"
        self.extentionHST = "_H1"
        
        self.activePosition = False 
        self.orderStartTime = None 
        self.activeOrder = None 
        
        return 
    
    def run(self, portfolio) : 
        
        
        lastPrice  = portfolio.getLastPrice(self.symbolName)
        # hstData    = portfolio.getHistoricalData(self.symbolName+self.extentionHST, 5, 0,    0, onlyOpen = True)
        # hstData2   = portfolio.getHistoricalData(self.symbolName, 5, 0,    0, onlyOpen = False)
        
        # 
        if lastPrice.get("market state") == "open" : 
            
            if not self.activePosition : 
                
                orderList = portfolio.placeOrder(self.symbolName,
                                                 action     = "long", 
                                                 orderType  = "MKT", 
                                                 volume     = 0.01, 
                                                 stoploss   = lastPrice.get("askprice")*(1 - 0.01), 
                                                 takeprofit = lastPrice.get("askprice")*(1 + 0.02))
                
                self.activeOrder = orderList[0] 
                self.activePosition = True 
            
        
        if len(portfolio.getActivePositions(self.symbolName)) > 0 : 
            self.activePosition = True 
        else : 
            self.activePosition = False 
            
        
        
        
        # print ("================================================")
        # print("data1 = ",pd.DataFrame(hstData))
        # print("data2 = ",pd.DataFrame(hstData2))
        # print("Last price time : ",str(lastPrice.get("date")),", current price ask : ",lastPrice.get("askprice"),
        #       "Market status : ", lastPrice.get("market state"))
        
        return 