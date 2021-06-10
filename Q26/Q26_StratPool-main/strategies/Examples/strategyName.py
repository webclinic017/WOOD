#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"../../")
sys.path.append(filename)

from indicators.moving_average import iMA

import matplotlib.pyplot as plt 




        

class STRATEGY : 
    
    def __init__(self) : 
        
        self.symbolName = "AUD.CAD"
        self.fastPeriod = 50 
        self.slowPeriod = 100 
        
        self.data = None 
        
        self.fastSMA = None 
        self.slowSMA = None 
        
        self.activePosition = False
        self.order = None 
        self.direction = None 
        
        return 
        
    
    def run(self, portfolio) : 
        """ 
        This function will be the only function that will be executed by the 
        simulation class. 
        """ 
        
        # We retrieve the data we need 
        data = portfolio.getHistoricalData(self.symbolName, -200, -1, 0, onlyOpen = True)
        lastPrice = portfolio.getLastPrice(self.symbolName)
        
        # If the market is open, we calculate our Moving averages 
        if lastPrice.get("market state") == "open" : 
            
            if len(data.get("askclose")) > self.slowPeriod : 
                
                if self.fastSMA is None : 
                    
                    self.fastSMA = iMA(data, 
                                       timeframe     = 1, 
                                       ma_period     = self.fastPeriod, 
                                       ma_method     = "linear", 
                                       ma_shift      = 0, 
                                       applied_price ="askclose", 
                                       shift         = 0)
                elif data.get("date")[-1] > self.data.get("date")[-1] : 
                    self.fastSMA = iMA(data, 
                                       timeframe     = 1, 
                                       ma_period     = self.fastPeriod, 
                                       ma_method     = "linear", 
                                       ma_shift      = 0, 
                                       applied_price ="askclose", 
                                       shift         = 0)
                
                if self.slowSMA is None : 
                    self.slowSMA = iMA(data,  
                                       timeframe     = 1, 
                                       ma_period     = self.slowPeriod, 
                                       ma_method     = "linear", 
                                       ma_shift      = 0, 
                                       applied_price ="askclose", 
                                       shift         = 0)
                elif data.get("date")[-1] > self.data.get("date")[-1] :
                    self.slowSMA = iMA(data, 
                                       timeframe     = 1, 
                                       ma_period     = self.slowPeriod, 
                                       ma_method     = "linear", 
                                       ma_shift      = 0, 
                                       applied_price ="askclose", 
                                       shift         = 0)
            

        if not self.activePosition : 
            
            if self.fastSMA.value > self.slowSMA.value : 
                
                orderList = portfolio.placeOrder(self.symbolName,
                                                 action     = "long", 
                                                 orderType  = "MKT", 
                                                 volume     = 0.1, 
                                                 stoploss   = 0., 
                                                 takeprofit = 99999999999999999.)
                
                self.direction = "long"
                self.order = orderList[0]
                self.activePosition = True 
            
            if self.fastSMA.value < self.slowSMA.value : 
                
                orderList = portfolio.placeOrder(self.symbolName,
                                                 action     = "short", 
                                                 orderType  = "MKT", 
                                                 volume     = 0.1, 
                                                 stoploss   = 99999999999999999., 
                                                 takeprofit = 0.)
                self.direction = "short"
                self.order = orderList[0]
                self.activePosition = True 
        
        else : 
            
            if ((self.fastSMA.value > self.slowSMA.value and self.direction == "short") or 
                (self.fastSMA.value < self.slowSMA.value and self.direction == "long")): 
                
                portfolio.closePosition(self.symbolName, self.order)
                self.order = None 
                self.direction = None 
                self.activePosition = False 
                
        self.data = data 
            
            
        
        # if True : 

        #     print ("======================================================")
        #     print ("STATE : ",lastPrice.get("market state"))
        #     print ("H state : ",data.get("market status")[-1])
        #     print ("date : ",str(lastPrice.get("time")))
        #     print ("close price : ",lastPrice.get("askclose"))
        #     print ("fast SMA : ",self.fastSMA[-1])
        #     print ("slow SMA : ",self.fastSMA[-1])
        # fig = plt.figure() 
        # plt.plot(data.get("date"), self.fastSMA.value, c = "blue") 
        # plt.plot(data.get("date"), self.slowSMA.value, c = "red")
        # plt.show()
        # plt.close(fig = fig)
                
        
        return 
        