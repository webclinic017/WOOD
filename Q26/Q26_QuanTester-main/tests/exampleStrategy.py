#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 


        

class STRATEGY : 
    
    def __init__(self) : 
        
        self.symbolName = "AUD.CAD"
        self.fastMM = 50 
        self.slowMM = 100 
        
        self.activePosition = False
        self.order = None 
        self.direction = None 
        
        return 
    
    def simpleMovingAverage(self, y, period = 20, offset = 0) : 
        """
        This function calculates the simple moving average indicator

        Parameters
        ----------
        y : LIST
            List of data.
        period : INT, optional
            Period of the simple moving average. The default is 20.
        offset : INT, optional
            Offset of the simple moving average. The default is 0.

        Returns
        -------
        None.

        """
        
        if (offset != 0) : 
            y = y[:-offset]
        
        sma_temp = [y[0]]
        for ii in range(1, len(y)) :  
            
            if (ii < period) : 
                sum_temp = 0 
                for jj in range(0, ii) : 
                    sum_temp += y[jj]
                sum_temp = sum_temp / (ii)
                
                sma_temp.append(sum_temp)

            if (ii >= period) : 
                sum_temp = 0
                for jj in range(ii - period, ii) : 
                    sum_temp += y[jj]
                sum_temp = sum_temp / (period)
                
                sma_temp.append(sum_temp)
                
        return sma_temp 
        
        
    
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
            
            if len(data.get("askclose")) > self.slowMM : 
            
                self.fastSMA = self.simpleMovingAverage(data.get("askclose"), period = self.fastMM, offset = 0) 
                self.slowSMA = self.simpleMovingAverage(data.get("askclose"), period = self.slowMM, offset = 0) 
        
        if not self.activePosition : 
            
            if self.fastSMA[-1] > self.slowSMA[-1] : 
                
                orderList = portfolio.placeOrder(self.symbolName,
                                                 action     = "long", 
                                                 orderType  = "MKT", 
                                                 volume     = 0.1, 
                                                 stoploss   = 0., 
                                                 takeprofit = 99999999999999999.)
                
                self.direction = "long"
                self.order = orderList[0]
                self.activePosition = True 
            
            if self.fastSMA[-1] < self.slowSMA[-1] : 
                
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
            
            if ((self.fastSMA[-1] > self.slowSMA[-1] and self.direction == "short") or 
                (self.fastSMA[-1] < self.slowSMA[-1] and self.direction == "long")): 
                
                portfolio.closePosition(self.symbolName, self.order)
                self.order = None 
                self.direction = None 
                self.activePosition = False 
            
            
        
        # if True : 

        #     print ("======================================================")
        #     print ("STATE : ",lastPrice.get("market state"))
        #     print ("H state : ",data.get("market status")[-1])
        #     print ("date : ",str(lastPrice.get("time")))
        #     print ("close price : ",lastPrice.get("askclose"))
        #     print ("fast SMA : ",self.fastSMA[-1])
        #     print ("slow SMA : ",self.fastSMA[-1])
                # fig = plt.figure() 
                # plt.plot(data.get("date"), self.fastSMA, c = "blue") 
                # plt.plot(data.get("date"), self.slowSMA, c = "red")
                # plt.show()
                # plt.close(fig = fig)
                
        
        return 
        