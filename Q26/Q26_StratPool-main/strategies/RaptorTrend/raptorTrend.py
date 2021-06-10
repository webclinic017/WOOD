#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"../../")
sys.path.append(filename)

import pandas as pd 
import numpy as np 
import datetime as dt 
import matplotlib.pyplot as plt 

"""
extern string assetName = "GOLD"; 
extern int timeframe = 5;
extern int subTimeframe = 1; // Timeframe used for calculating low and high points

extern double        minLeg = 1; // Volatility factor 
extern int        maxCandleLeg = 10; // Highest bar length between legs 
extern int        minCandleLeg = 1;  // Lowest bar length between legs 

extern string        patternName = "ABCD";       // Patterns code    
extern string       retracements = "50-200";    // Retracements percentage 
extern string           errorSup = "100-50";    // Retracements percentage error sup. 
extern string           errorInf = "100-50";    // Retracements percentage error inf. 
extern string   patternChirality = "both";        // Pattern chirality : 'up'/'down'/'both'
extern string         breakLevel = "CD";         // Break levels  
"""

class STRATEGY : 
    
    def __init__(self) : 
        
        """ 
        Name of the symbol to be used for the run  
        """
        self.symbolName       = "EUR.USD"
        
        """ 
        Entry parameters of the Raptor Trend strategy  
        """
        self.timeframeExt     = "_H1" 
        self.subTimeframeExt  = ""    # Timeframe used for calculating low and high points 
        
        self.minLeg           = 1     # Volatility factor 
        self.maxCandleLeg     = 10    # Highest bar length between legs 
        self.minCandleLeg     = 1    # Lowest bar length between legs 
        
        self.patternName      = "ABCD"  # Patterns code 
        self.retracements     = "67-200" # Retracements percentage 
        self.errorSup         = "30-100" # Retracements percentage error sup. 
        self.errorInf         = "30-100" # Retracements percentage error inf. 
        self.patternChirality = "both"  # Pattern chirality : 'up'/'down'/'both
        self.breakLevel       = "CD"    # Break levels
        
        self.patternMaxDelay = dt.timedelta(hours = 5) # Maximum time since the end of the last pattern 
        
        """ 
        Trading positions characteristics 
        """
        self.retraceTrigger = 0       # Percentage of retracement of the last wave to trigger an order 
        self.retraceStop    = 20 # Percentage of retracement of the last wave to trigger the stoploss val 
        self.retraceTarget  = 40 # Percentage of retracement of the last wave to trigger the target val      
        
        self.ratioSLTP      = 1 
        
        """ 
        Important data buffers 
        """ 
        self.volatility      = None 
        self.lastUpPattern   = None 
        self.lastDownPattern = None 
        
        
        
        self.activePosition = None  
        self.orderStartTime = None 
        
        
        self.activeOrder    = None 
        
        return
    
    def show(self, portfolio) : 
        
        plt.figure(figsize = (12, 8))
        for i in range(len(self.xL)) : 
            plt.plot(self.xL[i], self.yL[i], marker = "o", color = "blue")
        for i in range(len(self.xH)) : 
            plt.plot(self.xH[i], self.yH[i], marker = "o", color = "red")
        for i in range(len(self.patternUpList)) : 
            plt.plot(self.patternUpList[i][0], self.patternUpList[i][1], color = "black", lw = 3)
        for i in range(len(self.patternDownList)) : 
            plt.plot(self.patternDownList[i][0], self.patternDownList[i][1], color = "black", lw = 3)
        plt.plot(self.hstDataH.get("date"), self.hstDataH.get("asklow"), ls = '-', color = "black")
        plt.plot(self.hstDataH.get("date"), self.hstDataH.get("askhigh"), ls = '-', color = "black")
        plt.show()
        
        print ("===================================")
        print ("        STRAT. STATE               ")
        print ("===================================") 
        print ("Equity: ",portfolio.equity)
        print ("Last price : ",self.lastPrice.get("askclose"))
        print ("Actual position status : ",self.activePosition) 
        if len(portfolio.getActivePositions(self.symbolName)) == 1 :  
            print ("Entry point : ",round(portfolio.openPositions[0].executionPrice, 3),", Profit : ",round(portfolio.openPositions[0].profit, 3))
        
        
        return 
    
    def run(self, portfolio) : 
        
        
        lastPrice    = portfolio.getLastPrice(self.symbolName)
        self.lastPrice = lastPrice 
        #print ("date : ",lastPrice.get("date"))
        
        # We check is the market is open 
        if lastPrice.get("market state") == "open" : 
            
            # We retrieve historical data 
            # self.hstDataH     = portfolio.getHistoricalData(self.symbolName+self.timeframeExt, 100, 0,    0, onlyOpen = True)
            self.hstDataH     = portfolio.getHistoricalData(self.symbolName+self.subTimeframeExt, 100, 0,    0, onlyOpen = True)
            self.volatility = np.std(self.hstDataH.get("askclose"))

            #print (hstDataH.get("date"))
            self.xL, self.xH, self.yL, self.yH, self.iL, self.iH = getHighLow(self.hstDataH, backtractMode = True)
            self.patternUpList, self.patternDownList = self.getPatternList(self.xL, self.xH, self.yL, self.yH, self.iL, self.iH)
            
            if len(self.patternUpList) > 0 : 
                if lastPrice.get("date") - self.patternMaxDelay <= self.patternUpList[-1][0][-1] and self.activePosition is None : 
                    self.lastUpPattern = self.patternUpList[-1]
            if len(self.patternDownList) > 0 : 
                if lastPrice.get("date") - self.patternMaxDelay <= self.patternDownList[-1][0][-1] and self.activePosition is None : 
                    self.lastDownPattern = self.patternDownList[-1]
            
            
            # 
            if self.lastDownPattern is not None and self.lastUpPattern is not None : 
                if self.lastUpPattern[0][-1] > self.lastDownPattern[0][-1] : 
                    
                    if self.activePosition == "short" : 
                        portfolio.closePosition(self.symbolName, self.activeOrder) 
                        self.activePosition = None 
                        
                    if self.activePosition is None and lastPrice.get("askclose") > self.lastUpPattern[1][-1] and lastPrice.get("date") - self.patternMaxDelay <= self.lastUpPattern[0][-1] : 
                        stoploss = self.lastUpPattern[1][0]
                        takeprofit = self.lastUpPattern[1][-1] + (self.lastUpPattern[1][-1] - self.lastUpPattern[1][0])*self.ratioSLTP
                        
                        if lastPrice.get("askclose") > stoploss and lastPrice.get("askclose") < takeprofit :
                            orderList = portfolio.placeOrder(self.symbolName,
                                                             action     = "long", 
                                                             orderType  = "MKT", 
                                                             volume     = 0.1, 
                                                             stoploss   = stoploss, 
                                                             takeprofit = takeprofit)
                            self.activeOrder = orderList[0] 
                            self.activePosition = "long"
                        
                
                elif self.lastUpPattern[0][-1] < self.lastDownPattern[0][-1] and lastPrice.get("askclose") < self.lastDownPattern[1][-1] : 
                    
                    if self.activePosition == "long" : 
                        portfolio.closePosition(self.symbolName, self.activeOrder) 
                        self.activePosition = None 
                    
                    if self.activePosition is None and lastPrice.get("askclose") < self.lastDownPattern[1][-1] and lastPrice.get("date") - self.patternMaxDelay <= self.lastDownPattern[0][-1] : 
                        stoploss = self.lastDownPattern[1][0] 
                        takeprofit = self.lastDownPattern[1][-1] + (self.lastDownPattern[1][-1] - self.lastDownPattern[1][0])*self.ratioSLTP
                        
                        if lastPrice.get("askclose") < stoploss and lastPrice.get("askclose") > takeprofit : 
                            orderList = portfolio.placeOrder(self.symbolName,
                                                             action     = "short", 
                                                             orderType  = "MKT", 
                                                             volume     = 0.1, 
                                                             stoploss   = stoploss, 
                                                             takeprofit = takeprofit)
                            self.activeOrder = orderList[0] 
                            self.activePosition = "short"
                
            if len(portfolio.getActivePositions(self.symbolName)) == 0 : 
                self.activePosition = None

            
            # if ((not self.activePosition) and 
            #     (self.lastUpPattern is not None) and 
            #     (lastPrice.get("date") - self.patternMaxDelay <= self.lastUpPattern[0][-1])) : 
            #     if self.patternName == "ABCD" : 
                    
            #         stoploss = self.lastUpPattern[1][-1] - (self.lastUpPattern[1][-1] - self.lastUpPattern[1][-2])*self.retraceStop/100
            #         target   = self.lastUpPattern[1][-1] + (self.lastUpPattern[1][-1] - self.lastUpPattern[1][-2])*self.retraceTarget/100
            #         trigger  = self.lastUpPattern[1][-1] - (self.lastUpPattern[1][-1] - self.lastUpPattern[1][-2])*self.retraceTrigger/100
                    
            #         if lastPrice.get("askclose") >= trigger and lastPrice.get("askclose") > stoploss and lastPrice.get("askclose") < target : 
                        
            #             orderList = portfolio.placeOrder(self.symbolName,
            #                                               action     = "long", 
            #                                               orderType  = "MKT", 
            #                                               volume     = 0.1, 
            #                                               stoploss   = stoploss, 
            #                                               takeprofit = target)
                        
            #             # orderList = portfolio.placeOrder(self.symbolName,
            #             #                                   action     = "short", 
            #             #                                   orderType  = "MKT", 
            #             #                                   volume     = 0.1, 
            #             #                                   stoploss   = target, 
            #             #                                   takeprofit = stoploss)
                        
            #             print ("Order. Execution Price : ",lastPrice.get("askclose"),", Stoploss : ",stoploss,", Target : ",target)
                        
            #             self.activePosition = True 
                        
                        
            # if len(portfolio.getActivePositions(self.symbolName)) > 0 : 
            #     self.activePosition = True 
            # else : 
            #     self.activePosition = False 
            #     self.lastUpPattern = None 
            
            
            
            
            #print (patternList)
            # plt.figure(figsize = (12, 8))
            # for i in range(len(xL)) : 
            #     plt.plot(xL[i], yL[i], marker = "o", color = "blue")
            # for i in range(len(xH)) : 
            #     plt.plot(xH[i], yH[i], marker = "o", color = "red")
            # for i in range(len(patternUpList)) : 
            #     plt.plot(patternUpList[i][0], patternUpList[i][1], color = "black", lw = 3)
            # for i in range(len(patternDownList)) : 
            #     plt.plot(patternDownList[i][0], patternDownList[i][1], color = "black", lw = 3)
            # plt.plot(hstDataH.get("date"), hstDataH.get("asklow"), ls = '-', color = "black")
            # plt.plot(hstDataH.get("date"), hstDataH.get("askhigh"), ls = '-', color = "black")
            # plt.show()
            
        
        # print ("================================================")
        # print("data1 = ",pd.DataFrame(hstData))
        # print("data2 = ",pd.DataFrame(hstData2))
        # print("Last price time : ",str(lastPrice.get("date")),", current price ask : ",lastPrice.get("askprice"),
        #       "Market status : ", lastPrice.get("market state"))
        
        return 
    
    def getPatternList(self, xL, xH, yL, yH, iL, iH) : 
        patternUpList   = list()
        patternDownList = list()
        
        # Pattern properties initialisation 
        retracementsArray = [float(x)/100 for x in self.retracements.split("-")] 
        errorSupArray     = [float(x)/100 for x in self.errorSup.split("-")]
        errorInfArray     = [float(x)/100 for x in self.errorInf.split("-")]
        
        useD = False 
        useE = False 
        useF = False 
        useG = False 
        
        if "ABCD" in self.patternName : 
            useD = True 
        if "ABCDE" in self.patternName : 
            useE = True 
        if "ABCDEF" in self.patternName : 
            useF = True 
        if "ABCDEFG" in self.patternName : 
            useG = True 
        

        
        # Chirality "up" case 
        for i in range(len(xL)) : 
            for j in range(len(xH)) : 
                if self.checkHL(xL[i], xH[j], yL[i], yH[j]) and xH[j] > xL[i] : 
                    pattern = False 
                    for k in range(i+1, len(xL)) : 
                        if self.checkLeg(yH[j] - yL[k], yH[j] - yL[i], retracementsArray[0], errorInfArray[0], errorSupArray[0]) : 
                            if self.checkHL(xL[k], xH[j], yL[k], yH[j]) and xL[k] > xH[j] : 
                                if not useD : 
                                    if (iL[k] - iL[i] >= self.minCandleLeg and 
                                        iL[k] - iL[i] <= self.maxCandleLeg and 
                                        (self.patternChirality == "up" or self.patternChirality == "both")) : 
                                        pattern = True 
                                        patternUpList.append([[xL[i], xH[j], xL[k]],[yL[i], yH[j], yL[k]]])
                                else : 
                                    for l in range(j+1, len(xH)) : 
                                        if self.checkLeg(yH[l] - yL[k], yH[j] - yL[k], retracementsArray[1], errorInfArray[1], errorSupArray[1]) : 
                                            if self.checkHL(xL[k], xH[l], yL[k], yH[l]) and xL[k] < xH[l] : 
                                                if not useE : 
                                                    if ((iL[k] - iL[i] >= self.minCandleLeg and iH[l] - iH[j] >= self.minCandleLeg) and 
                                                        (iL[k] - iL[i] <= self.maxCandleLeg and iH[l] - iH[j] <= self.maxCandleLeg) and
                                                        (self.patternChirality == "up" or self.patternChirality == "both")) : 
                                                        pattern = True 
                                                        patternUpList.append([[xL[i], xH[j], xL[k], xH[l]],
                                                                            [yL[i], yH[j], yL[k], yH[l]]])
        
        # Chirality "down" case 
        for i in range(len(xH)) : 
            for j in range(len(xL)) : 
                if self.checkHL(xL[j], xH[i], yL[j], yH[i]) and xH[i] < xL[j] : 
                    pattern = False 
                    for k in range(i+1, len(xH)) : 
                        if self.checkLeg(yH[k] - yL[j], yH[i] - yL[j], retracementsArray[0], errorInfArray[0], errorSupArray[0]) : 
                            if self.checkHL(xL[j], xH[k], yL[j], yH[k]) and xL[j] < xH[k] : 
                                if not useD : 
                                    if (iH[k] - iH[i] >= self.minCandleLeg and 
                                        iH[k] - iH[i] <= self.maxCandleLeg and 
                                        (self.patternChirality == "down" or self.patternChirality == "both")) : 
                                        pattern = True 
                                        patternDownList.append([[xH[i], xL[j], xH[k]],[yH[i], yL[j], yH[k]]])
                                else : 
                                    for l in range(j+1, len(xL)) : 
                                        if self.checkLeg(yH[k] - yL[l], yH[k] - yL[j], retracementsArray[1], errorInfArray[1], errorSupArray[1]) : 
                                            if self.checkHL(xL[l], xH[k], yL[l], yH[k]) and xL[l] > xH[k] : 
                                                if not useE : 
                                                    if ((iH[k] - iH[i] >= self.minCandleLeg and iL[l] - iL[j] >= self.minCandleLeg) and 
                                                        (iH[k] - iH[i] <= self.maxCandleLeg and iL[l] - iL[j] <= self.maxCandleLeg) and 
                                                        (self.patternChirality == "down" or self.patternChirality == "both")) : 
                                                        pattern = True 
                                                        patternDownList.append([[xH[i], xL[j], xH[k], xL[l]],
                                                                                [yH[i], yL[j], yH[k], yL[l]]])
                                            
                                    
                    
        return patternUpList, patternDownList
        
    def checkHL(self, Xl, Xh, Yl, Yh) : 
        if Yh - Yl >= self.volatility*self.minLeg : 
            return True 
        else : 
            return False 
    
    def checkLeg(self, D, d, per, einf, esup) : 
        if (D >= per*(1 - einf)*d and D <= per*(1 + esup)*d) : 
            return True 
        else : 
            return False 
            
    

def getHighLow(data, backtractMode = False, maxBackTrack = 3) : 
        
    size = len(data.get("date"))
    
    # xA = np.empty(size, dtype = type(dt.datetime(2010, 11, 1)))
    # xB = np.empty(size, dtype = type(dt.datetime(2010, 11, 1)))
    # yA = np.zeros(size)
    # yB = np.zeros(size)
    xA = list()
    xB = list()
    yA = list()
    yB = list()
    iA = list() 
    iB = list()
    
    xA1 = data.get("date")[0]
    xB1 = data.get("date")[0]
    yA1 = None 
    yB1 = None 
    for i in range(1, size) : 
        # CASE 1 
        if data.get("askclose")[i] > data.get("askopen")[i] : 
            if data.get("askclose")[i-1] <= data.get("askopen")[i-1] : 
                if data.get("askclose")[i] > data.get("asklow")[i-1] : 
                    
                    # Backtract operation 
                    if backtractMode : 
                        locMin = data.get("asklow")[i] 
                        locTime = data.get("date")[i] 
                        trackTime = data.get("date")[i] 
                        locIndex = 0 
                        j = 0 
                        while(trackTime > xB1 and i-j-10 > 0 and j < maxBackTrack) : 
                            j = j + 1 
                            trackTime = data.get("date")[i-j] 
                            if data.get("asklow")[i-j] < locMin : 
                                locMin = data.get("asklow")[i-j]
                                locTime = data.get("date")[i-j] 
                                locIndex = j 
                        
                        yA.append(locMin)
                        xA.append(locTime)
                        iA.append(i-j)
                        
                        xA2 = xA1 
                        yA2 = yA1 
                        
                        xA1 = xA[-1] 
                        yA1 = yA[-1]
                    else : 
                        yA.append(data.get("asklow")[i])
                        xA.append(data.get("date")[i])
                        iA.append(i)
                        # yA[i] = data.get("asklow")[i]
                        # xA[i] = data.get("date")[i]
                        
                        xA2 = xA1 
                        yA2 = yA1 
                        
                        xA1 = xA[-1] 
                        yA1 = yA[-1]
                        
        # CASE 2 
        if data.get("askclose")[i] < data.get("askopen")[i] : 
            if data.get("askclose")[i-1] >= data.get("askopen")[i-1] : 
                if data.get("askclose")[i] < data.get("askhigh")[i-1] : 
                    
                    # Backtract operation 
                    if backtractMode : 
                        locMax = data.get("askhigh")[i] 
                        locTime = data.get("date")[i] 
                        trackTime = data.get("date")[i] 
                        locIndex = 0 
                        j = 0 
                        while (trackTime > xA1 and i - j - 10 > 0 and j < maxBackTrack) : 
                            j += 1 
                            trackTime = data.get("date")[i-j] 
                            if data.get("askhigh")[i-j] > locMax : 
                                locMax = data.get("askhigh")[i-j]
                                locTime = data.get("date")[i-j]
                                locIndex = j 
                                
                        yB.append(locMax)
                        xB.append(locTime)
                        iB.append(i-j)
                        
                        xB2 = xB1 
                        yB2 = yB1 
                        
                        xB1 = xB[-1] 
                        yB1 = yB[-1]
                    else : 
                        yB.append(data.get("askhigh")[i])
                        xB.append(data.get("date")[i])
                        iB.append(i)
                        # yB[i] = data.get("askhigh")[i]
                        # xB[i] = data.get("date")[i]
                        
                        xB2 = xB1 
                        yB2 = yB1 
                        
                        xB1 = xB[-1] 
                        yB1 = yB[-1]
                        
    
    return list(xA), list(xB), list(yA), list(yB), iA, iB
                    
                    

                    
                    


