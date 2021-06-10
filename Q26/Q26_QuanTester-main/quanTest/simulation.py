#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os 
import importlib
import datetime as dt 

from quanTest.analysis import ANALYSIS  
from quanTest.writer import WRITER 
from quanTest.diagnostics import TIMER

class SIMULATION(ANALYSIS, WRITER) : 

    def __init__(self, PORTFOLIO, PRICE_TABLE) : 
        # MAIN PARAMETERS 
        self.portfolio   = PORTFOLIO 
        self.priceTable  = PRICE_TABLE

        self.portfolio.historicalDataTimeframe = int(self.priceTable.priceList[0].baseTimeframe/dt.timedelta(minutes = 1))

        # SECONDARY PARAMETERS
        self.startIndex     = 0 
        self.stopIndex      = 999999999999999 
        self.subLoopModel   = "ohlc standard"#"close only"#
        self.maxHstDataSize = 1000



        self.strategyPath = None 
        self.strategyFile = None
        self.strategy     = None 

        # RUNTIME EVOLVING PARAMETERS 
        self.emulatedPriceTable = None 


        # LOG PARAMETERS 
        self.verbose  = False 
        self.portfolio.verbose = self.verbose 
        self.logEvery = 100 

        return 

    ###############################################################
    # BACKTEST OPERATIONS
    ###############################################################

    def importStrategy(self) : 

        #fileName, packageName = self.strategyPath.split('/')[-1][:-3], self.strategyPath.split('/')[-2]
        # filePath = filePath[:-len(fileName)-1-len(packageName)]
        # filePath = "/home/loann/Travail/Quantums/Travaux/Algorithmes/quantums_trader/qtrader/0.0.1/gui/dash_project/../../"
        #filePath = self.strategyPath[:-len(fileName)-3]
        #filePath = filePath[:-len(packageName)-1]
        sys.path.append(self.strategyPath) 
        #print (fileName, packageName, filePath)
        strategy = None 
        strategy = importlib.import_module(self.strategyFile)
        print (strategy)
        #from strategy import STRATEGY 
        self.strategy = strategy.STRATEGY()

        #sys.path.append(self.strategyPath)

    
    def parametersCheck(self) : 
        print ("Everything is fine, the simulation can be launched !")
        print ("This functionnality is not working for instance and need to be coded")
        return

    def run(self) : 
        
        # We initiate the simulation 
        i = self.startIndex 
        iMax = min(self.priceTable.len(), self.stopIndex)
        while i <= iMax : 

            # 1. We update the price value in the portfolio 
            #t1 = TIMER(name = "Price update")
            self.updatePrices(i) 
            #t1.stop()

            # 2. We calculate the EMULATED_PRICE_TABLE object that only contains past data 
            #t2 = TIMER(name = "History emulation")
            self.updateEmulatedHistory(i)
            #t2.stop()

            # 3. We enter in the sub loop where strategies are executed 
            #t3 = TIMER(name = "Strategy execution")
            self.subLoop() 
            #t3.stop()

            self.simulationState(i, iMax) 
            #print ("i = ",i,"/",iMax)
            # We increment the simulation 
            i += 1
            
        print ("Simulation terminated")
    
    def updatePrices(self, index) : 
        """ 
        Function that define the SYMBOL prices in the portfolio as a function of the given index 
        """
        currentPriceTable = self.priceTable.iloc(index)
        for key in list(self.portfolio.symbols.keys()) : 
            symbol = self.portfolio.symbols.get(key) 
            for skey in list(currentPriceTable.get(key).keys()) : 
                if skey != "market status" : 
                    setattr(symbol, skey, currentPriceTable.get(key).get(skey))
                if skey == "market status" : 
                    setattr(symbol, "marketState", currentPriceTable.get(key).get(skey))
    
    def updateEmulatedHistory(self, index) : 
        """ 
        Function that provide an emulated historical data array according the provided index 
        """ 
        array = dict()

        if index - 1 < 0 : index = 1 
        # We update the array for every base data symbols 
        for key in list(self.portfolio.symbols.keys()) : 
            index_ = index#self.priceTable.priceList[0].index[index]
            #index_ = self.priceTable.priceList[0].index.index(index)
            array.update({key : self.priceTable.array(key, max(0, index_ - self.maxHstDataSize), index_, format = "dictionnary")})
            # This is perfectly working like this MF ! 
        
        # We then update the array for every existing sampled data 
        for price in self.priceTable.priceList : 
            if price.sampled : 
                index_ = price.index[index]
                #print (index_)
                array.update({price.name : self.priceTable.array(price.name, max(0, index_ - self.maxHstDataSize), index_, format = "dictionnary")})
                

        self.emulatedPriceTable = array 
        self.portfolio.setHistoricalData(self.emulatedPriceTable)


    
    def executeStrategy(self) : 
        """ 

        """
        self.strategy.run(self.portfolio)
        pass 

    
    def subLoop(self) : 
        """ 

        """
        # 1. We retrieve the sub-loop price sequences 
        symbolPricesBid, symbolPricesAsk, size = self.subLoopModels()

        # 2. We apply the sub-loop 
        for i in range(size) : 
            for key in list(symbolPricesBid.keys()) : 
                self.portfolio.symbols.get(key).setCurrentPrice(bidprice = symbolPricesBid.get(key)[i], 
                                                                askprice = symbolPricesAsk.get(key)[i])
            self.executeStrategy() 
            self.portfolio.update()



    
    def subLoopModels(self) : 
        """ 
        Function that manage the evolution of the price at time scales lower than a candle scale one 
        """
        # 1. We define a symbolPrice sequence dict 
        symbolPricesAsk = dict()
        symbolPricesBid = dict()
        size = 0
        for key in list(self.portfolio.symbols.keys()) : 
            #symbol = self.portfolio.symbols.get(key) 
            symbolPricesAsk.update({key : list()}) 
            symbolPricesBid.update({key : list()})

        # 2. We apply the different sub-scale models 
        if self.subLoopModel == "close only"    : 
            size = 1 

            for key in list(symbolPricesAsk.keys()) : 
                symbolPricesAsk.get(key).append(self.portfolio.symbols.get(key).askclose)
            
            for key in list(symbolPricesBid.keys()) : 
                symbolPricesBid.get(key).append(self.portfolio.symbols.get(key).bidclose)

        if self.subLoopModel == "ohlc standard" : 
            size = 4
            
            for key in list(symbolPricesAsk.keys()) : 
                symbolPricesAsk.get(key).append(self.portfolio.symbols.get(key).askopen)
                symbolPricesAsk.get(key).append(self.portfolio.symbols.get(key).askhigh)
                symbolPricesAsk.get(key).append(self.portfolio.symbols.get(key).asklow)
                symbolPricesAsk.get(key).append(self.portfolio.symbols.get(key).askclose)

            for key in list(symbolPricesBid.keys()) : 
                symbolPricesBid.get(key).append(self.portfolio.symbols.get(key).bidopen)
                symbolPricesBid.get(key).append(self.portfolio.symbols.get(key).bidhigh)
                symbolPricesBid.get(key).append(self.portfolio.symbols.get(key).bidlow)
                symbolPricesBid.get(key).append(self.portfolio.symbols.get(key).bidclose)
            
            #print (list(symbolPricesAsk.keys()))
        
        # 3. We return the two sub-loop prices 
        return symbolPricesBid, symbolPricesAsk, size


    def simulationState(self, i, iMax) : 
        #print ((float(i)/iMax) % 0.1/100)
        #if (float(i)/iMax) % 0.1 == 0 : 
        if (i % self.logEvery == 0) : 
            print ("i = ",float(i)/iMax*100," %")
            # self.showEquityCurve()
            self.strategy.show(self.portfolio)





