#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd 
import datetime as dt 
import numpy as np 
import copy 
from quanTest import utils 

class PRICE : 

    def __init__(self, name) : 

        # Base properties 
        self.name       = name
        
        # Variables that allow to identify by default the name of the columns in the csv file 
        self.askOpen_    = "askopen"
        self.askHigh_    = "askhigh"
        self.askLow_     = "asklow"
        self.askClose_   = "askclose"

        self.bidOpen_    = "bidopen"
        self.bidHigh_    = "bidhigh"
        self.bidLow_     = "bidlow"
        self.bidClose_   = "bidclose"

        self.date_       = "time" 
        self.dateFormat_ = "%Y-%m-%d %H:%M:%S"
        self.volume_     = "volume"


        self.path        = None

        # Names of each list on the object PRICE
        self.askOpenTitle  = "askopen"
        self.askHighTitle  = "askhigh"
        self.askLowTitle   = "asklow"
        self.askCloseTitle = "askclose"

        self.bidOpenTitle  = "bidopen"
        self.bidHighTitle  = "bidhigh"
        self.bidLowTitle   = "bidlow"
        self.bidCloseTitle = "bidclose"

        self.dateTitle     = "time" 
        self.dateFormat    = "%Y-%m-%d %H:%M:%S"
        self.volumeTitle   = "volume"

        # Price object main properties 
        self.baseTimeframe = None
        # Local market time zone
        self.dataTimeZone      = 0 # UTC+...
        self.marketTimeZone    = 0 # UTC+...
        # Hours defined in the local market time zone 
        self.marketOpeningHour = "00:00"
        self.marketLunch       = None    # The format of this variable is : "HH:MM-HH:MM"
        self.marketBreakList   = list()  # The format of this variable is : ["HH:MM-HH:MM", "..."]
        self.marketClosingHour = "24:00"
        # Days of week where the market is open 
        self.daysOfWeek        = [0, 1, 2, 3, 4, 5, 6] # [#day of the week from 0 to 6]
        # List of dates where the market can be closed (vacancies ...)
        # To be done ..... 
        self.vacations         = list()

        # Initial, file structure 
        self.askOpen  = list()
        self.askHigh  = list()
        self.askLow   = list() 
        self.askClose = list()

        self.bidOpen  = list()
        self.bidHigh  = list()
        self.bidLow   = list() 
        self.bidClose = list()

        self.date     = list() 
        self.volume   = list()

        # Other important lists 
        self.marketStatus = list()    # Can be : "closed", "open"

        # Other informations 
        self.sampled = False # False if original data. True is sampled from sub resolution data 
        self.index   = list()

    def createCopy(self) : 
        return copy.deepcopy(self)

    def setColumnsTitle(self, 
                        askOpen    = None, 
                        askHigh    = None, 
                        askLow     = None, 
                        askClose   = None, 
                        bidOpen    = None, 
                        bidHigh    = None, 
                        bidLow     = None, 
                        bidClose   = None, 
                        date       = None,
                        dateFormat = None, 
                        volume     = None, 

                        splitDaysHours    = False, # Case where days and hours infos are not on the same column 
                        days              = None, 
                        hours             = None) : 
        """  
        This function allows to define the columns names in the file. 
        """
        if askOpen is not None : 
            self.askOpen_ = askOpen 
        if askHigh is not None : 
            self.askHigh_ = askHigh 
        if askLow is not None : 
            self.askLow_ = askLow 
        if askClose is not None : 
            self.askClose_ = askClose 
        if bidOpen is not None : 
            self.bidOpen_ = bidOpen 
        if bidHigh is not None : 
            self.bidHigh_ = bidHigh 
        if bidLow is not None : 
            self.bidLow_ = bidLow 
        if bidClose is not None : 
            self.bidClose_ = bidClose 
        if date is not None : 
            self.date_ = date 
        if dateFormat is not None : 
            self.dateFormat = dateFormat
        if volume is not None : 
            self.volume_ = volume 
        
        if splitDaysHours : 
            self.date_ = "split---"+days+"---"+hours
    
    def read(self, path) : 
        """ 
        Function that reads the csv file
        """

        df = pd.read_csv(path)

        try : 
            self.askOpen  = list(df[self.askOpen_])
        except : 
            pass 
        try : 
            self.askHigh  = list(df[self.askHigh_])
        except : 
            pass 
        try : 
            self.askLow   = list(df[self.askLow_])
        except : 
            pass 
        try : 
            self.askClose = list(df[self.askClose_])
        except : 
            pass 
        try : 
            self.bidOpen  = list(df[self.bidOpen_])
        except : 
            pass 
        try : 
            self.bidHigh  = list(df[self.bidHigh_])
        except : 
            pass 
        try : 
            self.bidLow   = list(df[self.bidLow_])
        except : 
            pass 
        try : 
            self.bidClose = list(df[self.bidClose_])
        except : 
            pass 
        try : 
            if not "split" in self.date_ : 
                tempDate      = list(df[self.date_])
                self.date     = [dt.datetime.strptime(x, self.dateFormat) for x in tempDate] 
            else : 
                locDate = self.date_.split("---")
                days_  = locDate[1] 
                hours_ = locDate[2]
                tempDays      = list(df[days_])
                tempHours     = list(df[hours_]) 
                self.date = list() 
                for i in range(len(tempDays)) : 
                    self.date.append(dt.datetime.strptime(tempDays[i]+" "+tempHours[i], self.dateFormat))
        except : 
            print ("An error occured")
            pass 
        try : 
            self.volume   = list(df[self.volume_])
        except : 
            pass 
    
    def setBaseTimeframe(self, 
                         timeframe = dt.timedelta(minutes = 1)) : 
        """ 
        Function allowing to define the base timeframe of the PRICE object.
        """
        if type(timeframe) == type(dt.timedelta(minutes = 1)) : 
            self.baseTimeframe = timeframe 
        else : 
            print ("Error. Bad timeframe reference format.")
    
    def fillMissingData(self, 
                        model = "constant") : 
        """ 
        Function that allows to fill the missing data so that it exists a price data candle for 
        every time step from the beginning to the end of the dataframe. 
        Different filling models can be used : 
            - "constant" : This model just fill the candles using the last knew price (the standard model, to be used when data quality is good)
            - other models to be defined ... 

        """
        if not self.sampled :

            filledAskOpen  = list() 
            filledAskHigh  = list()
            filledAskLow   = list()
            filledAskClose = list()

            filledBidOpen  = list() 
            filledBidHigh  = list()
            filledBidLow   = list()
            filledBidClose = list()

            filledDate     = list() 
            filledVolume   = list()

        

            if model == "constant" : 
                

                iniTime = self.date[0] 
                endTime = self.date[-1] 

                varTime = iniTime 
                varIndex = 0 
                while varTime <= endTime and varIndex < len(self.date): 
                    filledAskOpen.append(self.askOpen[varIndex])
                    filledAskHigh.append(self.askHigh[varIndex])
                    filledAskLow.append(self.askLow[varIndex])
                    filledAskClose.append(self.askClose[varIndex])

                    filledBidOpen.append(self.bidOpen[varIndex])
                    filledBidHigh.append(self.bidHigh[varIndex])
                    filledBidLow.append(self.bidLow[varIndex])
                    filledBidClose.append(self.bidClose[varIndex])

                    filledDate.append(varTime)
                    filledVolume.append(self.volume[varIndex])

                    if self.date[varIndex] == varTime : 
                        varIndex += 1 
                    
                    else :  
                        pass

                    # We increment the time variable from the base time delta 
                    varTime += self.baseTimeframe 

        
            self.askOpen  = filledAskOpen 
            self.askHigh  = filledAskHigh 
            self.askLow   = filledAskLow 
            self.askClose = filledAskClose

            self.bidOpen  = filledBidOpen 
            self.bidHigh  = filledBidHigh 
            self.bidLow   = filledBidLow 
            self.bidClose = filledBidClose

            self.date     = filledDate 
            self.volume   = filledVolume



    def shiftMarketTime(self, 
                        timeshift = 0) : 
        """ 
        Function that allows to shift the market hours to make it fit with 
        UTC+0 time if this is not already the case 
        """
        self.date = list(np.array(self.date) + dt.timedelta(hours = timeshift)) 
    
    def setMarketTimeZone(self, 
                          timezone = 0) : 
        """ 
        Function that allows to define the price time data time zone 
        according to UTC+0. 
        """
        self.marketTimeZone = timezone 
    
    def setDataTimeZone(self, 
                        timezone = 0) : 
        """ 
        Function that allows to define the timezone in which the data is printed 
        """
        self.dataTimeZone = timezone

    def setMarketState(self) : 
        

        for i in range(len(self.date)) : 

            if self.date[i].weekday() not in self.daysOfWeek : 
                self.marketStatus.append("closed") 
            else : 
                locHour   = "0"+str(self.date[i].hour) if  self.date[i].hour < 10 else str(self.date[i].hour) 
                locMinute = "0"+str(self.date[i].minute) if self.date[i].minute < 10 else str(self.date[i].minute)
                hourOfTheDay = locHour+":"+locMinute

                # We shift the hour of the day to have it in a local reference timeframe 
                h_dtz = int(locHour) 
                h_ut0 = h_dtz - self.dataTimeZone 
                if h_ut0 < 0 : 
                    h_ut0 = 24 + h_ut0  
                if h_ut0 > 23 : 
                    h_ut0 = 24 - h_ut0
                h_mtz = h_ut0 + self.marketTimeZone 
                if h_mtz > 23 : 
                    h_mtz = 24 - h_mtz 
                if h_mtz < 0 : 
                    h_mtz = 24 + h_mtz

                locHour   = "0"+str(h_mtz) if  h_mtz < 10 else str(h_mtz) 
                hourOfTheDay = locHour+":"+locMinute
                #print ("hOD : ",hourOfTheDay, ", h_dtz = ",h_dtz,", h_ut0 = ",h_ut0,", h_mtz = ",h_mtz)

                # We test the local hour of the day 
                locMarketState = "open" 
                if utils.compareHour(hourOfTheDay, "<=", self.marketOpeningHour) : 
                    locMarketState = "closed"
                if utils.compareHour(hourOfTheDay, ">=", self.marketClosingHour) : 
                    locMarketState = "closed"
                if self.marketLunch is not None : 
                    marketLunch = self.marketLunch.split("-")
                    beginLunch  = marketLunch[0]
                    endLunch    = marketLunch[1]
                    if utils.compareHour(hourOfTheDay, ">=", beginLunch) and utils.compareHour(hourOfTheDay, "<=", endLunch) : 
                        locMarketState = "closed"
                if len(self.marketBreakList) > 0 : 
                    for j in range(len(self.marketBreakList)) : 
                        marketBreak = self.marketBreakList[j].split("-")
                        beginBreak  = marketBreak[0]
                        endBreak    = marketBreak[1]
                        if utils.compareHour(hourOfTheDay, ">=", beginBreak) and utils.compareHour(hourOfTheDay, "<=", endBreak) : 
                            locMarketState = "closed"
                self.marketStatus.append(locMarketState) 
            
    def setBaseIndex(self) : 

        #if not self.sampled :

        self.index.append(-1) 

        for i in range (len(self.marketStatus)) : 

            if self.marketStatus[i] == "open" : 
                self.index.append(self.index[-1] + 1)
            else : 
                self.index.append(self.index[-1])
    
        del self.index[0]



    def timeDaySampler(self, 
                       baseTimeframe, 
                       timeframe) :

        marketOpeningHour = self.marketOpeningHour 
        marketClosingHour = self.marketClosingHour 
        marketLunch       = self.marketLunch 
        marketBreakList   = self.marketBreakList 


        dateList    = list() 
        dateListEnd = list()
        
        # 1. We start with the market open hour 
        dateList.append(marketOpeningHour) 
        dateListEnd.append(operation(dateList[-1], "+", operation(timeframe, "-", baseTimeframe)))
        currentTime = dateList[0] 
        
        marketBreakList_ = marketBreakList.copy() 
        
        if marketLunch is not None : 
            marketBreakList_.append(marketLunch)
        
        if len(marketBreakList_) > 0 : 
            # We check that the end of the period is not inside a break 
            insideBreak = False 
            breakIndexList = list()
            for i in range(len(marketBreakList_)) : 
                if (operation(dateListEnd[-1], ">=", marketBreakList_[i].split("-")[0]) and
                    #operation(dateListEnd[-1], "<", marketBreakList_[i].split("-")[1]) and 
                    operation(dateList[-1], "<", marketBreakList_[i].split("-")[0])): 
                    breakIndexList.append(i)
                    insideBreak = True 
            if insideBreak : 
                earlyIndex = breakIndexList[0] 
                for i in range (1, len(breakIndexList)) : 
                    if operation(marketBreakList_[breakIndexList[i]].split("-")[0], "<", marketBreakList_[earlyIndex].split("-")[0]) : 
                        earlyIndex = breakIndexList[i] 
                dateListEnd[-1] = operation(marketBreakList_[earlyIndex].split("-")[0], "-", baseTimeframe)
                #dateList[-1] = marketBreakList_[earlyIndex].split("-")[1]
        
        while operation(currentTime, "<", marketClosingHour): 
            
            # We add a new period 
            #dateList.append(operation(currentTime, "+", timeframe)) 
            dateList.append(operation(dateListEnd[-1], "+", baseTimeframe))
            dateListEnd.append(operation(dateList[-1], "+", operation(timeframe, "-", baseTimeframe)))
            
            
            if len(marketBreakList_) > 0 : 
                # We check that the end of the period is not inside a break 
                insideBreak = False 
                breakIndexList = list()
                for i in range(len(marketBreakList_)) : 
                    
                    if (operation(dateListEnd[-1], ">=", marketBreakList_[i].split("-")[0]) and
                        #operation(dateListEnd[-1], "<", marketBreakList_[i].split("-")[1]) and 
                        operation(dateList[-1], "<", marketBreakList_[i].split("-")[0])): 
                        
                        breakIndexList.append(i)
                        insideBreak = True 
                if insideBreak : 
                    earlyIndex = breakIndexList[0] 
                    for i in range (1, len(breakIndexList)) : 
                        if operation(marketBreakList_[breakIndexList[i]].split("-")[0], "<", marketBreakList_[earlyIndex].split("-")[0]) : 
                            earlyIndex = breakIndexList[i] 
                    dateListEnd[-1] = operation(marketBreakList_[earlyIndex].split("-")[0], "-", baseTimeframe)
                    #dateList[-1] = marketBreakList_[earlyIndex].split("-")[1]
                
                # We check that the begining of the period is not inside a break 
                insideBreak = False 
                breakIndexList = list()
                for i in range(len(marketBreakList_)) : 
                    if (operation(dateList[-1], ">=", marketBreakList_[i].split("-")[0]) and 
                        operation(dateList[-1], "<", marketBreakList_[i].split("-")[1])) : 
                        breakIndexList.append(i)
                        insideBreak = True 
                
                if insideBreak : 
                    lateIndex = breakIndexList[0] 
                    for i in range (1, len(breakIndexList)) : 
                        if operation(marketBreakList_[breakIndexList[i]].split("-")[1], ">", marketBreakList_[lateIndex].split("-")[1]) : 
                            lateIndex = breakIndexList[i] 
                            
                    dateList[-1]    = marketBreakList_[lateIndex].split("-")[1]
                    dateListEnd[-1] = operation(dateList[-1], "+", operation(timeframe, "-", baseTimeframe))
                    
                    # We check that the end of the period is not inside a break 
                    insideBreak = False 
                    breakIndexList = list()
                    for i in range(len(marketBreakList_)) : 
                        
                        if (operation(dateListEnd[-1], ">=", marketBreakList_[i].split("-")[0]) and
                            #operation(dateListEnd[-1], "<", marketBreakList_[i].split("-")[1]) and 
                            operation(dateList[-1], "<", marketBreakList_[i].split("-")[0])): 
                            
                            breakIndexList.append(i)
                            insideBreak = True 
                    if insideBreak : 
                        earlyIndex = breakIndexList[0] 
                        for i in range (1, len(breakIndexList)) : 
                            #print(marketBreakList_[i].split("-")[0])
                            if operation(marketBreakList_[breakIndexList[i]].split("-")[0], "<", marketBreakList_[earlyIndex].split("-")[0]) : 
                                earlyIndex = breakIndexList[i] 
                        dateListEnd[-1] = operation(marketBreakList_[earlyIndex].split("-")[0], "-", baseTimeframe)
                        #print(dateListEnd[-1])
                        #dateList[-1] = marketBreakList_[earlyIndex].split("-")[1]
            
            
            currentTime = dateList[-1]
            
            if operation(dateList[-1], ">=", marketClosingHour) : 
                del dateList[-1] 
                del dateListEnd[-1]
            if operation(dateListEnd[-1], ">", marketClosingHour) : 
                dateListEnd[-1] = operation(marketClosingHour, "-", baseTimeframe) 
        
        candlesList = list() 
        for i in range(len(dateList)) : 
            candlesList.append((dateList[i]+"-"+dateListEnd[i]))
        
        return candlesList

    def resampleData(self, newTimeFrame, name = None) : 

        # 0. We transform the base timeframe attribute into a readable format 
        baseTf_lst = str(self.baseTimeframe).split(":") 
        baseTf_h = int(baseTf_lst[0]) 
        baseTf_m = int(baseTf_lst[1]) 

        if baseTf_h < 10 : 
            baseTf_h = "0"+str(baseTf_h)
        else: 
            baseTf_h = str(baseTf_h)
        if baseTf_m < 10 : 
            baseTf_m = "0"+str(baseTf_m)
        else: 
            baseTf_m = str(baseTf_m)

        baseTf = baseTf_h+":"+baseTf_m 

        # We generate a day time sampler hours list 
        dayCandleList = self.timeDaySampler(baseTf, newTimeFrame)

        # We create a pandas dataframe from our data and pass date as index 
        df = pd.DataFrame({"askOpen" : self.askOpen, 
                        "askHigh" : self.askHigh, 
                        "askLow"  : self.askLow, 
                        "askClose": self.askClose, 
                        "bidOpen" : self.bidOpen, 
                        "bidHigh" : self.bidHigh, 
                        "bidLow"  : self.bidLow, 
                        "bidClose": self.bidClose, 
                        "date"    : self.date, 
                        "volume"  : self.volume, 
                        "market status" : self.marketStatus})


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
        volume    = list()
        
        for i in range(len(sampledData)) : 
            if len(sampledData[i]) > 0 : 
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

            #if len(index) > 0 : 
            #    index.append(self.date.index(date_[-1], index[-1]))
            #else : 
            #    index.append(self.date.index(date_[-1]))
            

        index     = list() 
        j = 0 
        for i in range(len(self.date)) : 
            if j+1 < len(date_) : 
                #print (self.date[i], date_[j], date_[j+1])
                if self.date[i] < date_[j] and j == 0 : 
                    index.append(-1)
                elif (self.date[i] >= date_[j] and self.date[i] <= date_[j+1]) : 
                    index.append(j)
                    #print (True)
                else : 
                    index.append(j+1)
                    j += 1
            
        self.askOpen = askOpen 
        self.askHigh = askHigh 
        self.askLow  = askLow 
        self.askClose= askClose

        self.bidOpen = bidOpen 
        self.bidHigh = bidHigh 
        self.bidLow  = bidLow 
        self.bidCLose= bidClose 

        #print (date_[6])




        self.date    = date_ 
        self.index   = index 
        
        self.volume  = volume  

        self.sampled = True 


        self.marketStatus = list()

        self.setBaseTimeframe(timeframe = dt.timedelta(hours = int(newTimeFrame.split(":")[0]), minutes = int(newTimeFrame.split(":")[1])))
        #print(self.date[6])

        if name is not None : 
            self.name = name 
        else : 
            self.name += "_resampled"
        

class PRICE_TABLE : 

    def __init__(self, priceList) : 
        self.priceList = priceList  # Here price list is a list of the objects PRICE to be synchronized 
        self.synchronized = False 
    
    def synchronize(self) : 

        # 1. We cut the useless edges of the data 
        lateGeneralBeginning = self.priceList[0].date[0]
        earlyGeneralEnd      = self.priceList[0].date[-1] 

        for i in range(1, len(self.priceList)) : 

            if lateGeneralBeginning > self.priceList[i].date[0] : 
                lateGeneralBeginning = self.priceList[i].date[0]
            
            if earlyGeneralEnd > self.priceList[i].date[-1] : 
                earlyGeneralEnd = self.priceList[i].date[-1]

        # 2. We fill the missing data 
        for i in range(len(self.priceList)) : 

            self.priceList[i].fillMissingData(model = "constant")
            self.priceList[i].setMarketState() 
            self.priceList[i].setBaseIndex()
        
        self.synchronized = True 
    
    def iloc(self, index, exceptSampled = True) : 
        table = dict() 
        for price in self.priceList : 
            
            toUpdate = True 
            if exceptSampled : 
                if price.sampled : 
                    toUpdate = False 
                
            if toUpdate : 
                if index < len(price.date) : 
                    table.update({price.name : {
                        "askopen"       : price.askOpen[index],
                        "askhigh"       : price.askHigh[index],
                        "asklow"        : price.askLow[index],
                        "askclose"      : price.askClose[index],
                        "bidopen"       : price.bidOpen[index], 
                        "bidhigh"       : price.bidHigh[index], 
                        "bidlow"        : price.bidLow[index], 
                        "bidclose"      : price.bidClose[index], 
                        "time"          : price.date[index], 
                        "volume"        : price.volume[index], 
                        "market status" : price.marketStatus[index] 
                    }})
                else : 
                    print ("Index out of range for symbol : ", price.name)
                    index = -1 
                    table.update({price.name : {
                        "askopen"       : price.askOpen[index],
                        "askhigh"       : price.askHigh[index],
                        "asklow"        : price.askLow[index],
                        "askclose"      : price.askClose[index],
                        "bidopen"       : price.bidOpen[index], 
                        "bidhigh"       : price.bidHigh[index], 
                        "bidlow"        : price.bidLow[index], 
                        "bidclose"      : price.bidClose[index], 
                        "time"          : price.date[index], 
                        "volume"        : price.volume[index], 
                        "market status" : price.marketStatus[index] 
                    }})
        return table 
    
    def len(self) : 

        if self.synchronized : 

            return len(self.priceList[0].date)
        
        else : 

            print ("Data not synchronzed yet, cannot return any length")



    def array(self, name, indexIni, indexEnd, format = "dictionnary") : 
        price = None 
        for i in range(len(self.priceList)) : 
            if self.priceList[i].name == name : 
                price = self.priceList[i] 

        if type(indexIni) == type(1) and type(indexEnd) == type(1) : 

            array_ = {"askopen"       : price.askOpen[indexIni : indexEnd],
                    "askhigh"         : price.askHigh[indexIni : indexEnd],
                    "asklow"          : price.askLow[indexIni : indexEnd],
                    "askclose"        : price.askClose[indexIni : indexEnd],
                    "bidopen"         : price.bidOpen[indexIni : indexEnd], 
                    "bidhigh"         : price.bidHigh[indexIni : indexEnd], 
                    "bidlow"          : price.bidLow[indexIni : indexEnd], 
                    "bidclose"        : price.bidClose[indexIni : indexEnd], 
                    "date"            : price.date[indexIni : indexEnd], 
                    "volume"          : price.volume[indexIni : indexEnd], 
                    "market status"   : price.marketStatus[indexIni : indexEnd]}

        if type(indexIni) == type(dt.datetime(2021, 1, 12, 12, 12)) and type(indexEnd) == type(dt.datetime(2021, 1, 12, 12, 12)) : 

            locIndexIni_ = min(price.date, key=lambda x: abs(x - indexIni))
            locIndexEnd_ = min(price.date, key=lambda x: abs(x - indexEnd))

            locIndexIni = price.date.index(locIndexIni_) 
            locIndexEnd = price.date.index(locIndexEnd_) 

            if locIndexIni_ > indexIni : 
                locIndexIni -= 1
            if locIndexEnd_ > indexEnd : 
                locIndexEnd -= 1 

            array_ = {"askopen"       : price.askOpen[locIndexIni : locIndexEnd],
                    "askhigh"         : price.askHigh[locIndexIni : locIndexEnd],
                    "asklow"          : price.askLow[locIndexIni : locIndexEnd],
                    "askclose"        : price.askClose[locIndexIni : locIndexEnd],
                    "bidopen"         : price.bidOpen[locIndexIni : locIndexEnd], 
                    "bidhigh"         : price.bidHigh[locIndexIni : locIndexEnd], 
                    "bidlow"          : price.bidLow[locIndexIni : locIndexEnd], 
                    "bidclose"        : price.bidClose[locIndexIni : locIndexEnd], 
                    "date"            : price.date[locIndexIni : locIndexEnd], 
                    "volume"          : price.volume[locIndexIni : locIndexEnd], 
                    "market status"   : price.marketStatus[locIndexIni : locIndexEnd]}

        if format == "dictionnary" : 
            return array_
        if format == "dataframe" : 
            df = pd.DataFrame(data = array_)
            return df 

    def isSampled(self, priceName) : 
        locIndex = None 
        for i in range(len(self.priceList)) : 
            price = self.priceList[i]
            if price.name == priceName : 
                locIndex = i
        return self.priceList[locIndex].sampled




def operation(h1, operator, h2) : 
    h1_ = h1 
    h2_ = h2
    
    h1 = h1.split(":")
    h1_hour, h1_minute = int(h1[0]), int(h1[1])
    h2 = h2.split(":") 
    h2_hour, h2_minute = int(h2[0]), int(h2[1]) 
    

    if operator == "+" : 
        h3_hour, h3_minute = None, None 
        h3 = None 
        
        h3_hour   = h1_hour + h2_hour 
        h3_minute = h1_minute + h2_minute 
        
        while h3_minute >= 60 : 
            h3_minute -= 60 
            h3_hour   += 1 

        
        if h3_hour   < 10 and h3_hour >= 0 : 
            h3_hour   = "0"+str(h3_hour) 
        else : 
            h3_hour = str(h3_hour)
        if h3_minute < 10 : 
            h3_minute = "0"+str(h3_minute)
        else : 
            h3_minute = str(h3_minute)
            
        h3 = h3_hour+":"+h3_minute
        
        return h3
    
    if operator == "-" : 
        h3_hour, h3_minute = None, None 
        h3 = None  
        
        h3_hour   = h1_hour - h2_hour 
        h3_minute = h1_minute - h2_minute 
        
        while h3_minute < 0 : 
            h3_minute  = 60 - abs(h3_minute) 
            h3_hour   -= 1
        
        if abs(h3_hour)   < 10 and h3_hour >= 0 : 
            h3_hour   = "0"+str(h3_hour) 
        elif abs(h3_hour)   < 10 and h3_hour < 0 : 
            h3_hour   = "-0"+str(h3_hour) 
        else : 
            h3_hour = str(h3_hour)
        if h3_minute < 10 : 
            h3_minute = "0"+str(h3_minute)
        else : 
            h3_minute = str(h3_minute)
            
        h3 = h3_hour+":"+h3_minute
        
        return h3
        
    
    if operator == "<" : 
        
        if h1_hour < h2_hour : 
            return True 
        if h1_hour > h2_hour : 
            return False 
        if h1_hour == h2_hour : 
            if h1_minute < h2_minute : 
                return True 
            else : 
                return False 
    
    if operator == ">" : 
        
        if h1_hour > h2_hour : 
            return True 
        if h1_hour < h2_hour : 
            return False 
        if h1_hour == h2_hour : 
            if h1_minute > h2_minute : 
                return True 
            else : 
                return False 
            
    if operator == ">=" : 
        
        if h1_hour > h2_hour : 
            return True 
        if h1_hour < h2_hour : 
            return False 
        if h1_hour == h2_hour : 
            if h1_minute >= h2_minute : 
                return True 
            else : 
                return False 
    
    if operator == "min" : 
        
        if h1_hour < h2_hour : 
            return h1_ 
        if h1_hour > h2_hour : 
            return h2_
        if h1_hour ==  h2_hour : 
            if h1_minute < h2_minute : 
                return h1_ 
            else : 
                return h2_
