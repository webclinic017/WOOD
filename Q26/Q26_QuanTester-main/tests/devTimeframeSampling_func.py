#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 08:49:49 2021

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
    



def timeSampler(marketOpeningHour,
                marketClosingHour, 
                marketLunch, 
                marketBreakList, 
                baseTimeframe, 
                timeframe) :

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
        
        
        
    
                
        # if marketLunch is not None : 
        #     if operation(dateList[-1], ">", marketLunch.split("-")[0]) : 
        #         dateList[-1] = marketLunch 
        
    
    candlesList = list() 
    for i in range(len(dateList)) : 
        candlesList.append((dateList[i]+"-"+dateListEnd[i]))
    
    return candlesList

marketOpeningHour = "00:00"
marketClosingHour = "24:00"
marketLunch       = None #"12:30-13:30"
marketBreakList   = list()#["09:30-10:00", "15:30-16:00"]

baseTimeframe = "00:05"
timeframe = "01:00" 

candlesList = timeSampler(marketOpeningHour,
                          marketClosingHour, 
                          marketLunch, 
                          marketBreakList, 
                          baseTimeframe, 
                          timeframe)

print (candlesList)

