#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:09:44 2021

@author: loann
"""
import numpy as np 


""" 
===============================================================================
WAVE SCORING MODEL FUNCTIONS
===============================================================================
""" 

def waveScoreNormalize(pointsLow, 
                       pointsHigh, 
                       waveScoreRange = [1, 100]) : 

    maxScore = 0 
    minScore = np.inf 
    for i in range(len(pointsLow)) : 
        if pointsLow[i].get("score:wave") > maxScore : 
            maxScore = pointsLow[i].get("score:wave")
        if pointsLow[i].get("score:wave") < minScore : 
            minScore = pointsLow[i].get("score:wave")
    for i in range(len(pointsHigh)) : 
        if pointsHigh[i].get("score:wave") > maxScore : 
            maxScore = pointsHigh[i].get("score:wave")
        if pointsHigh[i].get("score:wave") < minScore : 
            minScore = pointsHigh[i].get("score:wave")

    factor = (waveScoreRange[1] - waveScoreRange[0])/(maxScore - minScore)

    for i in range(len(pointsLow)) : 
        pointsLow[i].update({"score:wave" : int(waveScoreRange[0] + (pointsLow[i].get("score:wave") - minScore)*factor)})
    for i in range(len(pointsHigh)) : 
        pointsHigh[i].update({"score:wave" : int(waveScoreRange[0] + (pointsHigh[i].get("score:wave") - minScore)*factor)})
    
    return pointsLow, pointsHigh


def getWaveScoring(data, samplingSpace, pointsLow, pointsHigh, 
                   backtractMode = True, maxBackTrack = 3) :
    
    for i in range(len(samplingSpace)) : 
        sampledData, sampledSize = resample(data, samplingSpace[i])
        pointsLow, pointsHigh = updateScoring(sampledData, 
                                                   pointsLow, 
                                                   pointsHigh, 
                                                   backtractMode = backtractMode, 
                                                   maxBackTrack  = maxBackTrack)
    
    return pointsLow, pointsHigh


def updateScoring(sampledData, 
                  pointsLow, 
                  pointsHigh, 
                  backtractMode = True, 
                  maxBackTrack = 3) : 
    
    xLs_, xHs_, yLs, yHs, iLs_, iHs_ = getHighLow_(sampledData, backtractMode = backtractMode, maxBackTrack = maxBackTrack)
    
    iLs = list() 
    for i in range(len(iLs_)) : 
        iLs.append(sampledData.get("index min")[iLs_[i]])
    iHs = list() 
    for i in range(len(iHs_)) : 
        iHs.append(sampledData.get("index max")[iHs_[i]])
    
    
    for i in range(len(pointsLow)) : 
        for j in range(len(iLs)) :  
            if pointsLow[i].get("index") == iLs[j] : 
                pointsLow[i].update({"score:wave" : pointsLow[i].get("score:wave") + 1}) 
                
    for i in range(len(pointsHigh)) : 
        for j in range(len(iHs)) :  
            if pointsHigh[i].get("index") == iHs[j] : 
                pointsHigh[i].update({"score:wave" : pointsHigh[i].get("score:wave") + 1}) 
    
    return pointsLow, pointsHigh
    

def resample(data, samplingRate) : 
    size = len(data.get("date"))
    
    candleList = list() 
    for i in range(0, size, samplingRate) : 
        subData = {"askopen" : data.get("askopen")[i:i+samplingRate],
                   "askhigh" : data.get("askhigh")[i:i+samplingRate], 
                   "asklow"  : data.get("asklow")[i:i+samplingRate], 
                   "askclose": data.get("askclose")[i:i+samplingRate], 
                   "date"    : data.get("date")[i:i+samplingRate], 
                   "index"   : list(np.arange(i, i+samplingRate, 1, dtype = int))} 
        candleList.append(subData)
    
    sampledData = {"askopen" : list(), 
                   "askhigh" : list(), 
                   "asklow"  : list(), 
                   "askclose": list(), 
                   "date"    : list(), 
                   "index max"   : list(), 
                   "index min"   : list(), 
                   "index"       : list(),
                   "index from"  : list(), 
                   "index to"    : list()}
    
    for i in range(len(candleList)) : 
        sampledData.get("askopen").append(candleList[i].get("askopen")[0])
        sampledData.get("askhigh").append(max(candleList[i].get("askhigh")))
        sampledData.get("asklow").append(min(candleList[i].get("asklow")))
        sampledData.get("askclose").append(candleList[i].get("askclose")[-1])
        sampledData.get("index max").append(samplingRate*i + candleList[i].get("askhigh").index(max(candleList[i].get("askhigh"))))
        sampledData.get("index min").append(samplingRate*i + candleList[i].get("asklow").index(min(candleList[i].get("asklow"))))
        sampledData.get("index").append(int(sum(candleList[i].get("index"))/samplingRate))
        sampledData.get("index from").append(candleList[i].get("index")[0])
        sampledData.get("index to").append(candleList[i].get("index")[-1])
        #sampledData.get("date").append(int(sum(candleList[i].get("index"))/samplingRate))
        sampledData.get("date").append(candleList[i].get("date")[0])
        
    
    sampledSize = len(candleList)
    
    return sampledData, sampledSize
    
    
    

def getHighLow(data, backtractMode = False, maxBackTrack = 3, initialScore = 1) : 
    xL, xH, yL, yH, iL, iH = getHighLow_(data, 
                                         backtractMode = backtractMode, 
                                         maxBackTrack = maxBackTrack)
    
    pointsLow = list() 
    pointsHigh= list() 
    
    for i in range(len(xL)) : 
        pointsLow.append({"date" : xL[i], 
                          "price": yL[i], 
                          "index": iL[i],
                          "score:wave": initialScore})
    for i in range(len(yH)) : 
        pointsHigh.append({"date" : xH[i], 
                           "price": yH[i], 
                           "index": iH[i],
                           "score:wave": initialScore})
    
    return pointsLow, pointsHigh

def getHighLow_(data, backtractMode = False, maxBackTrack = 3) : 
         
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
        if data.get("askclose")[i] >= data.get("askopen")[i] : 
            if data.get("askclose")[i-1] <= data.get("askopen")[i-1] : 
                if data.get("askclose")[i] >= data.get("asklow")[i-1] : 

                    # Backtract operation 
                    if backtractMode : 
                        locMin = data.get("asklow")[i] 
                        locTime = data.get("date")[i] 
                        trackTime = data.get("date")[i] 
                        locIndex = 0 
                        j = 0 
                        while(trackTime > xB1 and i-j > 0 and j < maxBackTrack) : 
                            j = j + 1 
                            trackTime = data.get("date")[i-j] 
                            if data.get("asklow")[i-j] < locMin : 
                                locMin = data.get("asklow")[i-j]
                                locTime = data.get("date")[i-j] 
                                locIndex = j 

                        yA.append(locMin)
                        xA.append(locTime)
                        iA.append(i-locIndex)

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
        if data.get("askclose")[i] <= data.get("askopen")[i] : 
            if data.get("askclose")[i-1] >= data.get("askopen")[i-1] : 
                if data.get("askclose")[i] <= data.get("askhigh")[i-1] : 

                    # Backtract operation 
                    if backtractMode : 
                        locMax = data.get("askhigh")[i] 
                        locTime = data.get("date")[i] 
                        trackTime = data.get("date")[i] 
                        locIndex = 0 
                        j = 0 
                        while (trackTime > xA1 and i - j  > 0 and j < maxBackTrack) : 
                            j += 1 
                            trackTime = data.get("date")[i-j] 
                            if data.get("askhigh")[i-j] > locMax : 
                                locMax = data.get("askhigh")[i-j]
                                locTime = data.get("date")[i-j]
                                locIndex = j 

                        yB.append(locMax)
                        xB.append(locTime)
                        iB.append(i-locIndex)

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


""" 
===============================================================================
TIME SCORING MODEL FUNCTIONS
===============================================================================
""" 

def getTimeScoring(data, 
                   pointsLow, 
                   pointsHigh,
                   model = None, 
                   modelParamsLow = None,
                   modelParamsHigh = None) : 
    
    for i in range(len(pointsLow)) : 
        pointsLow[i].update({"score:time" : model(*modelParamsLow, pointsLow[i].get("index"))})
    for i in range(len(pointsHigh)) : 
        pointsHigh[i].update({"score:time" : model(*modelParamsHigh, pointsHigh[i].get("index"))})
    
    return pointsLow, pointsHigh