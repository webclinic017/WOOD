#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:24:27 2021

@author: loann
"""

import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"../../")
sys.path.append(filename)

import pandas as pd 
import numpy as np 
import datetime as dt 
import matplotlib.pyplot as plt 


import psychoSpectral_math as math 



def exponential(t0, tau, initialValue = 1., t = None) : 
    norm = initialValue*np.exp(1 / tau) 
    return norm*np.exp(-(t0 - t)/tau)


class STRATEGY : 
    
    def __init__(self) : 
        """ 
        Name of the symbol to be used for the run  
        """
        self.symbolName       = "EUR.USD"
        self.size             = 5000 
        
        # Wave scoring model parameters 
        self.waveScore        = 100 
        self.samplingSpace    = np.linspace(2, int(self.size/2.), self.waveScore, dtype = int)
        self.backtractMode    = True 
        self.maxBackTrack     = 3 
        self.waveScoreRange   = [1, 100]
        self.initialWaveScore = 1 
        
        # Time scoring model parameters 
        self.tau              = self.size
        self.iniVal           = 1 
        
        
        self.figNumber = 0 
        
        
        self.pointsLow = None 
        self.pointsHigh = None 
        
        return 
    
    def run(self, client) : 
        
        self.lastPrice = client.getLastPrice(self.symbolName)
        self.data = client.getHistoricalData(self.symbolName, self.size, 0, 0, onlyOpen = True) 
        
        if self.lastPrice.get("market state") == "open" :
        
            self.pointsLow, self.pointsHigh = None, None 
            
            self.pointsLow, self.pointsHigh = math.getHighLow(self.data, 
                                                              backtractMode = self.backtractMode, 
                                                              maxBackTrack  = self.maxBackTrack) 
            
            self.pointsLow, self.pointsHigh = math.getWaveScoring(self.data, 
                                                                  self.samplingSpace, 
                                                                  self.pointsLow, 
                                                                  self.pointsHigh, 
                                                                  backtractMode = self.backtractMode, 
                                                                  maxBackTrack  = self.maxBackTrack)
            
            self.pointsLow, self.pointsHigh = math.waveScoreNormalize(self.pointsLow, 
                                                                      self.pointsHigh, 
                                                                      waveScoreRange = self.waveScoreRange)
            
            # self.pointsLow, self.pointsHigh = math.getTimeScoring(self.data, 
            #                                                       self.pointsLow, 
            #                                                       self.pointsHigh, 
            #                                                       model           = exponential, 
            #                                                       modelParamsLow  = [self.size, self.tau, self.iniVal], 
            #                                                       modelParamsHigh = [self.size, self.tau, self.iniVal])
        
        

        return
    
    def show(self, client) : 
        
        if self.lastPrice.get("market state") == "open" : 
            
            plt.figure(figsize = (20, 12)) 
            
            plt.plot(np.arange(0, len(self.data.get("asklow")), 1), self.data.get("asklow"), c = "lightgrey")
            plt.plot(np.arange(0, len(self.data.get("askhigh")), 1), self.data.get("askhigh"), c = "lightgrey")
            
            for i in range(len(self.pointsLow)) : 
                #markerDown = "^"
                markerDown = "$"+str(int(self.pointsLow[i].get("score:wave")))+"$"
                colorDown = "red"
                plt.plot(self.pointsLow[i].get("index"), self.pointsLow[i].get("price"), 
                         marker = markerDown, markersize = min(2+self.pointsLow[i].get("score:wave"), 12**2), color = colorDown)
                
            for i in range(len(self.pointsHigh)) : 
                #markerUp = "v"
                markerUp = "$"+str(int(self.pointsHigh[i].get("score:wave")))+"$"
                colorUp = "blue"
                plt.plot(self.pointsHigh[i].get("index"), self.pointsHigh[i].get("price"), 
                         marker = markerUp, markersize = min(2+self.pointsHigh[i].get("score:wave"), 12**2), color = colorUp)
            
            figNumber = ""
            if self.figNumber < 10 : 
                figNumber = "000"+str(self.figNumber)
            if self.figNumber < 100 and self.figNumber >= 10 : 
                figNumber = "00"+str(self.figNumber)
            if self.figNumber < 1000 and self.figNumber >= 100 : 
                figNumber = "0"+str(self.figNumber)
            if self.figNumber < 10000 and self.figNumber >= 1000 : 
                figNumber = ""+str(self.figNumber)
            
            self.figNumber += 1 
            
            
            savePath = "/Volumes/DATA_SCIENCES/DEV/Q26/Sn1p3R/LumberJack/PICS"
            plt.savefig(savePath+"figure_"+str(figNumber)+".png", dpi = 300)
            plt.show()
        
        return 
    
    """ 
    
    """ 
    
