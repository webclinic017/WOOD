#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:34:50 2020

@author: lbrahimi
"""

import numpy as np 
import matplotlib.pyplot as plt 

class SMA : 
    
    def __init__(self, y, period = 20, offset = 0) : 
        """
        This function is an initializer of the simple moving average indicator

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
        
        self.value  = sma_temp 
        self.period = period 
        self.offset = offset 
        
        
    def plot(self, xsize = 8, ysize = 6) : 
        """
        Fonction permettant de tracer l'indicateur SMA

        Parameters
        ----------
        xsize : INT, optional
            DESCRIPTION. The default is 8.
        ysize : INT, optional
            DESCRIPTION. The default is 6.

        Returns
        -------
        None.

        """
        plt.figure(figsize=(xsize, ysize))
        plt.plot(self.value, ls='-', c="black", lw = 2)
        plt.show()
        plt.close()
    
    
    def update(self, last_y) : 
        """
        

        Parameters
        ----------
        last_y : LIST 
            This parameter is a list of the last data, the length should be higher than
            the period-offset.

        Returns
        -------
        None.

        """
        sma_temp = self.value 
        period   = self.period 
        offset   = self.offset 
        
        if (offset != 0) : 
            last_y = last_y[:-offset]
        
        # last_value = (sma_temp[-1]*period + last_y)/period
        
        # last_value = 
        sum_temp = 0 
        for jj in range(len(last_y) - period, len(last_y)) : 
            sum_temp += last_y[jj]
        sum_temp = sum_temp / period 
        
        
        sma_temp.append(sum_temp)
        
        self.value = sma_temp
        del self.value[0]
        
        
        
        
class MULTSMA : 
    
    def __init__(self, main_y, period = [20], offset = [0]) : 
        
        nsma = len(period)
        
        sma_temp = []
        for ll in range(nsma) : 
            
            if (offset[ll] != 0) : 
                y = main_y[:-offset[ll]].copy()
            else : 
                y = main_y.copy()
            
            sma_temp.append([y[0]])
        
            for ii in range(1, len(y)) :  
                
                if (ii < period[ll]) : 
                    sum_temp = 0 
                    for jj in range(0, ii) : 
                        sum_temp += y[jj]
                    sum_temp = sum_temp / (ii)
                    
                    sma_temp[ll].append(sum_temp)
    
                if (ii >= period[ll]) : 
                    sum_temp = 0
                    for jj in range(ii - period[ll], ii) : 
                        sum_temp += y[jj]
                    sum_temp = sum_temp / (period[ll])
                    
                    sma_temp[ll].append(sum_temp)
        
        self.value  = sma_temp 
        self.period = period 
        self.number = nsma
        self.offset = offset 
        
        
    def plot(self, xsize = 8, ysize = 6) : 
        """
        Fonction permettant de tracer l'indicateur TR

        Parameters
        ----------
        xsize : TYPE, optional
            DESCRIPTION. The default is 8.
        ysize : TYPE, optional
            DESCRIPTION. The default is 6.

        Returns
        -------
        None.

        """
        plt.figure(figsize=(xsize, ysize))
        for ll in range(self.number) : 
            plt.plot(self.value[ll], ls='-', lw = 2)
        plt.show()
        plt.close()
    
    
    def update(self, main_last_y) : 
        sma_temp = self.value 
        period   = self.period 
        nsma     = self.number
        offset   = self.offset 
        
        for ll in range(nsma) : 
            
            if (offset[ll] != 0) : 
                last_y = main_last_y[:-offset[ll]].copy()
            else : 
                last_y = main_last_y.copy()
            
            sum_temp = 0 
            for jj in range(len(last_y) - period[ll], len(last_y)) : 
                sum_temp += last_y[jj]
            sum_temp = sum_temp / period[ll] 
            
            
            sma_temp[ll].append(sum_temp)
        
            self.value[ll] = sma_temp[ll]
            del self.value[ll][0]
        
# Experimental tests 
# import datetime as dt 
# import sys 
        
# sys.path.append("../operators/")
# import dataReader as dtr 

# data = dtr.loadData(start_time = dt.datetime(2012, 3, 18, 10, 12), 
#                     stop_time  = dt.datetime(2012, 4, 25, 11, 18))


# for ii in range (101, len(data)-1) : 
#     sub_data = data[ii-100:ii]
    
#     if (ii == 101) : 
#         sma = MULTSMA(sub_data["askclose"], 
#                       period = list(np.linspace(2, 95, 20, dtype=int)), 
#                       offset = list(np.linspace(0, 20, 20, dtype=int)))
#         sma.plot() 

#     if (ii > 101) : 
#         sma.update(sub_data["askclose"])
#         sma.plot()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
        
        