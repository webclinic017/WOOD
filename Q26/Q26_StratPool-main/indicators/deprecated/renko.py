#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:47:53 2020

@author: lbrahimi
"""
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import average_true_range as atr 


class ATR_RENKO : 
    
    def __init__(self, high, low, close, source_date, source_prices, atr_period = 10, max_stairs = 3) : 
        """
        ATR renko initializator. 

        Parameters
        ----------
        source_date : datetime list 
            DESCRIPTION.
        source_prices : double list
            DESCRIPTION.
        brick_size : FLOAT, optional
            Brick size of the indicator. The default is 0.002.
        max_stairs : INT, optional
            Max number of stairs for the blocks. The default is 3.

        Returns
        -------
        None.

        """
        
        renko_date       = [source_date[0]  ]
        renko_prices     = [source_prices[0]]
        renko_directions = [0               ]
        TR  = atr.TR(high, low, close)
        ATR = atr.ATR(TR, atr_period)
        
        for ii in range(1, len(source_prices)) : 
            last_price = source_prices[ii]
            gap_div = int(float(last_price - renko_prices[-1]) / ATR.value[ii])
            
            if (gap_div != 0) : 
                
                if (np.abs(gap_div) == 1) : 
                    renko_date.append(source_date[ii])
                    renko_prices.append(renko_prices[-1] + gap_div * ATR.value[ii])
                    renko_directions.append(np.sign(gap_div))
                    
                if (np.abs(gap_div) == 2) : 
                    renko_date.append(source_date[ii])
                    renko_prices.append(renko_prices[-1] + gap_div * ATR.value[ii])
                    renko_directions.append(np.sign(gap_div))
                    
                if (np.abs(gap_div) >= 3) : 
                    renko_date.append(source_date[ii])
                    renko_prices.append(renko_prices[-1] + np.sign(gap_div) * 3 * ATR.value[ii])
                    renko_directions.append(np.sign(gap_div))
                    
        
        self.date       = renko_date
        self.prices     = renko_prices
        self.directions = renko_directions
        self.atr_value  = ATR
        self.max_stairs = max_stairs    


    def plot(self, xsize = 8, ysize = 6, data = None) : 
        fig = plt.figure(figsize=(xsize, ysize))
        gs = gridspec.GridSpec(ncols= 1, nrows = 2, figure = fig , height_ratios=[3, 1])
        gs.update(wspace=0.075, hspace=0.2) # set the spacing between axes. 
        
        ax0 = fig.add_subplot(gs[0])
        if (not(data.index[0] == None)) : 
            ax0.plot(data["askclose"], lw=1, color = "black")
        color = "green"
        for ii in range(1, len(self.prices)) : 
            if (self.directions[ii] == 1) : 
                color = "green"
            else : 
                color = "red"
            ax0.fill_between([self.date[ii-1], self.date[ii]], self.prices[ii-1], self.prices[ii], 
                            where=True, color=color, alpha=0.6)
        
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(self.atr_value.value, c="black", lw = 2)
        
        # fig.show()



    def update(self, high_i, low_i, close_im, last_date, last_price, first_date = None) : 
        """
        Simple Renko update function

        Parameters
        ----------
        last_date : DATETIME
            CURRENT TIME OF THE SIMULATION
        last_price : FLOAT
            DESCRIPTION.
        first_date : DATETIME, optional
            FIRST DATE OF THE CURRENT DATA VECTOR. The default is None.
            We need it in order to suppress old renko values

        Returns
        -------
        None.

        """
        
        renko_date       = self.date
        renko_prices     = self.prices
        renko_directions = self.directions
        max_stairs       = self.max_stairs
        ATR              = self.atr_value
        
        
        ATR.update(high_i, low_i, close_im)
        
        
        gap_div = int(float(last_price - renko_prices[-1]) / ATR.value[-1])
        
        if (gap_div != 0) : 
            
            if (np.abs(gap_div) == 1) :
                renko_date.append(last_date)
                renko_prices.append(renko_prices[-1] + gap_div * ATR.value[-1])
                renko_directions.append(np.sign(gap_div))
                
            if (np.abs(gap_div) == 2) :
                renko_date.append(last_date)
                renko_prices.append(renko_prices[-1] + gap_div * ATR.value[-1])
                renko_directions.append(np.sign(gap_div))
                
            if (np.abs(gap_div) >= 3) :
                renko_date.append(last_date)
                renko_prices.append(renko_prices[-1] + np.sign(gap_div) * 3 * ATR.value[-1])
                renko_directions.append(np.sign(gap_div))
        
        if (first_date) : 
            while(renko_date[0] < first_date) : 
                del renko_date[0]
                del renko_prices[0]
                del renko_directions[0]
        
        self.date       = renko_date
        self.prices     = renko_prices
        self.directions = renko_directions
        self.atr_value  = ATR










# Experimental tests 
# import sys 
# import datetime as dt 
# import time as ti        
        
# sys.path.append("../operators/")
# import dataReader as dtr 
# import candle as can


# data = dtr.loadData(path="../../../dukascopyData/", name="AAPL", timeframe="m1", ext="csv",
#               start_time = dt.datetime(2017, 3, 18, 10, 12), stop_time  = dt.datetime(2019, 4, 25, 11, 18),
#               time_format = '%d.%m.%Y %H:%M:%S.%f', time_shift = dt.timedelta(minutes=0)) 

# data2 = can.sampleCandle(data, timeframe="H1")


# # rk1 = ATR_RENKO(data2["askhigh"], data2["asklow"], data2["askclose"],
# #                 data2.index, data2["askclose"], atr_period = 100, max_stairs = 1)
# # rk1.plot(data = data2)
        
# for ii in range (201, len(data2)-1) : 
#     sub_data = data2[ii-200:ii]
    
#     if (ii == 201) : 
#         # rk1 = SIMPLE_RENKO(sub_data.index, sub_data["askclose"], brick_size=0.002, max_stairs = 3)
#         rk1 = ATR_RENKO(sub_data["askhigh"], sub_data["asklow"], sub_data["askclose"], 
#                         sub_data.index, sub_data["askclose"], atr_period = 60, max_stairs = 3)
#         rk1.plot(data=sub_data)
    
#     if (ii > 201) : 
#         # rk1.update(sub_data.index[-1], sub_data["askclose"].iloc[-1],
#         #             first_date = sub_data.index[0])
#         rk1.update(sub_data["askhigh"].iloc[-1], sub_data["asklow"].iloc[-1], sub_data["askclose"].iloc[-2], 
#                     sub_data.index[-1], sub_data["askclose"].iloc[-1], first_date = sub_data.index[0])
#         rk1.plot(data=sub_data, xsize=12, ysize=8)
    
#     print("Plot n°"+str(ii-201))
    
#     # ti.sleep(0.5)

        
        
        
        