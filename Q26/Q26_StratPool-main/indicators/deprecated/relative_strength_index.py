#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:46:10 2020

@author: lbrahimi
"""

import matplotlib.pyplot as plt 
import exponential_moving_average as ema 
import rexponential_moving_average as rma 
import simple_moving_average as sma
import numpy as np 
        


class RSI_MQ4 : 
    
    def __init__(self, close, period = 14) : 
        
        
        y_up   = np.zeros(len(close))
        y_down = np.zeros(len(close))
        rsi    = np.zeros(len(close))
        
        # Calcul préliminaire à la position = period 
        sump = 0.
        sumn = 0.
        for ii in range(1, period+1, 1) : 
            diff = close[ii] - close[ii-1]
            if (diff > 0.) : 
                sump += diff
            else : 
                sumn -= diff 
        y_up[period]   = sump/period
        y_down[period] = sumn/period
        if (y_down[period] != 0.) : 
            rsi[period] = 100. - 100./(1 + y_up[period]/y_down[period])
        else : 
            if (y_up[period] != 0) : 
                rsi[period] = 100. 
            else : 
                rsi[period] = 50. 
        
        # Calcul continu après la position = period 
        for ii in range(period+1, len(rsi)) : 

            diff = close[ii] - close[ii-1]
            if (diff > 0) : 
                y_up[ii]   = (y_up[ii-1]*(period - 1) +  diff)/period
                y_down[ii] = (y_down[ii-1]*(period - 1) + 0.)/period
            if (diff <= 0) : 
                y_up[ii]   = (y_up[ii-1]*(period - 1) +  0.)/period
                y_down[ii] = (y_down[ii-1]*(period - 1) - diff)/period
                
            if (y_down[ii] != 0.) : 
                rsi[ii] = 100. - 100./(1 + y_up[ii]/y_down[ii])
            else : 
                if (y_up[ii] != 0) : 
                    rsi[ii] = 100. 
                else : 
                    rsi[ii] = 50. 
        
        self.value  = rsi 
        self.up     = y_up
        self.down   = y_down 
        self.period = period
        



    def plot(self) : 
        
        plt.figure()
        plt.plot(self.value)
        plt.show()
        plt.close()





















class RSI_SMA : 
    
    def __init__(self, y_open, y_close, period = 14) : 
        
        
        y_up   = np.zeros(len(y_open))
        y_down = np.zeros(len(y_open))
        for ii in range(1, len(y_open)) : 
            # # Bear
            # if (y_open[ii] > y_close[ii]) : 
            #     y_down[ii] = abs(y_open[ii] - y_close[ii])
            # # Bull
            # if (y_open[ii] < y_close[ii]) : 
            #     y_up[ii] = abs(y_open[ii] - y_close[ii])
            # Bear
            if (y_close[ii-1] > y_close[ii]) : 
                y_down[ii] = abs(y_close[ii-1] - y_close[ii])
            # Bull
            if (y_close[ii-1] < y_close[ii]) : 
                y_up[ii] = abs(y_close[ii-1] - y_close[ii])
        
        sma_up    = sma.SMA(y_up, period = period)
        sma_down  = sma.SMA(y_down, period = period)
        
        rs = np.array(sma_up.value)/np.array(sma_down.value) 
        rsi = 100. - 100./(1 + rs)
        
        self.value      = rsi
        self.sma_up   = sma_up
        self.sma_down = sma_down
        self.period   = period 
    
    
    def plot(self) : 
        
        plt.figure()
        plt.plot(self.value)
        plt.show()
        plt.close()
        
    
    
    
    
    def update(self, last_open, last_close) : 
        
        # rsi = self.rsi
        sma_up = self.sma_up
        sma_down = self.sma_down
        
        y_up   = np.zeros(len(last_open))
        y_down = np.zeros(len(last_open))
        for ii in range(len(last_open)) : 
            # Bear
            if (last_open[ii] > last_close[ii]) : 
                y_down[ii] = abs(last_open[ii] - last_close[ii])
            # Bull
            if (last_open[ii] < last_close[ii]) : 
                y_up[ii] = abs(last_open[ii] - last_close[ii])
        
        sma_down.update(y_down)
        sma_up.update(y_up)
        
        rs = np.array(sma_up.value)/np.array(sma_down.value) 
        rsi = 100. - 100./(1 + rs)
        
        self.sma_up   = sma_up
        self.sma_down = sma_down
        self.value = rsi 
        
class RSI_SMA2 : 
    
    def __init__(self, y_open, y_close, period = 14) : 
        
        
        y_up   = np.zeros(len(y_open))
        y_down = np.zeros(len(y_open))
        for ii in range(len(y_open)) : 
            # # Bear
            # if (y_open[ii] > y_close[ii]) : 
            #     y_down[ii] = abs(y_open[ii] - y_close[ii])
            # # Bull
            # if (y_open[ii] < y_close[ii]) : 
            #     y_up[ii] = abs(y_open[ii] - y_close[ii])
            # Bear
            if (y_open[ii] > y_close[ii]) : 
                y_down[ii] = abs(y_open[ii] - y_close[ii])
            # Bull
            if (y_open[ii] < y_close[ii]) : 
                y_up[ii] = abs(y_open[ii] - y_close[ii])
        
        sma_up    = sma.SMA(y_up, period = period)
        sma_down  = sma.SMA(y_down, period = period)
        
        rs = np.array(sma_up.value)/np.array(sma_down.value) 
        rsi = 100. - 100./(1 + rs)
        
        self.value      = rsi
        self.sma_up   = sma_up
        self.sma_down = sma_down
        self.period   = period 
    
    
    def plot(self) : 
        
        plt.figure()
        plt.plot(self.value)
        plt.show()
        plt.close()
        
    
    
    
    
    def update(self, last_open, last_close) : 
        
        # rsi = self.rsi
        sma_up = self.sma_up
        sma_down = self.sma_down
        
        y_up   = np.zeros(len(last_open))
        y_down = np.zeros(len(last_open))
        for ii in range(len(last_open)) : 
            # Bear
            if (last_open[ii] > last_close[ii]) : 
                y_down[ii] = abs(last_open[ii] - last_close[ii])
            # Bull
            if (last_open[ii] < last_close[ii]) : 
                y_up[ii] = abs(last_open[ii] - last_close[ii])
        
        sma_down.update(y_down)
        sma_up.update(y_up)
        
        rs = np.array(sma_up.value)/np.array(sma_down.value) 
        rsi = 100. - 100./(1 + rs)
        
        self.sma_up   = sma_up
        self.sma_down = sma_down
        self.value = rsi 


class RSI_EMA : 
    
    def __init__(self, y_open, y_close, period = 14) : 
        
        
        y_up   = np.zeros(len(y_open))
        y_down = np.zeros(len(y_open))
        for ii in range(len(y_open)) : 
            # Bear
            if (y_open[ii] > y_close[ii]) : 
                y_down[ii] = abs(y_open[ii] - y_close[ii])
            # Bull
            if (y_open[ii] < y_close[ii]) : 
                y_up[ii] = abs(y_open[ii] - y_close[ii])
        
        ema_up    = ema.EMA(y_up, period = period)
        ema_down  = ema.EMA(y_down, period = period)
        
        rs = np.array(ema_up.value)/np.array(ema_down.value) 
        rsi = 100. - 100./(1 + rs)
        
        self.value      = rsi
        self.ema_up   = ema_up
        self.ema_down = ema_down
        self.period   = period 
    
    
    def plot(self) : 
        
        plt.figure()
        plt.plot(self.value)
        plt.show()
        plt.close()
        
    
    
    
    
    def update(self, last_open, last_close) : 
        
        # rsi = self.rsi
        ema_up = self.ema_up
        ema_down = self.ema_down
        
        y_up   = 0.
        y_down = 0.
        
        if (last_open > last_close) : 
            y_down = abs(last_open - last_close)
        if (last_open <= last_close) : 
            y_up = abs(last_open - last_close)
        
        ema_down.update(y_down)
        ema_up.update(y_up)
        
        rs = np.array(ema_up.value)/np.array(ema_down.value) 
        rsi = 100. - 100./(1 + rs)
        
        self.ema_up   = ema_up
        self.ema_down = ema_down
        self.value = rsi 
        
        

class RSI_RMA : 
    
    def __init__(self, y_open, y_close, period = 14) : 
        
        
        y_up   = np.zeros(len(y_open))
        y_down = np.zeros(len(y_open))
        for ii in range(1, len(y_open)) : 
            # Bear
            # if (y_open[ii] > y_close[ii]) : 
            if (y_close[ii-1] > y_close[ii]) :
                # y_down[ii] = abs(y_open[ii] - y_close[ii])
                y_down[ii] = abs(y_close[ii-1] - y_close[ii])
            # Bull
            # if (y_open[ii] < y_close[ii]) : 
            if (y_close[ii-1] < y_close[ii]) :
                # y_up[ii] = abs(y_open[ii] - y_close[ii])
                y_up[ii] = abs(y_close[ii-1] - y_close[ii])
        
        ema_up    = rma.RMA(y_up, period = period)
        ema_down  = rma.RMA(y_down, period = period)
        
        rs = np.array(ema_up.value)/np.array(ema_down.value) 
        rsi = 100. - 100./(1 + rs)
        
        self.value      = rsi
        self.ema_up   = ema_up
        self.ema_down = ema_down
        self.period   = period 
    
    
    def plot(self) : 
        
        plt.figure()
        plt.plot(self.value)
        plt.show()
        plt.close()
        
    
    
    
    
    def update(self, last_open, last_close) : 
        
        # rsi = self.rsi
        ema_up = self.ema_up
        ema_down = self.ema_down
        
        y_up   = 0.
        y_down = 0.
        
        if (last_open > last_close) : 
            y_down = abs(last_open - last_close)
        if (last_open <= last_close) : 
            y_up = abs(last_open - last_close)
        
        ema_down.update(y_down)
        ema_up.update(y_up)
        
        rs = np.array(ema_up.value)/np.array(ema_down.value) 
        rsi = 100. - 100./(1 + rs)
        
        self.ema_up   = ema_up
        self.ema_down = ema_down
        self.value = rsi 




        
"""
# Experimental tests 
import datetime
import sys 
import time as ti 
import pandas as pd 
        

sys.path.append("../../comm/FINNHUB/")
import functions_FINNHUB


# contract = {
#     "Category" : "forex", 
#     "Instrument" : "candle",
#     "Symbol" : "OANDA:GBP_JPY"
#     }

contract = {
    "Category" : "stock", 
    "Instrument" : "candle",
    "Symbol" : "AA"
    }



data_theo = pd.read_csv("/home/loann/Bureau/AA.csv")
data_theo["askclose"] = data_theo["Close"]
data_theo["askopen"] = data_theo["Open"]

app = functions_FINNHUB.client.FINNHUBApp()
data1 = functions_FINNHUB.getHistoricalData(app,
                                            time_end            = datetime.datetime.today() - datetime.timedelta(minutes = 1), 
                                            duration            = datetime.timedelta(days = 100), 
                                            contract            = contract,
                                            timeframe           = "D1", 
                                            regularTradingHours = 1, 
                                            waiting_time        = 10, 
                                            output_format       = "dataframe", 
                                            last_candle_hour    = None)




# rsi_sma1 = RSI_SMA(data1["askopen"], data1["askclose"], period = 14)
# rsi_sma2 = RSI_SMA2(data1["askopen"], data1["askclose"], period = 14)

rsi_rma = RSI_RMA(data1["askopen"], data1["askclose"], period = 14)
# rsi_ema = RSI_RMA(data1["askopen"], data1["askclose"], period = 14)
rsi_rma_theo = RSI_RMA(data_theo["askopen"], data_theo["askclose"], period = 14)

import matplotlib.pyplot as plt 

# plt.plot(rsi_sma1.value, c="blue")
# plt.plot(rsi_sma2.value, c="red") 
plt.plot(rsi_rma.value[-30:], c = "blue")
plt.plot(rsi_rma_theo.value[-30:], c = "red")


# sys.path.append("../operators/")
# import dataReader as dtr 

# data = dtr.loadData(start_time = dt.datetime(2012, 3, 18, 10, 12), 
#                     stop_time  = dt.datetime(2012, 4, 25, 11, 18))


# for ii in range (301, len(data)-1) : 
#     sub_data = data[ii-300:ii]
    
#     if (ii == 301) : 
#         rsi_sma = RSI_SMA(sub_data["askopen"], sub_data["askclose"], period = 14)
#         rsi_mq4 = RSI_MQ4(sub_data["askclose"], period = 14)
#         # rsi.plot() 

#     if (ii > 301) : 
#         rsi_sma = RSI_SMA(sub_data["askopen"], sub_data["askclose"], period = 56)
#         rsi_mq4 = RSI_MQ4(sub_data["askclose"], period = 56)
#         # rsi_sma.update(sub_data["askopen"], sub_data["askclose"])
#         # rsi.plot()   
        
#         plt.figure()
#         plt.plot(rsi_sma.value, c="blue")
#         plt.plot(rsi_mq4.value, c="red")
""" 
        
        
    