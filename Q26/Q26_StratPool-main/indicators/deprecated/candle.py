#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:51:58 2020

@author: lbrahimi
"""

import numpy as np
import matplotlib.pyplot as plt 
import datetime as dt 
import pandas as pd 
import sys 






class HOUR : 
    
    def __init__(self, hour = None, minute = None) : 
        self.hour   = hour 
        self.minute = minute



def from_data_to_Daily(sub_data) : 
    """
    This function cut the high precision data into one days packets. These packets 
    are returned in a list. 

    Parameters
    ----------
    sub_data : TYPE Pandas dataframe
        DESCRIPTION.

    Returns
    -------
    ddaily_data : TYPE list 
        Each element of the list is a Pandas dataframe containing all the data of a day 
    """
    start_time = sub_data.index[0]
    end_time   = sub_data.index[-1]
    daily_data = [] 
    time = dt.datetime(start_time.year, start_time.month, start_time.day)
    # print (time)
    # print (sub_data)
    while (time <= dt.datetime(end_time.year, end_time.month, end_time.day)) : 
        daily_data.append(sub_data[str(time).replace(" ","").replace("00:00:00","")])
        time += dt.timedelta(days=1)
    ddaily_data = []
    for ti in range(len(daily_data)) : 
        if (len(daily_data[ti]) > 0) : 
            ddaily_data.append(daily_data[ti])
    return ddaily_data 

def from_daily_data(sub_data) : 
    ddaily_data = []
    for ti in range(len(sub_data)) : 
        ddaily_data.append(sub_data.iloc[ti])
    return ddaily_data 

def sample_data_from_daily(daily_data, data_length = dt.timedelta(minutes = 60)) : 
    """
    This function allows to sample daily data in small packets by automatically detecting the 
    open and the close of the day. It is usefull for intraday_data 

    Parameters
    ----------
    daily_data : TYPE
        DESCRIPTION.
    data_length : TYPE, optional
        DESCRIPTION. The default is dt.timedelta(minutes = 60).

    Returns
    -------
    ssampled_data : TYPE
        DESCRIPTION.

    """
    sampled_data = []
    time_open    = [] 
    time_close   = []

    for ti in range(len(daily_data)) : 
        time_start = daily_data[ti].index[0] 
        while (time_start + data_length <= daily_data[ti].index[-1]) : 
            sampled_data.append(daily_data[ti].between_time(str(time_start.time()), str((time_start + data_length).time()), include_end = False))
            time_open.append(time_start)
            time_close.append(time_start + data_length - dt.timedelta(minutes = 1))
            time_start += data_length
        else : 
            sampled_data.append(daily_data[ti].between_time(str(time_start.time()), str(daily_data[ti].index[-1].time()), include_end = False))
            time_open.append(time_start) 
            time_close.append(daily_data[ti].index[-1] - dt.timedelta(minutes = 1))
            # pass
    
    ssampled_data = []
    ttime_open    = []
    ttime_close   = []
    for ti in range(len(sampled_data)) : 
        if (len(sampled_data[ti]) > 0) : 
            ssampled_data.append(sampled_data[ti])
            ttime_open.append(time_open[ti]) 
            ttime_close.append(time_close[ti])
            
    return ssampled_data, ttime_open, ttime_close 




def get_candle_length(timeframe) : 
    # if (initial_timeframe == "M1" or initial_timeframe == "D1") : 
    if (timeframe == "M1")   : return dt.timedelta(minutes = 1) 
    if (timeframe == "M2")   : return dt.timedelta(minutes = 2) 
    if (timeframe == "M3")   : return dt.timedelta(minutes = 3) 
    if (timeframe == "M5")   : return dt.timedelta(minutes = 5) 
    if (timeframe == "M10")  : return dt.timedelta(minutes = 10) 
    if (timeframe == "M15")  : return dt.timedelta(minutes = 15) 
    if (timeframe == "M20")   : return dt.timedelta(minutes = 20) 
    if (timeframe == "M30")  : return dt.timedelta(minutes = 30) 
    if (timeframe == "M45")  : return dt.timedelta(minutes = 45) 
    if (timeframe == "H1")   : return dt.timedelta(minutes = 60) 
    if (timeframe == "H2")   : return dt.timedelta(minutes = 120) 
    if (timeframe == "H3")   : return dt.timedelta(minutes = 180) 
    if (timeframe == "H4")   : return dt.timedelta(minutes = 240) 
    if (timeframe == "H6")   : return dt.timedelta(minutes = 360) 
    if (timeframe == "H8")   : return dt.timedelta(minutes = 480) 
    if (timeframe == "H10")  : return dt.timedelta(minutes = 600) 
    if (timeframe == "H12")  : return dt.timedelta(minutes = 720) 
    if (timeframe == "D1")   : return dt.timedelta(minutes = 1440)
        
    
def is_higher(t, ini_t) : 
    t_time = get_candle_length(t)
    t_ini_t = get_candle_length(ini_t) 
    if (t_time > t_ini_t) : 
        return True 
    else : 
        return False 
        
def get_last_index_of_day_before(value) :
    i = -1 
    time = value["tclose"].iloc[i]
    start_day = time.day 
    day = time.day
    
    while (day == start_day) : 
        i -= 1
        time = value["tclose"].iloc[i]
        day = time.day 
    
    return i 

def get_last_index_of_day_after(value) :
    i = 0
    time = value.index[i]
    start_day = time.day 
    day = time.day
    
    while (day == start_day) : 
        i += 1
        time = value.index[i]
        day = time.day 
    
    return i 



class CANDLE : 
    
    def __init__(self,
                 sub_data, 
                 initial_timeframe     = "M1",
                 timeframe             = "H1") : 
        
        
        data_length = get_candle_length(timeframe)
        
        if (is_higher(timeframe, initial_timeframe)) : 
            daily_data = from_data_to_Daily(sub_data)  
            higher = True 
        else : 
            daily_data = from_daily_data(sub_data) 
            higher = False 
            # print (daily_data[0].index[0])

        
        ttime_open = []
        ttime_close = []
        if (timeframe == "D1") : 
            sampled_data = daily_data 
            # print(sampled_data[0])
            for ti in range(len(sampled_data)) : 
                ttime_open.append(sampled_data[ti].index[0])
                ttime_close.append(sampled_data[ti].index[-1])
            
                
        if (timeframe == "M1"  or 
            timeframe == "M2"  or 
            timeframe == "M3"  or
            timeframe == "M5"  or
            timeframe == "M10" or
            timeframe == "M15" or
            timeframe == "M20" or 
            timeframe == "M30" or
            timeframe == "M45" or
            timeframe == "H1"  or
            timeframe == "H2"  or
            timeframe == "H3"  or
            timeframe == "H4"  or
            timeframe == "H6"  or
            timeframe == "H8"  or
            timeframe == "H10" or
            timeframe == "H12") : 
            sampled_data, ttime_open, ttime_close = sample_data_from_daily(daily_data, data_length = data_length)
            
        askhigh  = []
        asklow   = []
        askopen  = []
        askclose = []  
        time     = []
        tclose   = []
        number   = []
        
        # print (sampled_data[0])
        # print (sampled_data[0].date)
        for ti in range(len(sampled_data)) : 
            number.append(ti)
            # time.append(ttime_open[ti])
            # print (len(sampled_data[ti]))
            if (higher) : 
                # print ("Hello")
                time.append(sampled_data[ti].index[0])
                tclose.append(sampled_data[ti].index[-1])
                
                askopen.append(sampled_data[ti]["askopen"].iloc[0])
                
                askclose.append(sampled_data[ti]["askclose"].iloc[-1])
                
                asklow.append(min(sampled_data[ti]["asklow"]))
                
                askhigh.append(max(sampled_data[ti]["askhigh"]))
            else : 
                time.append(sampled_data[ti].name)
                tclose.append(sampled_data[ti].name)
                
                askopen.append(sampled_data[ti].askopen)
                
                askclose.append(sampled_data[ti].askclose)
                
                asklow.append(sampled_data[ti].asklow)
                
                askhigh.append(sampled_data[ti].askhigh)
                
        
        
        data_new = pd.DataFrame({"number"   :number,
                                 "askopen"  :askopen,
                                 "askclose" :askclose,
                                 "asklow"   :asklow,
                                 "askhigh"  :askhigh,
                                 "tclose"   : tclose},
                                  index = time)
        
        
        self.ini_value        = sub_data
        self.value            = data_new 
        self.length           = data_length
        self.timeframe        = timeframe 
        self.initial_timeframe= initial_timeframe
        
    
    
    def update(self, new_data, constrain_axis = True) : 
        """
        Update function, it is working. 

        Parameters
        ----------
        new_data : TYPE
            DESCRIPTION.
        constrain_axis : TYPE, optional
            This parameter allow to suppress edges data according to the edges of the new dataset. The default is True.

        Returns
        -------
        None.

        """
        
        self.ini_value        = new_data
        
        timedelta = get_candle_length(self.initial_timeframe)
        
        # We retrieve the tmin and tmax of the new data
        tnew_open  = new_data.index[0]  
        tnew_close = new_data.index[-1] 
        
        t_open  = self.value.index[get_last_index_of_day_after(self.value)] - timedelta
        t_close = self.value["tclose"].iloc[get_last_index_of_day_before(self.value)] + timedelta 

        
        
        
        # There are different cases 
        if (tnew_close > t_close) :             
            # We retrive the tmin and tmax of the actual data 
            t_open  = self.value.index[0]
            t_close = self.value["tclose"].iloc[get_last_index_of_day_before(self.value)] + timedelta 
            
            # We add the right hand data 
            temp_ini_data = self.ini_value.loc[t_close:tnew_close] 
            temp_data = CANDLE(temp_ini_data, initial_timeframe = self.initial_timeframe, timeframe = self.timeframe).value
            self.value = pd.concat([self.value.loc[:t_close], temp_data])
            
            if (constrain_axis) : 
                max_time = self.value.index[0]
                for time in self.value.index : 
                    if (tnew_open > time) : 
                        max_time = time
                self.value = self.value.loc[max_time:]
                
            
        if (tnew_open < t_open)   :             
            # We retrive the tmin and tmax of the actual data 
            t_open  = self.value.index[get_last_index_of_day_after(self.value)] - timedelta
            t_close = self.value["tclose"].iloc[-1]
            
            # We add the left hand data 
            temp_ini_data = self.ini_value.loc[tnew_open:t_open - timedelta]
            temp_data = CANDLE(temp_ini_data, initial_timeframe = self.initial_timeframe, timeframe = self.timeframe).value
            self.value = pd.concat([temp_data, self.value.loc[t_open:]])
            
            if (constrain_axis) : 
                min_time = self.value.index[-1] 
                for time in self.value.index[::-1] : 
                    if (tnew_close < time) : 
                        print ("hello !!!")
                        min_time = time 
                self.value = self.value.loc[:min_time]
                

                
            
        






# Experimental tests ! 
# import matplotlib as mpl
# import matplotlib.gridspec as gridspec
# mpl.rc('figure', max_open_warning = 0)

# sys.path.append("../operators/")
# import dataReader as dtr 
# data = dtr.loadData(path = "/media/lbrahimi/FREECOM HDD/TRAVAIL/QUANTUMS/HISTORICAL_DATA/FINNHUB_DATA/STOCKS/US exchanges/FB_2019-2020.csv",
#                     start_time = dt.datetime(2019, 8, 21, 0, 12), 
#                     stop_time  = dt.datetime(2020, 4, 23, 11, 18), 
#                     open_close = {"open":dt.time(hour = 12, minute = 0), "close" : dt.time(hour = 20, minute = 0)}, 
#                     time_shift = dt.timedelta(hours = 2))

            

# i_start = 10000
# size = 1000
# sub_data = data[i_start : i_start + size]
# max_val = 5000
# min_val = 0
# delta = 100

# cdle_2 = CANDLE(sub_data, 
#               initial_timeframe = "M1",
#               timeframe         = "H2")

# # while (i_start + size < max_val) : 
# while (i_start + size > min_val) : 

#     sub_data = data[i_start : i_start + size]
#     cdle_1 = CANDLE(sub_data, 
#                   initial_timeframe = "M1",
#                   timeframe         = "H2")
#     cdle_2.update(sub_data, constrain_axis = True)
    
    
    
    
#     fig = plt.figure(figsize=(20, 15)) 
#     gs = gridspec.GridSpec(ncols= 1, nrows = 2, figure = fig)
#     gs.update(wspace=0.075, hspace=0.2) # set the spacing between axes.
    
    
#     ax0 = fig.add_subplot(gs[0])
#     ax0.set_title("NO UPDATED")
#     ax0.plot(cdle_1.ini_value.index, cdle_1.ini_value["asklow"], c="grey")
#     ax0.plot(cdle_1.ini_value.index, cdle_1.ini_value["askhigh"], c="grey")
#     color = None
#     for ii in range(0, len(cdle_1.value)) : 
#         if (cdle_1.value["askclose"].iloc[ii] >= cdle_1.value["askopen"].iloc[ii]) : 
#             color = "blue"
#         else : 
#             color = "red"
        
#         dtime = 0.5*(cdle_1.length)
#         ax0.plot([cdle_1.value.index[ii] + dtime, cdle_1.value.index[ii] + dtime],
#                   [cdle_1.value["asklow"].iloc[ii], cdle_1.value["askhigh"].iloc[ii]], 
#                   lw= 0.5 , color="black")
        
#         ax0.fill_between([cdle_1.value.index[ii], cdle_1.value["tclose"].iloc[ii]], 
#                           cdle_1.value["askopen"].iloc[ii], cdle_1.value["askclose"].iloc[ii], 
#                           where=True, color=color, alpha=1)
        
        
#     ax1 = fig.add_subplot(gs[1])
#     ax1.set_title("UPDATED")
#     ax1.plot(cdle_2.ini_value.index, cdle_2.ini_value["asklow"], c="grey")
#     ax1.plot(cdle_2.ini_value.index, cdle_2.ini_value["askhigh"], c="grey")
#     color = None
#     for ii in range(0, len(cdle_2.value)) : 
#         if (cdle_2.value["askclose"].iloc[ii] >= cdle_2.value["askopen"].iloc[ii]) : 
#             color = "blue"
#         else : 
#             color = "red"
        
#         dtime = 0.5*(cdle_2.length)
#         ax1.plot([cdle_2.value.index[ii] + dtime, cdle_2.value.index[ii] + dtime],
#                   [cdle_2.value["asklow"].iloc[ii], cdle_2.value["askhigh"].iloc[ii]], 
#                   lw= 0.5 , color="black")
        
#         ax1.fill_between([cdle_2.value.index[ii], cdle_2.value["tclose"].iloc[ii]], 
#                           cdle_2.value["askopen"].iloc[ii], cdle_2.value["askclose"].iloc[ii], 
#                           where=True, color=color, alpha=1)
    
    
    
#     plt.show()
    
#     # i_start += delta
#     i_start -= delta



            
            
            