#!/usr/local/bin/env python3.7
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
#  The MIT License (MIT)
#  Copyright (c) 2020 Philippe Ostiguy
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################

"""Helper module to manipulate csv and pandas Dataframe"""
import csv
import pandas as pd
import datetime as dt
from functools import wraps
from statsmodels.tsa.stattools import adfuller
import os.path
import numpy as np

def ordinal_date(function):
        """Wrapper to add an ordinal date"""
        @wraps(function)
        def wrapper(cls,date_name,date_debut,date_fin, name_,directory,asset, ordinal_name,is_fx,dup_col):
            series_ = function(cls,date_name,date_debut,date_fin, name_,directory,asset, ordinal_name,is_fx,dup_col)
            series_.Date = pd.to_datetime(series_.Date)
            series_[ordinal_name] = pd.to_datetime(series_[date_name]).map(dt.datetime.toordinal)
            return series_

        return wrapper

class ManipData():
    """Class to manipulate data"""

    @classmethod
    def __init__(cls,dir_,file_name,extension =""):
        cls.dir_ = dir_ #directory
        cls.filename = file_name
        cls.extension_ = extension #if there is an extension added to the filename

    @classmethod
    def write_csv_(cls, dir_output, name_out, add_doc = "", is_walkfoward = False, **kwargs):
        """ Write data to a csv

        Parameters
        ----------
        dir_output : str
            directory where we want our data to be written
        name_out : str
            name of the file name
        is_walkfoward : bool
            says if we are doing a walkfoward analyis. If `True`, we have to create a separate training and test file
        **kwargs : keyword param
            dictionary with keys and items to be written in the file
        """

        if is_walkfoward:
            write_type = 'a'
            func = 'writer.writerow'
        else :
            write_type = 'w'
            func = 'str'

        with open(dir_output + name_out + add_doc + ".csv" , write_type, newline='') as f:
            writer = csv.writer(f)
            eval(func)('')
            for key, item in kwargs.items():
                writer.writerow([key,item])


    @classmethod
    def erase_content(cls):
        """Method to erase contents of a csv file"""
        filename = cls.dir_ + cls.filename + cls.extension_ + ".csv"
        if os.path.isfile(filename):
            with open(filename,"w+") as f:
                f.close()

    @classmethod
    @ordinal_date
    def csv_to_pandas(cls, date_name,date_debut,date_fin, name_,directory,asset, ordinal_name = '',is_fx = False,
                      dup_col = None):
        """Return the csv to a pandas Dataframe

        The function remove nan value with `series_.dropna()` and remove the data when the market is closed with
        `series_.drop_duplicates()`
        """

        if is_fx:
            dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S')
        else :
            dateparse = None
        series_ = pd.DataFrame()
        _series = pd.read_csv(directory
                               + asset + '.csv', usecols=list(name_.columns),parse_dates=[date_name],
                              date_parser=dateparse)
        series_ =_series.loc[(_series[date_name] >= date_debut) & (_series[date_name] < date_fin)]
        if series_.empty:
            raise Exception("Desired date range not available in the current files")

        series_ = series_.dropna() #drop nan values
        if dup_col != None:
            #If all values in column self.dup_col are the same, we erase them
            series_ = series_.drop_duplicates(keep=False,subset=list(dup_col.keys()))
        series_=series_.reset_index(drop=True)

        return series_

    @classmethod
    def sous_series_(cls,series_,nb_data,point_data=0):
        """Returns a sub-series to calculate the value of the indicator with in a precise"""

        cls.sous_series=series_.iloc[point_data:point_data + nb_data,:]
        if nb_data > len(series_):
            raise Exception("Not enough necessary data to calculate the indicator")
        return cls.sous_series

    @classmethod
    def de_trend(cls, series, date_name, date_ordinal_name, default_data, period =1, p_value = .05):
        """Remove the trend from the series by differentating the current serie

        First value is set to 0 to avoid error (of the differentiated serie)

        Parameters
        ---------
        `period` : int
            Number of periods used for differencing. Default : first difference

        Return
        ------
        `series_diff` : Pandas Dataframe
            The stationary series

        """

        series_diff = series.copy()
        series_diff.drop([date_name, date_ordinal_name], axis=1, inplace=True)
        series_diff = series_diff.diff(periods=period)  # differencing with previous row
        series_diff.loc[:(period - 1), :] = 0 #Make first row equal to 0
        series_diff.insert(0, date_name, series[date_name]) #re-insert the peiod columns
        series_diff[date_ordinal_name] = series[date_ordinal_name]
        if adfuller(series_diff[default_data])[1] > p_value:
            raise Exception("The series is not stationary")
        return series_diff

    @classmethod
    def nan_list(cls,list_):
        """Check if a list has one empty value

        Return
        ------
        Bool : `True` or `False`
            Return `True` if at least one value in the list is `nan` and `False otherwise
        """

        return True if True in np.isnan(list_) else False

    @classmethod
    def pd_tolist(cls,pd_, row_name):
        """Transform a pandas column to a list. It makes sure it is an integer"""
        pd__ = pd_.loc[:, row_name].tolist()
        try:
           t = [int(i) for i in pd__]
        except:
            raise Exception("Mistake happened in pd_tolist")
        else :
            return [int(i) for i in pd__]