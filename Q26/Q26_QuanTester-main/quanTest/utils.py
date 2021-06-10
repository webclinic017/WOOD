#!/usr/bin/env python3
# -*- coding: utf-8 -*- 


def compareHour(hour1, operator, hour2) : 
    """ 
    Function that allows to compare two hours 
    that are defined in string formats as : HH:MM
    """
    hour1    = hour1.split(":") 
    hour_1   = int(hour1[0]) 
    minute_1 = int(hour1[1])

    hour2    = hour2.split(":")
    hour_2   = int(hour2[0])
    minute_2 = int(hour2[1])

    if operator == ">=" : 

        if hour_1 > hour_2 : 
            return True 
        elif hour_1 < hour_2 : 
            return False 
        elif hour_1 == hour_2 : 
            if minute_1 > minute_2 : 
                return True 
            elif minute_1 < minute_2 : 
                return False 
            else : 
                return True 

    if operator == "<=" : 

        if hour_1 > hour_2 : 
            return False 
        elif hour_1 < hour_2 : 
            return True 
        elif hour_1 == hour_2 : 
            if minute_1 > minute_2 : 
                return False 
            elif minute_1 < minute_2 : 
                return True 
            else : 
                return False

    return 