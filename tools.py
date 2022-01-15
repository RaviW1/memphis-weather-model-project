#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: raviwijeratne
"""
import numpy as np
import matplotlib.pyplot as plt



def scatter(x, y):
    x_jitter = x + np.random.normal(size=x.size, scale=.5)
    y_jitter = y + np.random.normal(size=x.size, scale=.5)
    plt.plot(x_jitter, y_jitter, color='black', marker='.', linestyle='none', alpha=.05,)
    plt.show()
    
def find_day_of_year(year, month, day):
    days_per_month = np.array([
        31, #January
        28, #February
        31, #March
        30, #April
        31, #May
        30, #June
        31, #July
        31, #August
        30, #September
        31, #October
        30, #November
        31, #December
    ])
    if year % 4 == 0:
        days_per_month[1] += 1
    day_of_year = np.sum(np.array(days_per_month[:month - 1])) + day - 1
    return day_of_year

def find_autocorr(values, length=100):
    
        autocorr = []
        for shift in range(1, length):
            correlation = np.corrcoef(values[:-shift], values[shift:])[1, 0]
            autocorr.append(correlation)
        return autocorr