#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:05:57 2018

@author: raviwijeratne
"""

import weather_prediction as wp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

predictor = wp.Predictor()
temps, year, month, day, date = predictor.get_data()

date_df = pd.DataFrame(
        {'month': month,
         'day': day})
temps_df = pd.DataFrame({'temps': temps})

date_array = np.array(date_df)
temps_array = np.array(temps_df)

X_train, X_test, Y_train, Y_test = train_test_split(date_array, temps_array, test_size=0.2, random_state = 12) 

values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
 

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

prediction = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % regressor.score(X_test, Y_test))
print("Mean Absolute Error: %.2f" % mean_absolute_error(Y_test, prediction))
print("Median Absolute Error: %.2f" % median_absolute_error(Y_test, prediction))

print X_test
print "today's temperature", regressor.predict(np.array([[20, 6]]))