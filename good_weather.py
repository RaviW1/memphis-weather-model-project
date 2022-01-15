#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 08:30:44 2018

@author: raviwijeratne
"""
import sys
import weather_prediction as pw

def decide(test_year=2018, test_month=6, test_day=16, test_temp=86):
    predictor = pw.Predictor()
    prediction = predictor.predict(test_year, test_month, test_day, test_temp)
    print('For year =', test_year,
          ', month=', test_month,
          ', day=', test_day,
          ', temp=', test_temp)
    print('Predicted temprature is', prediction)
    
    
    if prediction >= 75:
        decision = True
    else:
        decision = False
    
    return decision
        

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Try it like this:')
        print('    python good_weather.py <year> <month> <day> <temp>')
    else: 
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        day = int(sys.argv[3])
        temp = float(sys.argv[4])
        decision = decide(
                test_year=year, test_month=month, test_day=day, test_temp=temp )        
        if decision == True:
            print("There will be good weather.")
        else:
            print("There will not be good weather.")