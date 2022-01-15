
# coding: utf-8

# In[257]:


import numpy as np
import warnings
import tools


# In[240]:
class Predictor(object):
    def __init__(self):
        temps, year, month, day, date = self.get_data()
        self.temp_calendar = self.build_temp_calendar(temps, year, month, day)
        deseasonalized_temps = np.zeros(temps.size)
        for i, temp in enumerate(temps):
            seasonal_temp = self.find_seasonal_temp(year[i], month[i], day[i])
            deseasonalized_temps[i] = temp - seasonal_temp
            
        self.slope, self.intercept = self.get_three_day_coefficients(deseasonalized_temps)







    def get_data(self):

        weather_filename = "180703_weather_data.csv"
        weather_file = open(weather_filename)
        weather_data = weather_file.read()
        weather_file.close()




        lines = weather_data.split('\n')
        labels = lines[0]
        values = lines[1:]
        n_values = len(values)
        
        #print lines[:100]





        year = []
        month = []
        day = []
        date = []
        max_temp = []
        j_date = 3
        j_max_temp = 17

        for i_row in range(n_values):
            split_values = values[i_row].split(',')
            split_values[:] = [i.replace('"', "") for i in split_values]
            
            if len(split_values) >= j_max_temp:
                date.append(split_values[j_date])
               
                max_temp.append(float(split_values[j_max_temp]))






        date[:] = [i.replace('"', "") for i in date]
        j_day = 2
        j_month = 1
        j_year = 0
        n_date = len(date)
        date_values = date[0:]
        for dates in range(n_date):
            split_date = date_values[dates].split('-')
            year.append(int(split_date[j_year]))
            month.append(int(split_date[j_month]))
            day.append(int(split_date[j_day]))
    



        warnings.simplefilter(action='ignore', category=FutureWarning)
        temps = np.array(max_temp)
        temps[np.where(temps == "")] = np.nan
        #print(split_values)
        #print(max_temp)
        #print(temps)
        i_start = np.where(np.logical_not(np.isnan(temps)))[0][0]
        temps = temps[i_start:]
        year = year[i_start:]
        month = month[i_start:]
        day = day[i_start:]
        i_nans = np.where(np.isnan(temps))[0]

        for i in range(temps.size):
            if np.isnan(temps[i]):
                temps[i] = temps[i - 1]
            
        return (temps, year, month, day, date)
        


# In[292]:




    def build_temp_calendar(self, temps, year, month, day):
        day_of_year = np.zeros(temps.size)
        for i_row in range(temps.size):
            day_of_year[i_row] = tools.find_day_of_year(year[i_row], month[i_row], day[i_row])


        median_temp_calendar = np.zeros(366)
        #ten_day_medians = np.zeros(temps.size)
        for i_day in range(0, 365):
            low_day = i_day - 5     
            high_day = i_day + 4
    
            if low_day < 0:
                low_day += 365
    
            if high_day > 365:
                high_day += -365
                
            if low_day < high_day:
                i_window_days = np.where(np.logical_and(day_of_year >= low_day, day_of_year <= high_day))
            
            else:
                i_window_days = np.where(np.logical_or(day_of_year >= low_day, day_of_year <= high_day))
    
            ten_day_median = np.median(temps[i_window_days])
            median_temp_calendar[i_day] = ten_day_median
            #ten_day_medians[np.where(day_of_year == i_day)] = ten_day_median
    
    
            if i_day == 364:
                #ten_day_medians[np.where(day_of_year == 365)] = ten_day_median
                median_temp_calendar[365] = ten_day_median
            
            return median_temp_calendar

    def get_three_day_coefficients(self, residuals):
        slope, intercept = np.polyfit(residuals[:-3], residuals[3:], 1)
        return (slope, intercept)

# In[313]:


    def find_seasonal_temp(self, year, month, day):   
        doy = tools.find_day_of_year(year, month, day)
        seasonal_temp = self.temp_calendar[doy]
        return seasonal_temp
    
    def predict_deseasonalized(self, three_day_temp):
        predicted_temp = self.intercept + self.slope * three_day_temp
        return predicted_temp
    
    def deseasonalize(self, temp, doy):
        deseasonalized_temp = temp - self.temp_calendar[doy]
        return deseasonalized_temp
    

    def reseasonalize(self, deseasonalized_temp, doy):
        reseasonalized_temp = deseasonalized_temp + self.temp_calendar[doy]
        return reseasonalized_temp
    
    def predict(self, year, month, day, past_temp):
        doy = tools.find_day_of_year(year, month, day)
        doy_past = doy - 3
        if doy_past < 0:
            doy_past += 365
        deseasonalized_temp = self.deseasonalize(past_temp, doy_past)
        deseasonalized_prediction = self.predict_deseasonalized(deseasonalized_temp)
        prediction = self.reseasonalize(deseasonalized_prediction, doy)
        return prediction


def test():
    predictor = Predictor()
    temps, year, month, day, date = predictor.get_data()
    deseasonalized_predictions = np.zeros(temps.size)
    deseasonalized_temps = np.zeros(temps.size)
    doy = np.zeros(temps.size, dtype=np.int)
    
    for i, temp in enumerate(temps):
        seasonal_temp = predictor.find_seasonal_temp(year[i], month[i], day[i])
        deseasonalized_temps[i] = temp - seasonal_temp
        doy[i] = tools.find_day_of_year(year[i], month[i], day[i])
                    
    for i, temp in enumerate(deseasonalized_temps):        
        deseasonalized_predictions[i] = predictor.predict_deseasonalized(temp)
    
    predictions = np.zeros(temps.size - 3)
    for i, temp in enumerate(deseasonalized_predictions[:-3]):
        predictions[i] = predictor.reseasonalize(temp, doy[i + 3])
    
    residuals = temps[3:] - predictions
    print('MAE:', np.mean(np.abs(residuals)))
    actuals = temps[3:]
    
    
    sensitivity = []
    targets = np.arange(84, 90)
    for target in targets:
        i_warm = np.where(actuals > 85)[0]
        i_warm_predictions = np.where(predictions > target)[0]
        n_true_positives = np.intersect1d(i_warm, i_warm_predictions).size
        n_false_negatives = np.setdiff1d(i_warm, i_warm_predictions).size
        n_false_positives = np.setdiff1d(i_warm_predictions, i_warm).size
        n_true_negatives = (actuals.size - n_true_positives - n_false_positives  - n_false_negatives)
        #print("Accurately predicted warm", n_true_positives, "times")
        #print("Predicted cold when it was warm", n_false_negatives, "times")
        #print("Predicted warm when it was cold", n_false_positives, "times")
        #print("Accurately predicted cold", n_true_positives, "times")
        sensitivity.append(float(n_true_positives) / float((n_true_positives + n_false_positives)))
        #print("Fraction of warm times", sensitivity[-1])    

def test_single():
    predictor = Predictor()
    test_year = 2018
    test_month = 6
    test_day = 16
    test_temp = 94
    prediction = predictor.predict(test_year, test_month, test_day, test_temp)
    print('For year=', test_year,
          ', month=', test_month,
          ', day=', test_day,
          ', temp=', test_temp)
    print('predicted temprature is', prediction)

if __name__ == '__main__':
    predictor = Predictor()
    test()
    test_single()
    









