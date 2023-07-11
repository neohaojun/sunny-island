import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.dates as mdates 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing

xformatter = mdates.DateFormatter('%H:%M') # for time axis plots

# import plotly.offline as py
# py.init_notebook_mode(connected=True)

import sklearn
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings('ignore')

# Import all available data 
df_gen2 = pd.read_csv("/Users/neohaojun/Documents/PlatformIO/Projects/sunny-island/model/input/solar-power-generation-data/Plant_2_Generation_Data.csv")
df_weather2 = pd.read_csv("/Users/neohaojun/Documents/PlatformIO/Projects/sunny-island/model/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")

# drop unnecessary columns and merge both dataframes along DATE_TIME
df_plant2 = pd.merge(df_gen2.drop(columns = ['PLANT_ID']), df_weather2.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

# add inverter number column to dataframe
sensorkeys = df_plant2.SOURCE_KEY.unique().tolist() # unique sensor keys
sensornumbers = list(range(1,len(sensorkeys)+1)) # sensor number
dict_sensor = dict(zip(sensorkeys, sensornumbers)) # dictionary of sensor numbers and corresponding keys

# add column
df_plant2['SENSOR_NUM'] = 0
for i in range(df_gen2.shape[0]):
    df_plant2['SENSOR_NUM'][i] = dict_sensor[df_gen2["SOURCE_KEY"][i]]

# add Sensor Number as string
df_plant2["SENSOR_NAME"] = df_plant2["SENSOR_NUM"].apply(str) # add string column of sensor name

# adding separate time and date columns
df_plant2["DATE"] = pd.to_datetime(df_plant2["DATE_TIME"]).dt.date # add new column with date
df_plant2["TIME"] = pd.to_datetime(df_plant2["DATE_TIME"]).dt.time # add new column with time

# add hours and minutes for ml models
df_plant2['HOURS'] = pd.to_datetime(df_plant2['TIME'],format='%H:%M:%S').dt.hour
df_plant2['MINUTES'] = pd.to_datetime(df_plant2['TIME'],format='%H:%M:%S').dt.minute
df_plant2['MINUTES_PASS'] = df_plant2['MINUTES'] + df_plant2['HOURS']*60

# add date as string column
df_plant2["DATE_STR"] = df_plant2["DATE"].astype(str) # add column with date as string

df_plant2 = df_plant2.fillna(0)

cols_corr = ["DC_POWER", "AMBIENT_TEMPERATURE", "IRRADIATION", "WIND_SPEED", "VISIBILITY"]
corrMatrix = df_plant2[cols_corr].corr()
plt.figure(figsize=(15,5))
fig_corr = sns.heatmap(corrMatrix, annot=True)
plt.show()

# Model
reg = LinearRegression()

# choose training data
x = df_plant2.iloc[:, [3,4,5,6,7]]
y = df_plant2["DC_POWER"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.8, random_state = 0)

# #fit & predict
reg.fit(x_train, y_train)
prediction = reg.predict(x_test)

print(r2_score(y_test, prediction))

plt.figure(figsize=(15,5))
plt.scatter(y_test, prediction)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# save prediction, residual, and absolute residual
df_plant2["Prediction"] = reg.predict(df_plant2.iloc[:, [3,4,5,6,7]])
df_plant2["Residual"] = df_plant2["Prediction"] - df_plant2["DC_POWER"]
df_plant2["Residual_abs"] = df_plant2["Residual"].abs()

fig = px.scatter(df_plant2, x="DATE_TIME", y="DC_POWER", title="Fault Identification: Linear model (Zoomed in)", color="Residual_abs", labels={"DC_POWER":"DC Power (kW)", "DATE_TIME":"Date Time", "Residual_abs":"Residual"}, range_x=[datetime.date(2020, 5, 15), datetime.date(2020, 6, 17)])
fig.update_traces(marker=dict(size=3, opacity=0.7), selector=dict(mode='marker'))
fig.show()

# set confidence range for residual for fault
limit_fault=400

# Create new column to check proper operation
# Return "Normal" if operation is normal and "Fault" if operation is faulty
df_plant2["STATUS"] = 0
for index in df_plant2.index:
    if  df_plant2["Residual_abs"][index] > limit_fault:
        df_plant2["STATUS"][index] = "Fault"  
    else:
        df_plant2["STATUS"][index] = "Normal"
