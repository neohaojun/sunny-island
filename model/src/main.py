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
df_gen1 = pd.read_csv("/Users/neohaojun/Documents/sunny island/input/solar-power-generation-data/Plant_1_Generation_Data.csv")
df_gen2 = pd.read_csv("/Users/neohaojun/Documents/sunny island//input/solar-power-generation-data/Plant_2_Generation_Data.csv")

df_weather1 = pd.read_csv("/Users/neohaojun/Documents/sunny island//input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
df_weather2 = pd.read_csv("/Users/neohaojun/Documents/sunny island//input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")

df_raj_weather = pd.read_csv("/Users/neohaojun/Documents/sunny island//input/solar-power-generation-data/rajasthan_weather.csv")

# PLANT 1 STARTS HERE

# # adjust datetime format
# df_gen1['DATE_TIME'] = pd.to_datetime(df_gen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
# df_weather1['DATE_TIME'] = pd.to_datetime(df_weather1['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

# # drop unnecessary columns and merge both dataframes along DATE_TIME
# df_plant1 = pd.merge(df_gen1.drop(columns = ['PLANT_ID']), df_weather1.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

# # add inverter number column to dataframe
# sensorkeys = df_plant1.SOURCE_KEY.unique().tolist() # unique sensor keys
# sensornumbers = list(range(1,len(sensorkeys)+1)) # sensor number
# dict_sensor = dict(zip(sensorkeys, sensornumbers)) # dictionary of sensor numbers and corresponding keys

# # add column
# df_plant1['SENSOR_NUM'] = 0
# for i in range(df_gen1.shape[0]):
#     df_plant1['SENSOR_NUM'][i] = dict_sensor[df_gen1["SOURCE_KEY"][i]]

# # add Sensor Number as string
# df_plant1["SENSOR_NAME"] = df_plant1["SENSOR_NUM"].apply(str) # add string column of sensor name

# # adding separate time and date columns
# df_plant1["DATE"] = pd.to_datetime(df_plant1["DATE_TIME"]).dt.date # add new column with date
# df_plant1["TIME"] = pd.to_datetime(df_plant1["DATE_TIME"]).dt.time # add new column with time

# # add hours and minutes for ml models
# df_plant1['HOURS'] = pd.to_datetime(df_plant1['TIME'],format='%H:%M:%S').dt.hour
# df_plant1['MINUTES'] = pd.to_datetime(df_plant1['TIME'],format='%H:%M:%S').dt.minute
# df_plant1['MINUTES_PASS'] = df_plant1['MINUTES'] + df_plant1['HOURS']*60

# # add date as string column
# df_plant1["DATE_STR"] = df_plant1["DATE"].astype(str) # add column with date as string

# cols_corr = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "SENSOR_NUM", "HOURS", "MINUTES", "MINUTES_PASS"]
# corrMatrix = df_plant1[cols_corr].corr()
# plt.figure(figsize=(15,5))
# fig_corr = sns.heatmap(corrMatrix,cmap="YlGnBu", annot=True)
# plt.show()

# PLANT 2 STARTS HERE

# adjust datetime format
# df_gen2['DATE_TIME'] = pd.to_datetime(df_gen2['DATE_TIME'],format = '%d-%m-%Y %H:%M')
# df_weather2['DATE_TIME'] = pd.to_datetime(df_weather2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

# drop unnecessary columns and merge both dataframes along DATE_TIME
df_plant2 = pd.merge(df_gen2.drop(columns = ['PLANT_ID']), df_weather2.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

# df_plant2 - df_plant2[df_plant2["SOURCE_KEY"].str.contains("81aHJ1q11NBPMrL") == False]
# df_plant2 - df_plant2[df_plant2["SOURCE_KEY"].str.contains("Et9kgGMDl729KT4") == False]
# df_plant2 - df_plant2[df_plant2["SOURCE_KEY"].str.contains("Quc1TzYxW2pYoWX") == False]
# df_plant2 - df_plant2[df_plant2["SOURCE_KEY"].str.contains("xoJJ8DcxJEcupym") == False]
# df_plant2 - df_plant2[df_plant2["SOURCE_KEY"].str.contains("Et9kgGMDl729KT4") == False]
# df_plant2 - df_plant2[df_plant2["SOURCE_KEY"].str.contains("Qf4GUc1pJu5T6c6") == False]
# df_plant2 - df_plant2[df_plant2["SOURCE_KEY"].str.contains("rrq4fwE8jgrTyWY") == False]
# df_plant2 - df_plant2[df_plant2["SOURCE_KEY"].str.contains("oZZkBaNadn6DNKz") == False]


# add inverter number column to dataframe
sensorkeys = df_plant2.SOURCE_KEY.unique().tolist() # unique sensor keys
sensornumbers = list(range(1,len(sensorkeys)+1)) # sensor number
dict_sensor = dict(zip(sensorkeys, sensornumbers)) # dictionary of sensor numbers and corresponding keys

# # add column
# df_plant2['SENSOR_NUM'] = 0
# for i in range(df_gen2.shape[0]):
#     df_plant2['SENSOR_NUM'][i] = dict_sensor[df_gen2["SOURCE_KEY"][i]]

# # add Sensor Number as string
# df_plant2["SENSOR_NAME"] = df_plant2["SENSOR_NUM"].apply(str) # add string column of sensor name

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

# cols_corr = ["DC_POWER", "AMBIENT_TEMPERATURE", "IRRADIATION", "WIND_SPEED", "VISIBILITY"]
# corrMatrix = df_plant2[cols_corr].corr()
# plt.figure(figsize=(15,5))
# fig_corr = sns.heatmap(corrMatrix,cmap="YlGnBu", annot=True)
# plt.show()

# solar_dc = df_plant2.pivot_table(values='DC_POWER', index='TIME', columns='DATE')

# def Daywise_plot(data= None, row = None, col = None, title=''):
#     cols = data.columns # take all column
#     gp = plt.figure(figsize=(20,40)) 
    
#     gp.subplots_adjust(wspace=0.2, hspace=0.5)
#     for i in range(1, len(cols)+1):
#         ax = gp.add_subplot(row,col, i)
#         data[cols[i-1]].plot(ax=ax, color='red')
#         ax.set_title('{} {}'.format(title, cols[i-1]),color='blue')
        
# Daywise_plot(data=solar_dc, row=12, col=3)
# plt.savefig("myplot.png", dpi = 300)
# plt.show()

# Model
reg = LinearRegression()

# df_plant2 = df_plant2.drop(df_plant2[df_plant2.DC_POWER == 0 & df_plant2 == "2020-05-15"].index)

# choose training data
# train_dates = ["2020-05-15", "2020-05-16", "2020-05-18", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27"]
# df_train = df_plant2[df_plant2["DATE_STR"].isin(train_dates)]

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
