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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

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

# PLANT 2 STARTS HERE

# drop unnecessary columns and merge both dataframes along DATE_TIME
df_plant2 = pd.merge(df_gen2.drop(columns = ['PLANT_ID']), df_weather2.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

# add inverter number column to dataframe
sensorkeys = df_plant2.SOURCE_KEY.unique().tolist() # unique sensor keys
sensornumbers = list(range(1,len(sensorkeys)+1)) # sensor number
dict_sensor = dict(zip(sensorkeys, sensornumbers)) # dictionary of sensor numbers and corresponding keys

df_plant2 = df_plant2.fillna(0)

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

# cols_corr = ["DC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "WIND_SPEED", "HUMIDITY", "VISIBILITY"]
# cols_corr = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "HOURS", "MINUTES", "MINUTES_PASS", "WIND_SPEED", "HUMIDITY", "VISIBILITY"]
# corrMatrix = df_plant2[cols_corr].corr()
# plt.figure(figsize=(15,5))
# plt.show()

# ax = sns.heatmap(
#     corrMatrix, 
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     annot=True
# )
# ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# );
# plt.show()

# solar_dc = df_plant2.pivot_table(values='IRRADIATION', index='TIME', columns='DATE')

# def Daywise_plot(data= None, row = None, col = None, title=''):
#     cols = data.columns # take all column
#     gp = plt.figure(figsize=(20,40)) 
    
#     gp.subplots_adjust(wspace=0.2, hspace=0.5)
#     for i in range(1, len(cols)+1):
#         ax = gp.add_subplot(row,col, i)
#         data[cols[i-1]].plot(ax=ax, color='red')
#         ax.set_title('{} {}'.format(title, cols[i-1]),color='blue')
        
# Daywise_plot(data=solar_dc, row=12, col=3)
# plt.savefig("myplot2.png", dpi = 300)
# plt.show()

# Model
reg = LinearRegression()

# choose training data
# train_dates = ["2020-05-15", "2020-05-16", "2020-05-18", "2020-05-22", "2020-05-23", "2020-05-24", "2020-05-25", "2020-05-26", "2020-05-27"]
# df_train = df_plant2[df_plant2["DATE_STR"].isin(train_dates)]
# x = df_train.iloc[:, [3,4,5,6,7]]
# y = df_train["DC_POWER"]

# reg.fit(x, y)
# prediction = reg.predict(df_plant2.iloc[:, [3,4,5,6,7]])

# print(r2_score(df_plant2["DC_POWER"], prediction))

x = df_plant2.iloc[:, [6, 7, 8, 9, 11, 12]]
y = df_plant2["DC_POWER"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# #fit & predict
reg.fit(x_train, y_train)
prediction = reg.predict(x_test)

print("MLR r2 score:", r2_score(y_test, prediction))
print("MLR MSE score:", mean_squared_error(y_test, prediction))
print("MLR MAE score:", mean_absolute_error(y_test, prediction))

# plt.figure(figsize=(15,5))
# plt.scatter(y_test, prediction)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Actual vs Predicted")
# plt.show()

# save prediction, residual, and absolute residual
df_plant2["Prediction"] = reg.predict(x)
df_plant2["Residual"] = df_plant2["Prediction"] - df_plant2["DC_POWER"]
df_plant2["Residual_abs"] = df_plant2["Residual"].abs()

fig = px.scatter(df_plant2, x="DATE_TIME", y="DC_POWER", title="", color="Residual_abs", labels={"DC_POWER":"DC Power (kW)", "DATE_TIME":"Date & Time", "Residual_abs":"Residual"}, range_x=[datetime.date(2020, 5, 22), datetime.date(2020, 5, 27)])
fig.update_traces(marker=dict(size=3, opacity=0.7), selector=dict(mode='marker'))
fig.show()

# set confidence range for residual for fault
limit_fault=400

# Create new column to check proper operation
# Return "Normal" if operation is normal and "Fault" if operation is faulty
df_plant2["CLEAN"] = 0

for index in df_plant2.index:
    if  df_plant2["Residual_abs"][index] > limit_fault:
        df_plant2["CLEAN"][index] = 1
    else:
        df_plant2["CLEAN"][index] = 0

X = df_plant2.iloc[:, [6, 7, 8, 9, 11, 12]]
Y = df_plant2["CLEAN"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# over_sampler = SMOTE(sampling_strategy=0.05)
# X_res, y_res = over_sampler.fit_resample(X_train, Y_train)

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1_score = f1_score(Y_test, Y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)

# # Get and reshape confusion matrix data
# matrix = confusion_matrix(y_test, y_pred)
# matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# # Build the plot
# plt.figure(figsize=(16,7))
# sns.set(font_scale=1.4)
# sns.heatmap(matrix, annot=True, annot_kws={'size':10},
#             cmap=plt.cm.Greens, linewidths=0.2)

# # Add labels to the plot
# class_names = ["DC_POWER", "AMBIENT_TEMPERATURE", "IRRADIATION", "WIND_SPEED", "HUMIDITY", "VISIBILITY"]
# tick_marks = np.arange(len(class_names))
# tick_marks2 = tick_marks + 0.5
# plt.xticks(tick_marks, class_names, rotation=25)
# plt.yticks(tick_marks2, class_names, rotation=0)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix for Random Forest Model')
# plt.show()

# print(confusion_matrix(Y_test, Y_pred))

ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred)
plt.show()

# # Create a series containing feature importances from the model and feature names from the training data
# feature_importances = pd.Series(best_rf.feature_importances_, index=x_train.columns).sort_values(ascending=False)

# # Plot a simple bar chart
# feature_importances.plot.bar();
