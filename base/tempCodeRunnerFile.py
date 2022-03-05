import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
df=pd.read_csv('data/train/nabil.csv', encoding = 'unicode_escape')
index_no = df.columns.get_loc('Close')
dfp = df.iloc[:,index_no:index_no+1].values
dfp
index_d = df.columns.get_loc('Date')
dtp=df.iloc[:,index_d:index_d+1]
column = df.iloc[:, index_no:index_no+1]
dtp['Close']=column
dtp['Date'] = pd.to_datetime(dtp['Date'], format='%Y/%m/%d')
dtp=dtp.set_index('Date')
dtp
dtp.sort_values(["Date"],axis=0,ascending=[True], inplace=True)
dtp.tail()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

df_scaled = scaler.fit_transform(dtp)
forecast_features_set = []
labels = []
for i in range(60, 615):
    forecast_features_set.append(df_scaled[i-60:i, 0])
    labels.append(df_scaled[i, 0])
forecast_features_set , labels = np.array(forecast_features_set ), np.array(labels)
forecast_features_set = np.reshape(forecast_features_set, (forecast_features_set.shape[0], forecast_features_set.shape[1], 1))
forecast_features_set.shape
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(forecast_features_set.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(forecast_features_set, labels, epochs = 50, batch_size = 20)
forecast_list=[]

batch=df_scaled[-forecast_features_set.shape[1]:].reshape((1,forecast_features_set.shape[1],1))

for i in range(forecast_features_set.shape[1]):
    forecast_list.append(model.predict(batch)[0])
    batch = np.append(batch[:,1:,:], [[forecast_list[i]]], axis=1)
df_predict=pd.DataFrame(scaler.inverse_transform(forecast_list),index=df[-forecast_features_set.shape[1]:].index, 
                        columns=["prediction"])
df_predict =pd.concat([dtp,df_predict],axis=1)
df_predict.tail()
from pandas.tseries.offsets import DateOffset
add_dates=[dtp.index[-1]+DateOffset(days=x) for x in range(0,61)]
future_dates=pd.DataFrame(index=add_dates[1:],columns=dtp.columns)
future_dates.tail(60)
df_forecast=pd.DataFrame(scaler.inverse_transform(forecast_list),index=future_dates[-forecast_features_set.shape[1]:].index, columns=["prediction"])
df_forecast =pd.concat([dtp,df_forecast],axis=1)
df_forecast.tail(60)