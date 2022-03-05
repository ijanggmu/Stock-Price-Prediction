from http.client import PRECONDITION_FAILED
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r'C:\Users\GGMU\Downloads\bokl.csv')
df
df_p = df.iloc[:, 0:2]
df_p
df_p.Date = pd.to_datetime(df_p.Date, format="%d/%m/%Y")
df_p = df_p.set_index('Date')
df_p.head()
scaler = MinMaxScaler(feature_range=(0, 1))
df_s = scaler.fit_transform(df_p)
features_set = []
labels = []
for i in range(60, 586):
    features_set.append(df_s[i-60:i, 0])
    labels.append(df_s[i, 0])
features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(
    features_set, (features_set.shape[0], features_set.shape[1], 1))
features_set.shape
df_c = pd.read_csv(r'C:\Users\GGMU\Downloads\bokl_train.csv')
df_c
df_t_p = df_c.iloc[:, 0:2]
df_t_p
df_total = pd.concat((df['Open'], df_c['Open']), axis=0)
test_inputs = df_total[len(df_total) - len(df_c) - 60:].values
test_inputs.shape
test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)
test_features = []
for i in range(60, 89):
    test_features.append(test_inputs[i-60:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(
    test_features, (test_features.shape[0], test_features.shape[1], 1))
test_features.shape
filename = 'lstm/boklmodel.h5'
loaded_model = load_model(filename)
predictions = loaded_model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
predictions
pdf = pd.DataFrame(predictions, columns = ['prediction'])
pdf
df_t_p['prediction']=pdf
df_t_p
plt.figure(figsize=(15, 6))
plt.plot(df_t_p.loc[:,'Date'],df_t_p.loc[:,'Open'], color='blue', label='Actual BOKL Stock Price')
plt.plot(df_t_p.loc[:,'Date'],df_t_p.loc[:,'prediction'], color='red', label='Predicted BOKL Stock Price')
plt.plot(df_t_p, color='red', label='Predicted BOKL Stock Price')
# plt.plot(df_t_p, color='blue', label='Actual BOKL Stock Price')
# plt.plot(predictions, color='red', label='Predicted BOKL Stock Price')
plt.xticks(rotation=40)
plt.grid(linewidth=1)
plt.title('BOKL Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('BOKL')
plt.legend()
plt.show()
