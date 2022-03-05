import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from matplotlib import image
from pandas.tseries.offsets import DateOffset

df=pd.read_csv('data/train/nabil.csv')
dfp = df.iloc[:, 2:3].values
dfp
scaler = MinMaxScaler(feature_range=(0, 1))
dfs = scaler.fit_transform(dfp)

features_set = []
labels = []
for i in range(60, len(dfp)):
    features_set.append(dfs[i-60:i, 0])
    labels.append(dfs[i, 0])
features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(
features_set, (features_set.shape[0], features_set.shape[1], 1))
features_set.shape
filename = 'models/nabil.h5'
loaded_model = load_model(filename)
dfc = pd.read_csv('data/test/nabil.csv')
dtp = dfc.iloc[:, 2:3]
dtp
print(type(dtp))
dt = pd.concat((df['Open'], dfc['Open']), axis=0)
test_inputs = dt[len(dt) - len(dfc) - 60:].values
test_inputs.shape
len(test_inputs)

test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)

test_features = []
for i in range(60, len(test_inputs)):
    test_features.append(test_inputs[i-60:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(
test_features, (test_features.shape[0], test_features.shape[1], 1))
test_features.shape
predictions = loaded_model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
predictions
pdf = pd.DataFrame(predictions, columns = ['prediction'])
print(type(predictions))
a_pdf=pdf.append(dtp)

        # chart=get_plot(dtp,predictions)
        plt.figure(figsize=(10, 6))
        plt.plot(dtp, color='blue', label='Actual Nabil Stock Price')
        plt.plot(predictions, color='red', label='Predicted Nabil Stock Price')
        plt.title('Nabil Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Nabil Stock Price')
        plt.legend()
        # plt.show()
        # plt.savefig("static/images/output.png")