import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('data/train/nabil.csv')
df.sort_values(by=['Date'],ascending=[True],inplace=True)
df.head()
df1=df.reset_index()
df1.drop(['index'], axis=1, inplace=True)
df1
df1.tail(30)
testd=df1.tail(30)
testd
testd=testd.reset_index()
testd.drop(['index'], axis=1, inplace=True)
testd
testd.to_csv('data/test/nbbbbbbt.csv')
traind=df1.head(len(df1)-len(testd))
traind.to_csv('data/train/nbbbl.csv')
df1.shape
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
df1.shape
# spliting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
# converting array to matrix
import numpy
def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
         a=dataset[i:(i+time_step),0]
         dataX.append(a)
         dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX),numpy.array(dataY)

time_step=100
X_train,y_train=create_dataset(train_data,time_step)
X_test,ytest=create_dataset(train_data,time_step)
print(X_train.shape)
# reshape timro three dimensions 
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# model=Sequential()
# model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error',optimizer='adam')
# model.summary()
# model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
filename='nbbbl'
# model.save('models/'+ filename +'.h5')
from tensorflow.keras.models import load_model
model = load_model('models/'+filename+'.h5')
import tensorflow as tf
# lets do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
# transformback to original format
test_data=scaler.inverse_transform(test_data)
test_data
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
test_predict
predict = pd.DataFrame(test_predict, columns = ['prediction'])
actual = pd.DataFrame(test_predict, columns = ['prediction'])

import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))
math.sqrt(mean_squared_error(ytest,test_predict))
### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(test_data, color='blue', label='Actual Nabil Stock Price')
plt.plot(train_predict, color='red', label='Predicted Nabil Stock Price')
plt.title('Nabil Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Nabil Stock Price')
plt.legend()
plt.show()
 # plt.savefig("static/images/output.png")