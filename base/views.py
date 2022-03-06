from email import message
from .models import CreateUserForm
from django.core.files.storage import FileSystemStorage
from django.http import request
from django.shortcuts import redirect, render, HttpResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from pandas.tseries.offsets import DateOffset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
from matplotlib import image
import base64
from io import BytesIO
#Create your views here.
def home(request):
    return render(request, 'base/home.html')


# def dashboard(request):
#     return render(request, 'base/dashboard.html')


# def product(request):
#     return render(request, 'base/product.html')


# def room(request):
#     return render(request, 'room.html')


def userProfile(request, pk):
    user = User.objects.get(id=pk)
    context = {'user': user}
    return render(request, 'base/profile.html', context)
def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    page = 'login'
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            user = User.objects.get(username=username)
        except:
            messages.error(request, 'User doesnot exist')
            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, 'Username Or Password doesnot exist')

    context = {'page': page}
    return render(request, 'base/login_register.html', context)


def logoutUser(request):
    logout(request)
    return redirect('home')


def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    form = CreateUserForm()
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
    if form.is_valid():
        user = form.save(commit=False)
        user.username = user.username.lower()
        user.save()
        login(request, user)
        return redirect('home')
    else:
        messages.error(request, 'An error occured during registration')

    return render(request, 'base/login_register.html', {'form': form})
def search(request):
    if request.method=='POST':
        search_text=request.POST['search_text']
        df=pd.read_csv('data/train/'+search_text.lower()+'.csv', encoding = 'unicode_escape')
        index_no = df.columns.get_loc('Close')
        index_no
        dfp = df.iloc[:,index_no:index_no+1].values
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
        filename = 'models/'+search_text.lower()+'.h5'
        loaded_model = load_model(filename)
        dfc = pd.read_csv('data/test/'+search_text.lower()+'.csv', encoding = 'unicode_escape')
        index_d = dfc.columns.get_loc('Date')
        dtp=dfc.iloc[:,index_d:index_d+1]
        column = dfc.iloc[:, index_no:index_no+1]
        # print(dtp)
        dtp['Close']=column
        # print(dtp)
        dtp['Date'] = pd.to_datetime(dtp['Date'], format='%Y-%m-%d')
        # for i in date:
        #     print(type(i))
        # print(type(dfp))
        dt = pd.concat((df['Open'], dfc['Open']), axis=0)
        test_inputs = dt[len(dt) - len(dfc) - 60:].values
        test_inputs.shape

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
        # print(type(pdf))
        # print(pdf)
        dtp['prediction']=pdf.round()
        # print(dtp)
        
        # z=dtp.iloc[:,2:3]
        # for i in z:
        #     print(type(i))

        final_data=[]
        for i in range(dtp.shape[0]):
            temp=dtp.iloc[i]
            final_data.append(dict(temp))
        # print(type(final_data))
        
         

        # chart=get_plot(dtp,predictions)
        # plt.figure(figsize=(15, 8))
        # plt.plot(dtp.loc[:,'Date'],dtp.loc[:,'Open'], color='blue', label='Actual Nabil Stock Price')
        # plt.plot(dtp.loc[:,'Date'],dtp.loc[:,'prediction'], color='red', label='Predicted Nabil Stock Price')
        # plt.title('Nabil Stock Price Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Nabil Stock Price')
        # plt.legend()
        # plt.xticks(rotation=40)
        # plt.grid(linewidth=1)
        # plt.show()
        # plt.savefig("static/images/output.png")

        # 'chart':chart
        #forecast part
        df=pd.read_csv('data/'+search_text.lower()+'.csv', encoding = 'unicode_escape')
        index_no = df.columns.get_loc('Close')
        dfp = df.iloc[:,index_no:index_no+1].values
        dfp
        index_d = df.columns.get_loc('Date')
        dtp=df.iloc[:,index_d:index_d+1]
        column = df.iloc[:, index_no:index_no+1]
        dtp['Close']=column
        dtp['Date'] = pd.to_datetime(dtp['Date'], format='%Y/%m/%d')
        dtp=dtp.set_index('Date')
        print(dtp)
        dtp.sort_values(["Date"],axis=0,ascending=[True], inplace=True)
        print(dtp.tail())
        forecast_list=[]

        batch=dfs[-features_set.shape[1]:].reshape((1,features_set.shape[1],1))

        for i in range(features_set.shape[1]):
            forecast_list.append(loaded_model.predict(batch)[0])
            batch = np.append(batch[:,1:,:], [[forecast_list[i]]], axis=1)
        df_predict=pd.DataFrame(scaler.inverse_transform(forecast_list),index=dtp[-features_set.shape[1]:].index, 
                        columns=["prediction"])
        df_predict =pd.concat([dtp,df_predict],axis=1)
        from pandas.tseries.offsets import DateOffset
        add_dates=[dtp.index[-1]+DateOffset(days=x) for x in range(0,61)]
        future_dates=pd.DataFrame(index=add_dates[1:],columns=dtp.columns)
        future_dates.tail(60)
        df_forecast=pd.DataFrame(scaler.inverse_transform(forecast_list),index=future_dates[-features_set.shape[1]:].index, columns=["prediction"])
        df_forecast =pd.concat([dtp,df_forecast],axis=1)
        forecast_final=df_forecast.round().tail(60)
        print(type(forecast_final))
        print(forecast_final)
        context={'search_text':search_text,'predictions':predictions,'dtp':dtp,'final':final_data,'dat':index_d}
        return render(request, 'base/search.html',context)

def stock(request):
    if request.method =='POST':
        uploaded_file=request.FILES['document']
        title=request.POST['text']
        savefile=FileSystemStorage()
        name=savefile.save('data/'+ title +'.csv',uploaded_file)
        df=pd.read_csv('data/'+title+'.csv')
        df.sort_values(by=['Date'],ascending=[True],inplace=True)
        df1=df.reset_index()
        df1.drop(['index'], axis=1, inplace=True)
        df1
        df1.tail(30)
        testd=df1.tail(30)
        testd
        testd=testd.reset_index()
        testd.drop(['index'], axis=1, inplace=True)
        testd
        testd.to_csv('data/test/'+title.lower()+'.csv')
        traind=df1.head(len(df1)-len(testd))
        traind.to_csv('data/train/'+title.lower()+'.csv')
        dft=pd.read_csv('data/train/'+title.lower()+'.csv', encoding = 'unicode_escape')
        dft.shape
        dft
        index_no = dft.columns.get_loc('Close')
        index_no
        dfp = dft.iloc[:,index_no:index_no+1].values
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
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True,
                input_shape=(features_set.shape[1], 1)))

        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(features_set, labels, epochs=50, batch_size=20)
        filename=title.lower()
        model.save('models/'+ filename +'.h5')
    return render(request,'base/stock.html')