from email import message
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


def dashboard(request):
    return render(request, 'base/dashboard.html')


def product(request):
    return render(request, 'base/product.html')


def room(request):
    return render(request, 'room.html')


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
            user = user.objects.get(username=username)
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
    form = UserCreationForm()
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
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
        dfp = df.iloc[:, 2:3].values
        dfp
        scaler = MinMaxScaler(feature_range=(0, 1))
        dfs = scaler.fit_transform(dfp)

        features_set = []
        labels = []
        for i in range(60, 586):
            features_set.append(dfs[i-60:i, 0])
            labels.append(dfs[i, 0])

        features_set, labels = np.array(features_set), np.array(labels)

        features_set = np.reshape(
        features_set, (features_set.shape[0], features_set.shape[1], 1))
        features_set.shape
        filename = 'models/'+search_text.lower()+'.h5'
        loaded_model = load_model(filename)
        dfc = pd.read_csv('data/test/'+search_text.lower()+'.csv', encoding = 'unicode_escape')
        dtp = dfc.iloc[:, 2:3]
        dtp
        dt = pd.concat((df['Open'], dfc['Open']), axis=0)
        test_inputs = dt[len(dt) - len(dfc) - 60:].values
        test_inputs.shape

        test_inputs = test_inputs.reshape(-1, 1)
        test_inputs = scaler.transform(test_inputs)

        test_features = []
        for i in range(60, 90):
            test_features.append(test_inputs[i-60:i, 0])

        test_features = np.array(test_features)
        test_features = np.reshape(
        test_features, (test_features.shape[0], test_features.shape[1], 1))
        test_features.shape
        predictions = loaded_model.predict(test_features)
        predictions = scaler.inverse_transform(predictions)
        print(predictions)
        # chart=get_plot(dtp,predictions)
        # plt.figure(figsize=(10, 6))
        # plt.plot(dtp, color='blue', label='Actual Nabil Stock Price')
        # plt.plot(predictions, color='red', label='Predicted Nabil Stock Price')
        # plt.title('Nabil Stock Price Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Nabil Stock Price')
        # plt.legend()
        # plt.show()

        # 'chart':chart
        context={'search_text':search_text,'predictions':predictions,}
        return render(request, 'base/search.html',context)

def stock(request):
    return render(request,'base/stock.html')
# # #def get_graph():
# #     buffer= BytesIO()
# #     plt.savefig(buffer,format='png')
# #     buffer.seek(0)
# #     image_png=buffer.getvalue()
# #     graph=base64.b64decode(image_png)
# #     graph=graph.str.decode('utf-8')
# #     buffer.close()
# #     return graph
# # #def get_plot(dtp,predictions):
#     plt.switch_backend('AGG')
#     plt.title('Nabil Stock Price Prediction')
#     plt.xlabel('Date')
#     plt.ylabel('Nabil Stock Price')
#     plt.plot(dtp, color='blue', label='Actual Nabil Stock Price')
#     plt.plot(predictions, color='red', label='Predicted Nabil Stock Price')
#     plt.legend()
#     plt.figure(figsize=(10, 6))
#     graph=get_graph()
#     return graph