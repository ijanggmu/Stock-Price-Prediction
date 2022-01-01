from django.http import request
from django.shortcuts import redirect, render
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import time
# Create your views here.


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
