from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.loginPage, name="login"),
    path('logout/', views.logoutUser, name="logout"),
    path('register/', views.registerPage, name="register"),
    path('profile/<str:pk/', views.userProfile, name="user-profile"),
    path('', views.home, name="home"),
    path('product/', views.product, name="product"),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('room/', views.room, name="room"),
]
