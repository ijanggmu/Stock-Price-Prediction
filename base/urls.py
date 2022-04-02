from django.urls import path
from . import views
from .views import Home
urlpatterns = [
    path('search/', views.search, name="search"),
    path('login/', views.loginPage, name="login"),
    path('logout/', views.logoutUser, name="logout"),
    path('register/', views.registerPage, name="register"),
    path('profile/<str:pk/', views.userProfile, name="user-profile"),
    path('', Home.as_view(), name="home"),
    # path('product/', views.product, name="product"),
    # path('dashboard/', views.dashboard, name="dashboard"),
    # path('room/', views.room, name="room"),
    path('stock/', views.stock, name="stock"),
]
