from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', views.MyLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
]