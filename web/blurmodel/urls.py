from django.urls import path
from . import views

urlpatterns = [
    path('onpublish', views.runStreaming, name='rtmp_onpublish'),
    path('start', views.streaming, name='startstreaming'),
]