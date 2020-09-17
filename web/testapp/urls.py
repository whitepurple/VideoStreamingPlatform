from django.contrib import admin
from django.urls import path
from . import views


def fake_view(*args, **kwargs):
    """ This view should never be called because the URL paths
        that map here will be served by nginx directly.
    """
    raise Exception("This should never be called!")

urlpatterns = [
    path('home', views.home, name = "home"),
    path('streaming/<str:username>', views.stream, name = "stream"),
    path('mypage', views.mypage, name = "mypage"),
    path('editface', views.registerface, name = "editface"),

    path('src', views.index, name = "srctest"),
    path('src1', views.VideoView, name = "vtest"),

    path("start_stream", views.start_stream, name="start-stream"),
    path("stop_stream", views.stop_stream, name="stop-stream"),
    path("live/<username>/index.m3u8", fake_view, name="hls-url"),
    path("tt", views.doublepublishtest, name="teste"),
]