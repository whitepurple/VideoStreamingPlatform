from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.response import Response
import random
import numpy as np
import urllib
import json
import cv2
from .cam import VideoCamera
from django.views.decorators import gzip
from django.http import HttpResponse,StreamingHttpResponse
from django.contrib.auth.decorators import login_required

from diceuser.models import DiceUser

from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import redirect, get_object_or_404
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from .models import Stream

from .streaming import show_detectedVideo
import subprocess
from django.conf import settings
 
# Create your views here.

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         return frame 

def VideoView(request):
    return render(request, 'video.html', {'img':gen(VideoCamera())})

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def index(request): 
    try:
        return StreamingHttpResponse(gen(VideoCamera()),content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("aborted")

########################################new template

@login_required(login_url='/login/')
def home(request):
    users = DiceUser.objects.filter(is_superuser=False).all()

    return render(request, 'index.html', {'users':users})

@login_required(login_url='/login/')
def stream(request, username):
    alpha = 'qwerasdfzxcvtyuighjkbnmopl         '
    users = DiceUser.objects.filter(is_superuser=False).exclude(username=username).all()
    vuser = DiceUser.objects.filter(username=username)[0]
    comments = []
    for k in range(100):
        comment = {}
        comment['nickname'] = 'nickname'
        comment['id'] = 'user_id'
        length = random.randrange(1, 100)
        text=''
        for i in range(length):
            text+=random.choice(alpha)
        comment['text'] = text
        comment['color'] = (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))
        comments.append(comment)
        
        user_redirect = "http://218.150.183.59/live/{}/index.m3u8".format(vuser.username)
        
    return render(request, 'video-page.html', { 'comments':comments, 
                                                'vuser': vuser, 
                                                'users': users,
                                                'src':user_redirect} )


########################################streaming

@require_POST
@csrf_exempt
def start_stream(request):
    """ This view is called when a stream starts.
    """
    stream = get_object_or_404(Stream, key=request.POST["name"])

    # Ban streamers by setting them inactive
    if not stream.user.is_active:
        return HttpResponseForbidden("Inactive user")

    # Don't allow the same stream to be published multiple times
    if stream.started_at:
        return HttpResponseForbidden("Already streaming")

    stream.started_at = timezone.now()
    stream.save()
    # try:
    #     import requests, json
    #     tmpData = {
    #         "name" : stream.key
    #     }
    #     r = requests.post("http://218.150.183.59:8000/tt", data=tmpData)
    # except:
    #     pass
    # Redirect to the streamer's public username    
    return redirect("/" +stream.user.username)


@require_POST
@csrf_exempt
def stop_stream(request):
    """ This view is called when a stream stops.
    """
    Stream.objects.filter(key=request.POST["name"]).update(started_at=None)
    return HttpResponse("OK")




@csrf_exempt
def doublepublishtest(request):
    print('hihi')
    stream = get_object_or_404(Stream, key=request.POST["name"])
    print(stream.key)
    output_path = 'rtmp://218.150.183.59:1935/encode/{}'.format(stream.key)
    input_path = 'rtmp://218.150.183.59:1935/key/{}'.format(stream.key)
    print(output_path)
    print(input_path)
    # a = subprocess.run(['python3',settings.BASE_DIR+'/testapp/streaming.py',input_path, output_path], capture_output=True)
    # print(a.stdout)
    show_detectedVideo(input_path, output_path)
    print('OK')
    return HttpResponse("OK")