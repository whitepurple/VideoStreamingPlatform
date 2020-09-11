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
    users = DiceUser.objects\
                .filter(is_superuser=False)\
                .extra(select={'profile':"int('user_id') % 7"})\
                .values('id','username','last_login')
    for user in users:
        user['profile'] = 'img/v{}.png'.format(user['id']%7+1)
    print(users)
    return render(request, 'index.html', {'users':users})

@login_required(login_url='/login/')
def stream(request, user_pk):
    alpha = 'qwerasdfzxcvtyuighjkbnmopl         '
    user = DiceUser.objects.filter(id=user_pk).values('id','username','last_login')[0]
    user['profile'] = 'img/s{}.png'.format(user['id']%7+1)
    print(user)
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
        
        user_redirect = "http://218.150.183.58/live/{}/index.m3u8".format(user['username'])
        
    return render(request, 'video-page.html', { 'comments':comments, 
                                                'vuser': user, 
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

    # Redirect to the streamer's public username
    return redirect("/" +stream.user.username)


@require_POST
@csrf_exempt
def stop_stream(request):
    """ This view is called when a stream stops.
    """
    Stream.objects.filter(key=request.POST["name"]).update(started_at=None)
    return HttpResponse("OK")
