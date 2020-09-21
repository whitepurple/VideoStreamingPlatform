from django.http import response
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
import random
import numpy as np
import urllib
import json
import cv2
from django.views.decorators import gzip
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import redirect, get_object_or_404
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from .models import Stream, Face
from .utils import *

import subprocess
from django.conf import settings
from diceuser.models import DiceUser
from .forms import VideoForm

def page_not_found(request,exception):
    return render(request, '404.html',status=404)

########################################new template

@login_required(login_url='/login/')
def home(request):
    users = get_users_orderby_streaming()

    return render(request, 'index.html', {'users':users})

@login_required(login_url='/login/')
def mypage(request):
    users = get_users_orderby_streaming()
    user = get_object_or_404(DiceUser, username=request.POST["name"])
    faces = user.registerd_faces.all()
    form= VideoForm()
    rtmp_path = f'rtmp://218.150.183.59:1935/key/{user.stream.key}'
    return render(request, 'subscriptions.html', {'users':users,
                                                    'faces':faces,
                                                    'videoform':form,
                                                    'key':rtmp_path})
@require_POST
def editregister(request):
    user = get_object_or_404(DiceUser, username=request.POST["name"])
    print(user)
    faces = user.registerd_faces.all()
    faceregister = request.POST.getlist('faces',[])
    for i, f in enumerate(faces):
        f.is_registerd = True if faceregister[i] == 'T' else False
        f.save()

    return redirect('home')

# from blurmodel.embedding import Embedding
from blurmodel.embedding import Embedding
from django.core.files import File

@require_POST
def registerface(request):
    form= VideoForm(request.POST, request.FILES)
    user = get_object_or_404(DiceUser, username=request.POST["username"])
    if form.is_valid():
        face = form.save(commit=False)
        face.streamer = user
        face.name = request.POST["facename"]
        if len(face.name) == 0:
            face.name = 'someone'
        face.save()
        embedding = Embedding(settings.MEDIA_ROOT+ '/'+str(face.videofile))
        face.embedding = embedding.getEmbedding()
        path = embedding.getFace()
        f = open(path, 'rb')
        face.profile =File(f, name=str.join('/',path.split('/')[-1:]))
        face.save()
        print('savetest')
        f.close
    return redirect('home')

    

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
    return redirect("/" +stream.user.username)


@require_POST
@csrf_exempt
def stop_stream(request):
    """ This view is called when a stream stops.
    """
    Stream.objects.filter(key=request.POST["name"]).update(started_at=None)
    return HttpResponse("OK")
