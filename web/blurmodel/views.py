from django.http.response import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
import threading
# Create your views here.
# from .StreamingRTMP import StreamingRTMP
from .StreamingRTMP import StreamingRTMP

from asgiref.sync import sync_to_async
from testapp.models import Stream
import numpy as np
import asyncio
import subprocess

@csrf_exempt
def streaming(request):
    key = request.POST["name"]
    streaming = StreamingRTMP(key)
    streaming.run()
    return HttpResponse("streaming start")

@csrf_exempt
def runStreaming(request):
    stream = get_object_or_404(Stream, key=request.POST["name"])
    command = ['curl',
                '-F', f'name={request.POST["name"]}',
                '--max-time', '0',
                '-X', 'POST', 'http://218.150.183.59:8000/start',
                ]
    
    p = subprocess.Popen(command)
    return HttpResponse(f"{stream.user} is streaming...")

