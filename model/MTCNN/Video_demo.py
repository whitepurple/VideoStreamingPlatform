import warnings
warnings.filterwarnings('ignore')
from facenet_pytorch import MTCNN
from ffpyplayer.player import MediaPlayer
import time
import torch
import cv2
import numpy as np
import random
import math

def setDevice():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    
    return device

def setMTCNN():
    return MTCNN(margin=20,min_face_size=50,device=setDevice())

def faceblur(frame, box):
    width_rad = int((box[2]-box[0])/2)
    height_rad = int((box[3]-box[1])/2)    
    c_x = int((box[0]+box[2])/2)
    c_y = int((box[1]+box[3])/2)

    rad = int(math.sqrt(width_rad*width_rad + height_rad*height_rad)*1.05)
    
    blurred_img = cv2.blur(frame,(100,100))
    mask = np.zeros(frame.shape,dtype=np.uint8)
    mask = cv2.circle(mask,(c_x,c_y),rad,(255,255,255),-1)
    blurface = np.where(mask!=np.array([255, 255, 255]), frame, blurred_img)

    return blurface

def boxminmax(box,frame):
    box[0] = np.clip(box[0],0,frame.shape[1])
    box[1] = np.clip(box[1],0,frame.shape[0])
    box[2] = np.clip(box[2],0,frame.shape[1])
    box[3] = np.clip(box[3],0,frame.shape[0])
    return box

def drawbox(frame, boxes,ori_width,ori_height,width, height):
    for box in boxes:
        if box[-1] > 0:
            box[0], box[1], box[2], box[3] = box[0] * ori_width / width, box[1] * ori_height / height, box[2] * ori_width / width, box[3] * ori_height / height
            box = box.astype(int)
            box = boxminmax(box,frame)
            #print(frame.shape)
            frame = faceblur(frame,box)  
            frame = cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)
    return frame

def show_detectedVideo(video):
    MTCNN = setMTCNN()
    cap = cv2.VideoCapture(video)
    audio = MediaPlayer(video)
    if cap.isOpened():
        print("Width: {}, height {}".format(cap.get(3), cap.get(4)))
        prevTime = 0
    index = 0
    while True:
        start_fps = time.time()
        if index % 100 == 0:        
            ret, frame = cap.read()
            audio_frame, val = audio.get_frame()        
            ori_height = frame.shape[0]
            ori_width = frame.shape[1]
            #detect_frame = cv2.resize(frame,(480,360),cv2.INTER_AREA)
            detect_frame = frame
            if ret:
                bounding_boxes, probs = MTCNN.detect(cv2.cvtColor(detect_frame,cv2.COLOR_BGR2RGB),False)     
                if bounding_boxes is not None:
                    probs = probs.reshape(probs.shape[0],1)           
                    box_probs = np.column_stack([bounding_boxes,probs])
                    frame = drawbox(frame,box_probs,ori_width,ori_height,ori_width,ori_height)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # else :
            #     print("Error")
            end_fps = time.time()
            fps = 1/(end_fps-start_fps)
            print(fps)
            #frame = cv2.resize(frame,(200,360),cv2.INTER_CUBIC)
            cv2.putText(frame,'FPS :'+str(int(fps)),(0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.FONT_HERSHEY_COMPLEX)
            cv2.imshow('Video Demo',frame)     
        index = index + 1
    cap.release()
    cap.destroyAllWindows()


if __name__ == "__main__":
    #show_detectedVideo("http://218.150.183.58/live/testuser/index.m3u8")
    show_detectedVideo("Test.mp4")        