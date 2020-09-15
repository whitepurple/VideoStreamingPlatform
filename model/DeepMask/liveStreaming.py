from __future__ import print_function
from facenet_pytorch import MTCNN
from ffpyplayer.player import MediaPlayer
from torch.nn import DataParallel
from src.backbone import resnet_face18
from src.config import Config

import warnings
warnings.filterwarnings('ignore')
import time
import torch
import cv2
import numpy as np
import random
import math
import subprocess


def setDevice():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))  
    return device

def setMTCNN():
    return MTCNN(margin=20,min_face_size=50,device=setDevice())

def setARCFACE():
    model = resnet_face18(False)
    model = DataParallel(model).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(torch.load('src/weights/resnet18_pretrain.pth'))
    model.eval()
    return model

def load_image(image): 
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def arcMargin(image, model):
    image = load_image(image)
    image = torch.from_numpy(image)
    image = image.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    feature = model(image)
    feature_1 = feature[::2]
    feature_2 = feature[1::2]
    feature = torch.cat((feature_1, feature_2),dim=1)  
    return feature

def cosine_similarity_all(x1, x2): # 여러개의 Input에 대한 Cosine Metric
    """
    ex) x1 size [128, 512], x2 size [1, 512]
        similarity size [128, 1]
    """
    #assert len(x1.size()) == len(x2.size()) == 2
    #assert x1.size(1) == x2.size(1)
    x2t = torch.transpose(x2, 0, 1)
    inner_product = torch.mm(x1, x2t)
    normx1 = torch.norm(x1,dim=1).unsqueeze(1)
    normx2 = torch.norm(x2,dim=1).unsqueeze(0)

    return inner_product / (normx1*normx2)

def getSimilarity(face_img,regist_img, model):
    t1 = time.time()
    face_embedding = arcMargin(face_img,model)   
    t2 = time.time()
    #regist_embedding = arcMargin(regist_img,model)
    #regist_embedding = torch.randn(200, 1024, dtype=torch.float).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))   
    regist_embedding = face_embedding
    t3 = time.time()
    similarity =cosine_similarity_all(face_embedding,regist_embedding)
    t4 = time.time()
    similarity = torch.max(similarity)
    t5 = time.time()
    #print(t5-t4, t4-t3, t3-t2, t2-t1)
    #print(similarity)
    return similarity

def makeEmbedding():
    return 0

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

def drawbox(frame, boxes,ori_width,ori_height,width, height,face_model):
    for box in boxes:
        if box[-1] > 0.9:
            box[0], box[1], box[2], box[3] = box[0] * ori_width / width, box[1] * ori_height / height, box[2] * ori_width / width, box[3] * ori_height / height
            box = box.astype(int)
            box = boxminmax(box,frame)
            face_img = frame[box[1]:box[3],box[0]:box[2]]
            face_img = cv2.resize(face_img,(128,128))
            face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
            similarity = getSimilarity(face_img,face_img,face_model)
            if similarity < 0.08:
                frame = faceblur(frame,box)  
            frame = cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)
    return frame

def show_detectedVideo(video):
    MTCNN = setMTCNN()
    print("Load MTCNN Complete!!")    
    ARCFACE = setARCFACE()
    print("Load ARCFACE Complete!!")    
    output_path = 'rtmp://218.150.183.59:1935/encode/testkey'
    input_path = 'rtmp://218.150.183.59:1935/key/testkey'
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command1 = ['ffmpeg',
            #'-thread_queue_size', '500',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', str(fps),
            '-i', '-',
                
            '-itsoffset','1500ms',
            # '-copyts','-start_at_zero',
            #'-thread_queue_size', '500',
            '-re',#str(fps),
            '-f', 'live_flv',
            '-i', input_path,            
            '-map', '0:v:0',
            '-map', '1:a:0,0:v:0',
            '-c:v', 'libx264',
            '-pix_fmt', 'rgb4',
            '-c:a', 'copy',
            '-b:a', '128k',
            '-b:v','2500k',            
            '-f', 'flv',
            '-r', str(fps),
            '-preset','fast',        
            '-tune','zerolatency', 
            output_path
            ]

        # using subprocess and pipe to fetch frame data
    p1 = subprocess.Popen(command1, stdin=subprocess.PIPE)
    if cap.isOpened():
        print("Width: {}, height {}".format(cap.get(3), cap.get(4)))
        prevTime = 0
    while True:   
        ret, frame = cap.read()
        ori_height = frame.shape[0]
        ori_width = frame.shape[1]
        #detect_frame = cv2.resize(frame,(360,360),cv2.INTER_AREA)
        detect_frame = frame
        if ret:
            bounding_boxes, probs = MTCNN.detect(cv2.cvtColor(detect_frame,cv2.COLOR_BGR2RGB),False)     
            if bounding_boxes is not None:
                probs = probs.reshape(probs.shape[0],1)           
                box_probs = np.column_stack([bounding_boxes,probs])
                frame = drawbox(frame,box_probs,ori_width,ori_height,ori_width,ori_height,ARCFACE)
 
        p1.stdin.write(frame.tobytes())  
    cap.release()
    cap.destroyAllWindows()


if __name__ == "__main__":


    
    show_detectedVideo("rtmp://218.150.183.59:1935/key/testkey")        