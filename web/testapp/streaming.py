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
import subprocess
import sys

def setDevice(device=0):
    device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    
    return device

def setMTCNN(device=0):
    return MTCNN(margin=20,min_face_size=50,device=setDevice(device))

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

def show_detectedVideo(input_path,output_path, device=0):
    MTCNN = setMTCNN(device)
    cap = cv2.VideoCapture(input_path)
    print('caprute done')
    if cap.isOpened():
        print("Width: {}, height {}".format(cap.get(3), cap.get(4)))
            # gather video info to ffmpeg
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # command and params for ffmpeg
        command1 = ['ffmpeg',
                '-thread_queue_size', '500',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                '-i', '-',
                    
                '-itsoffset','1500ms',
                '-copyts','-start_at_zero',
                '-thread_queue_size', '500',
                '-r',str(fps),
                '-f', 'live_flv',
                '-i', input_path,
              
                '-map', '0:v:0',
                '-map', '1:a:0,0:v:0',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'copy',
                '-f', 'flv',
                '-r', str(fps),
                output_path
                ]

        # using subprocess and pipe to fetch frame data
        p1 = subprocess.Popen(command1, stdin=subprocess.PIPE)
    print('while start')
    try:
        while cap.isOpened():
            t1 = time.time()
            ret, frame = cap.read() 
            t2 = time.time()
            if ret:
                ori_height = frame.shape[0]
                ori_width = frame.shape[1]
                #detect_frame = cv2.resize(frame,(480,360),cv2.INTER_AREA)
                detect_frame = frame
                bounding_boxes, probs = MTCNN.detect(cv2.cvtColor(detect_frame,cv2.COLOR_BGR2RGB),False)
                if bounding_boxes is not None:
                    probs = probs.reshape(probs.shape[0],1)           
                    box_probs = np.column_stack([bounding_boxes,probs])
                    frame = drawbox(frame,box_probs,ori_width,ori_height,ori_width,ori_height)
                
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                    # break
                # else :
                #     print("Error")
                end_fps = time.time()
                fps = 1/(end_fps-start_fps)
                #cv2.putText(frame,'FPS :'+str(int(fps)),(0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.FONT_HERSHEY_COMPLEX)     
                start = time.time()
                p1.stdin.write(frame.tobytes()) 
                print(time.time()-start)

            else:
                p1.kill()
                print('ffmpeg killed')
                break
        #cap.release()
        #cap.destroyAllWindows()
    except:
        p1.kill()
        print('ffmpeg killed cv')
        pass
    print('while done')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        output_path = 'rtmp://218.150.183.59:1935/encode/ssbkey'
        input_path = 'rtmp://218.150.183.59:1935/key/testkey'
    else:
        output_path = sys.argv[1]
        input_path = sys.argv[2]
    show_detectedVideo(input_path, output_path)                
    #show_detectedVideo("Test2.mp4")        