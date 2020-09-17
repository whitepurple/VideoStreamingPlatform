import time
import asyncio
from asyncio import Queue
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import math
import ffmpeg
import subprocess
from src2.backbone import resnet_face18
from src2.config import Config
from torch.nn import DataParallel
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

def cropface(frame, boxes):
    crop_face = cv2.cvtColor(cv2.resize(frame[boxes[1]:boxes[3],boxes[0]:boxes[2]],(128,128)),cv2.COLOR_BGR2GRAY)
    crop_face = np.dstack((crop_face, np.fliplr(crop_face)))
    crop_face = crop_face.transpose((2, 0, 1))            
    crop_face = crop_face[:, np.newaxis, :, :]   
    return crop_face

def convertArcface(frame,boxes):
    crop_list=[]
    for box in boxes:
        if box[-1] > 0.9:
            box = box.astype(int)
            box = boxminmax(box,frame)
            crop_list.append(cropface(frame,box))           
            frame = cv2.rectangle(frame,(int(box[0] * width / 480),int(box[1] * height / 640)),(int(box[2] * width / 480),int(box[3] * height / 640)),(0,0,255),1)
    return crop_list

def getArcface(crop_list,model):
    crop_list = np.array(crop_list)
    crop_list = crop_list.astype(np.float32, copy=False)
    crop_list -= 127.5
    crop_list /= 127.5            
    crop_list = torch.from_numpy(crop_list) 
    crop_list = crop_list.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))     
    #print(crop_list.shape)
    crop_list = torch.reshape(crop_list,(crop_list.shape[0]*crop_list.shape[1],crop_list.shape[2],crop_list.shape[3],crop_list.shape[4]))
    #print(crop_list.shape)
    out = model(crop_list)
    #print("Before : ",out.shape)
    out = torch.reshape(out,(out.shape[0]//2,out.shape[1]*2))            
    return out

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

def faceblur(frame, box):
    box[0] = box[0]
    box[1] = box[1]
    box[2] = box[2]
    box[3] = box[3]
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

# def faceblur(frame, box):
#     x1 = int(box[0])
#     y1 = int(box[1])
#     x2 = int(box[2])
#     y2 = int(box[3])
#     face_img = frame[y1:y2, x1:x2]
#     w = x2 - x1
#     h = y2 - y1
#     if w / 10 > 1 and h / 10 > 1:
#         face_img = cv2.resize(face_img, ((int)(w / 10), (int)(h / 10)))  # UnKnown의 Box에 있는 img를 10배 축소
#         face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)  # 10배 축소한 얼굴을 다시 원본 Size로 Resize
#         frame[y1:y2, x1:x2] = face_img

#     return frame

from concurrent.futures import ProcessPoolExecutor
executor = ProcessPoolExecutor(1)


async def read_frame():
    while True:
        ret, frame = await loop.run_in_executor(None, cap.read)
        
        # frame = await read_frame_queue.get()
        ## read_frame_queue.task_done()
        if not ret:
            frame = end
        # if frame == end:
            crop_frame_queue.put_nowait(frame)
            print(f'!- - {frame.shape} to device!')
            break
        ############
        # await asyncio.sleep(0.0666)
        # print(f'- frame {frame.shape} captured!')
        frame = (
            np
            .frombuffer(frame, np.uint8)
            .reshape([height, width, 3])
        )
        ############
        await read_frame_queue.put(frame)
        #read_frame_queue.task_done()


async def crop_and_to_device():
    while True:
        frame = await read_frame_queue.get()
        ## read_frame_queue.task_done()
        if frame == end:
            crop_frame_queue.put_nowait(frame)
            print(f'!- - {frame.shape} to device!')
            break
        ############
        # await asyncio.sleep(0.05)
        # print(f'- - frame {frame.shape} to device!')
        ori_height = frame.shape[0]
        ori_width = frame.shape[1]
        detect_frame = cv2.resize(frame,(480,640),cv2.INTER_AREA)
        bounding_boxes, probs = MTCNN.detect(cv2.cvtColor(detect_frame,cv2.COLOR_BGR2RGB),False) 
        box_and_crop_list = []
        crop_list = []
        if bounding_boxes is not None:
            probs = probs.reshape(probs.shape[0],1)           
            box_probs = np.column_stack([bounding_boxes,probs])  
            crop_list = convertArcface(frame,box_probs) 
            #print(out_matrix.shape) 
        box_and_crop_list.append(crop_list)
        box_and_crop_list.append(bounding_boxes)
        ############
        await crop_frame_queue.put(frame)
        await crop_list_queue.put(box_and_crop_list)
        #crop_frame_queue.task_done()

async def model_blur():
    while True:
        frame = await crop_frame_queue.get()
        box_and_crop_list = await crop_list_queue.get()
        ## crop_frame_queue.task_done()
        if frame == end:
            blur_frame_queue.put_nowait(frame)
            print(f'!- - - {frame.shape} blured!')
            break
        crop_list, bounding_boxes = box_and_crop_list
        ############
        # await asyncio.sleep(0.013)
        # print(f'- - - frame {frame.shape} blured!')
        if len(crop_list) != 0:
            out_matrix = getArcface(crop_list,ARCFACE)
            cv2.putText(frame,'OUT :'+str(out_matrix.shape),(0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.FONT_HERSHEY_COMPLEX)
            #print(out_matrix.shape, out_matrix[:2,:].shape)
            sim_matrix = cosine_similarity_all(out_matrix,out_matrix[0].view(1,-1))
            #print(sim_matrix)            
            out, inds = torch.max(sim_matrix,dim=1)
            #cv2.putText(frame,'Sim :'+str(sim_matrix),(0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.FONT_HERSHEY_COMPLEX)
            #print(sim_matrix)            
            #print(out)
            # for i in range(len(out)):
            #     if out[i] < 0.5:
    
        ############
        await blur_frame_queue.put((frame,bounding_boxes))
        #blur_frame_queue.task_done()



async def write_stdin():
    while True:
        frame, bounding_boxes = await blur_frame_queue.get()
        ## blur_frame_queue.task_done()
        if frame == end:
            print('write done!')
            print(f'!- - - -  {frame.shape} writed!')
            break
        ############
        # out_stream.stdin.write(
        #     frame
        #     .astype(np.uint8)
        #     .tobytes()
        # )
        if bounding_boxes is not None:
            for b in bounding_boxes:
                frame = faceblur(frame,b) 
        p1.stdin.write(frame.tobytes())  
        # await asyncio.sleep(0.001)
        #print(f'- - - - frame {frame.shape} writed!')
        ############

def put_frame(video_stream):
    while True:
        frame = video_stream.stdout.read(width * height * 3)
        # frame = await loop.run_in_executor(None, video_stream.stdout.read, width * height * 3)
        if not frame:
            break
        rawvideo.put_nowait(frame)
        # print('asd')

# async def process_async():
#     start = time.time()
#     await asyncio.wait([
#         find_users_async(3),
#         find_users_async(2),
#         find_users_async(1),
#     ])
#     end = time.time()
#     print(f'>>> 비동기 처리 총 소요 시간: {end - start}')

if __name__ == '__main__':

    MTCNN = setMTCNN()
    ARCFACE = setARCFACE()
    output_path = 'rtmp://218.150.183.59:1935/encode/testkey'
    input_path = 'rtmp://218.150.183.59:1935/key/testkey'
    cap = cv2.VideoCapture(input_path)
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command1 = ['ffmpeg',
            '-thread_queue_size', '5000',
            # '-framerate', '30',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', '30',
            '-i', '-',
                
            '-itsoffset','1500ms',
            # '-copyts','-start_at_zero',
            '-thread_queue_size', '5000',
            '-re', #'30',
            '-f', 'live_flv',
            '-i', input_path,        

            '-map', '0:v:0',
            '-map', '1:a:0,0:v:0',
            # '-map', '1:a:0',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'copy',
            '-b:a', '128k',
            '-b:v','2500',            
            '-f', 'flv',
            # '-filter:v', 'fps=fps=15',
            '-r', '30',
            '-g', '30',
            #'-preset','slow',        
            '-tune','zerolatency', 
            '-crf', '23',
            # '-maxrate', '2000k', 
            # '-bufsize', '1000k', 
            # '-pass', '2',
            # '-movflags', '+faststart',
            '-x264opts', 'opencl',
            #'-loglevel','quiet',
            output_path
            ]
        # using subprocess and pipe to fetch frame data
    p1 = subprocess.Popen(command1, stdin=subprocess.PIPE)

    rawvideo = Queue()
    read_frame_queue = Queue()
    crop_frame_queue = Queue()
    crop_list_queue = Queue()
    blur_frame_queue = Queue()
    end = torch.ones(1)

    # for a in range(150):
    #     rawvideo.put_nowait(torch.ones(1,a))
    # am = AsyncMasking()
    # asyncio.run(process_async())
    loop = asyncio.get_event_loop()
    
    loop.run_until_complete(
        asyncio.gather(
            read_frame(),
            crop_and_to_device(),
            model_blur(),
            write_stdin()
        )
    )
    # print(time.time())