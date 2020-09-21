import asyncio
from asyncio import Queue
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import subprocess
from .src2.backbone import resnet_face18
from torch.nn import DataParallel
import time
import os
from concurrent.futures import ProcessPoolExecutor
from django.conf import settings
from testapp.models import Stream
from django.shortcuts import get_object_or_404
#from diceuser.models import DiceUser

os.environ["CUDA_VISIBLE_DEVICES"]="1" #"0,1,2,3"
executor = ProcessPoolExecutor(1)

class StreamingRTMP:
    stream_num = 0
    def __init__(self, key):
        # self.gpunum = StreamingRTMP.stream_num%4
        # StreamingRTMP.stream_num += 1
        # print(self.gpunum)
        self.key = key
        self.device = self.setDevice()
        print(self.device)
        self.updateEmbedding()
        self.straight = cv2.imread(os.path.join(settings.STATIC_ROOT, 'images/') +'merong.bmp')
        self.left = cv2.imread(os.path.join(settings.STATIC_ROOT, 'images/') +'left.jpg')
        self.right = cv2.imread(os.path.join(settings.STATIC_ROOT, 'images/') +'right.jpg')
        self.OUT = np.zeros(1000)
        self.m_MTCNN = self.setMTCNN()
        self.m_ARCFACE = self.setARCFACE()
        self.output_path = f'rtmp://218.150.183.59:1935/encode/{key}'
        self.input_path = f'rtmp://218.150.183.59:1935/key/{key}'
        self.cap = cv2.VideoCapture(self.input_path)
        self.fps = 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.command1 = ['ffmpeg',
                         '-thread_queue_size', '5000',
                         '-f', 'rawvideo',
                         '-vcodec', 'rawvideo',
                         '-pix_fmt', 'bgr24',
                         '-s', "{}x{}".format(self.width, self.height),
                         '-r', '30',
                         '-i', '-',
                             
                         '-itsoffset','1500ms',
                         '-thread_queue_size', '5000',
                         '-re',
                         '-f', 'live_flv',
                         '-i', self.input_path,        
 
                         '-map', '0:v:0',
                         '-map', '1:a:0,0:v:0',
                         '-c:v', 'libx264',
                         '-pix_fmt', 'yuv420p',
                         '-c:a', 'copy',         
                         '-f', 'flv',
                         '-r', '30',
                         '-g', '60', 
                         '-tune','zerolatency', 
                         '-crf', '17',
                        #  '-x264opts', 'opencl',
                         self.output_path]
        # using subprocess and pipe to fetch frame data
        self.p1 = subprocess.Popen(self.command1, stdin=subprocess.PIPE)
        self.loop = asyncio.new_event_loop()

        self.rawvideo = Queue(loop = self.loop)
        self.read_frame_queue = Queue(loop = self.loop)
        self.crop_frame_queue = Queue(loop = self.loop)
        self.blur_frame_queue = Queue(loop = self.loop)
        self.end = torch.ones(1)
        self.taskdone = False
        self.readtime = 0
        self.croptime = 0
        self.blurtime = 0
        self.writetime = 0
    
    def __del__(self):
        self.p1.kill()
        self.p1.wait(timeout=3)

    def run(self):
        self.loop.run_until_complete(
            asyncio.gather(
                self.read_frame(),
                self.detect_and_crop(),
                self.model_blur(),
                self.write_stdin(),
                self.embeddingLoop(), loop = self.loop
            )
        )
        self.p1.kill()
        self.p1.wait(timeout=3)
        return True
    

    def updateEmbedding(self):
        stream = get_object_or_404(Stream, key=self.key)
        faces = stream.user.registerd_faces.filter(is_registerd=True).all()
        embedding = []
        registerdEmbedding = np.full((3,1024),100, dtype=np.float32)
        for f in faces:
            frame = (
                np
                .frombuffer(f.embedding, np.float32)
                .reshape([-1,1024])
            )
            embedding.append(frame)
        if len(embedding) != 0:
            # print(embedding)
            raw = np.concatenate(embedding)
            registerdEmbedding = raw.reshape(-1, 1024)

        self.registerdEmbedding = torch.from_numpy(registerdEmbedding).to(self.device)

    def setDevice(self):
        # device = torch.device(f'cuda:{self.gpunum}' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {device}')  
        return device

    def setMTCNN(self):
        return MTCNN(margin=20,min_face_size=50,device=self.device)

    def setARCFACE(self):
        model = resnet_face18(False)
        model = DataParallel(model).to(self.device)
        model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'static/') + 'src/weights/resnet18_pretrain.pth'))
        # model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'static/') + 'src/weights/resnet18_KFace.pth'))
        model.eval()
        return model

    def boxClip(self, box,frame):
        # frame : (640,480,3)
        box[0] = np.clip(box[0],0,frame.shape[1]) 
        box[1] = np.clip(box[1],0,frame.shape[0])
        box[2] = np.clip(box[2],0,frame.shape[1])
        box[3] = np.clip(box[3],0,frame.shape[0])
        return box

    def faceCrop(self, frame, box):
        # crop_face = cv2.cvtColor(cv2.resize(frame[boxes[1]:boxes[3],boxes[0]:boxes[2]],(128,128)),cv2.COLOR_BGR2GRAY)
        face_crop = cv2.cvtColor(cv2.resize(frame[box[1]:box[3],box[0]:box[2]],(128,128),cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        face_crop = np.dstack((face_crop, np.fliplr(face_crop)))
        face_crop = face_crop.transpose((2, 0, 1))            
        face_crop = face_crop[:, np.newaxis, :, :]   
        return face_crop

    def faceCropList(self, frame,boxes):
        face_crop_list = []
        for box in boxes:
            box = box.astype(int)
            box = self.boxClip(box,frame)
            face_crop_list.append(self.faceCrop(frame,box))           
            #frame = cv2.rectangle(frame,(int(box[0]),int(box[1]),(int(box[2]),int(box[3])),(0,0,255),1))
        return face_crop_list

    def getEmbedding(self, face_crop_list,model):
        n = len(face_crop_list)
        face_crop_list = np.array(face_crop_list)
        face_crop_list = face_crop_list.astype(np.float32, copy=False)
        face_crop_list -= 127.5
        face_crop_list /= 127.5            
        face_crop_list = torch.from_numpy(face_crop_list) 
        face_crop_list = face_crop_list.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))     
        face_crop_list = torch.reshape(face_crop_list,(face_crop_list.shape[0]*face_crop_list.shape[1],face_crop_list.shape[2],face_crop_list.shape[3],face_crop_list.shape[4]))
        out = model(face_crop_list)
        out = torch.reshape(out,(out.shape[0]//2,out.shape[1]*2))
        #fe_1 = out[::2]
        #fe_2 = out[1::2]
        
        #out = np.hstack((fe_1.reshape(n,-1).detach().cpu().numpy(), fe_2.reshape(n,-1).detach().cpu().numpy()))      
        return out

    def cosine_similarity_all(self, x1, x2):
        x2t = torch.transpose(x2, 0, 1)
        inner_product = torch.mm(x1, x2t)
        normx1 = torch.norm(x1,dim=1).unsqueeze(1)
        normx2 = torch.norm(x2,dim=1).unsqueeze(0)
        return inner_product / (normx1*normx2)

    def faceblur(self, frame, box,landmark):
        nose = landmark[2][0]
        difference = nose-box[0]
        box = box.astype(int)
        box[0], box[1], box[2], box[3] = box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10 
        box = self.boxClip(box,frame)    
        fwidth = box[2]-box[0]    

        if difference < fwidth/4:
            face_img = self.right
        elif difference > fwidth*3/4:    
            face_img = self.left
        else :
            face_img = self.straight

        face_img = cv2.resize(face_img,(box[2]-box[0],box[3]-box[1]),cv2.INTER_AREA)
        frame[box[1]:box[3],box[0]:box[2]] = face_img

        return frame

    def faceblur2(self,frame, box, landmark):
        box = box.astype(int)
        box[0], box[1], box[2], box[3] = box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10 
        box = self.boxClip(box,frame)  
        face = frame[box[1]:box[3],box[0]:box[2]]
        blur_img = cv2.resize(self.straight,(face.shape[1],face.shape[0]),cv2.INTER_AREA)    
        blurface = np.where(blur_img!=np.array([0, 0, 0]), blur_img, face)
        frame[box[1]:box[3],box[0]:box[2]] = blurface
        return frame

    async def embeddingLoop(self):
        while not self.taskdone:
            await self.loop.run_in_executor(None, self.updateEmbedding)
            await asyncio.sleep(2)

    async def read_frame(self):
        while True:
            ret, frame = await self.loop.run_in_executor(None, self.cap.read)
            #point = time.time()
            # frame = await read_frame_queue.get()
            ## read_frame_queue.task_done()
            if not ret:
                frame = self.end
                self.read_frame_queue.put_nowait(frame)
                self.taskdone = True
                print('!- End read Frame!')
                break
            ############
            # print(f'- frame {frame.shape} captured!')
            frame = (
                np
                .frombuffer(frame, np.uint8)
                .reshape([self.height, self.width, 3])
            )
            ############
            await self.read_frame_queue.put(frame)
            # self.readtime = time.time() - point

            # print(self.readtime, end='/')
            # print(self.croptime, end='/')
            # print(self.blurtime, end='/')
            # print(self.writetime)
            #read_frame_queue.task_done()


    async def detect_and_crop(self):
        while True:
            frame = await self.read_frame_queue.get()
            #point = time.time()
            if frame == self.end:
                self.crop_frame_queue.put_nowait(frame)
                print(f'!- - End to device!')
                break
            ############
            # print(f'- - frame {frame.shape} to device!')
            bounding_boxes, probs, landmark = self.m_MTCNN.detect(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),True) 

            crop_list = []
            if bounding_boxes is not None:
                probs = probs.reshape(probs.shape[0],1)         
                box_probs = np.column_stack([bounding_boxes,probs])
                box_probs = [bp for bp in box_probs if bp[-1] > 0.95]          
                crop_list = self.faceCropList(frame,box_probs)
            else:
                box_probs = []
            ############
            await self.crop_frame_queue.put((frame, crop_list, box_probs, landmark))
            # self.croptime = time.time() - point
            #crop_frame_queue.task_done()

    async def model_blur(self):
        while True:
            info = await self.crop_frame_queue.get()
            #point = time.time()
            ## crop_frame_queue.task_done()
            if info == self.end:
                self.blur_frame_queue.put_nowait(info)
                print(f'!- - - End blured!')
                break
            frame, crop_list, box_probs, landmark = info
            ############
            # await asyncio.sleep(0.013)
            # print(f'- - - frame {frame.shape} blured!')
            
            out = self.OUT[:len(box_probs)]

            if len(crop_list) != 0:
                crop_sim_matrix = self.getEmbedding(crop_list,self.m_ARCFACE)
                sim_matrix = self.cosine_similarity_all(crop_sim_matrix,self.registerdEmbedding) # crop_sim_matrix : 비교할 대상, . 뒤 : 등록된 임베딩
                score, inds = torch.topk(sim_matrix,3,dim=1)     
                out = torch.mean(score,1,True)
                # print(out.shape)

            ############
            await self.blur_frame_queue.put((frame, box_probs, landmark, out))
            # self.blurtime = time.time() - point
            #blur_frame_queue.task_done()

    async def write_stdin(self):
        while True:
            info = await self.blur_frame_queue.get()
            #point = time.time()
            if info == self.end:
                print('write done!')
                print(f'!- - - -  End writed!')
                break
            frame, box_probs, landmark, out = info
            ############
            # print(out.shape, bounding_boxes.shape)
            if box_probs is not None:
                for i, o in enumerate(out):
                    if o < 0.63: 
                        b = box_probs[i].astype(int)
                        b = self.boxClip(b,frame)
                        frame = self.faceblur2(frame,b,landmark[i]) 

            self.p1.stdin.write(frame.tobytes())
            # self.writetime = time.time() - point
            #print(f'- - - - frame {frame.shape} writed!')
            ############
