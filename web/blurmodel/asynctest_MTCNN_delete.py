import asyncio
from asyncio import Queue
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import subprocess

from .src2.backbone import resnet_face18
from torch.nn import DataParallel
import os
from concurrent.futures import ProcessPoolExecutor
from django.conf import settings

#from diceuser.models import DiceUser


executor = ProcessPoolExecutor(1)
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class StreamingRTMP:
    def __init__(self, registerdEmbedding):
        self.registerdEmbedding = torch.from_numpy(registerdEmbedding).to(self.setGPUDevice())
        self.straight = cv2.imread(os.path.join(settings.STATIC_ROOT, 'images/') +'straight.jpg')
        self.left = cv2.imread(os.path.join(settings.STATIC_ROOT, 'images/') +'left.jpg')
        self.right = cv2.imread(os.path.join(settings.STATIC_ROOT, 'images/') +'right.jpg')
        self.OUT = np.zeros(1000)
        self.m_MTCNN = self.setMTCNN()
        self.m_ARCFACE = self.setARCFACE()
        self.output_path = 'rtmp://218.150.183.59:1935/encode/testkey'
        self.input_path = 'rtmp://218.150.183.59:1935/key/testkey'
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
                         '-re', #'30',
                         '-f', 'live_flv',
                         '-i', self.input_path,        

                         '-map', '0:v:0',
                         '-map', '1:a:0,0:v:0',
                         '-c:v', 'libx264',
                         '-pix_fmt', 'yuv420p',
                         '-c:a', 'copy',
                         # '-b:a', '128k',
                         # '-b:v','2500',            
                         '-f', 'flv',
                         '-r', '30',
                         '-g', '60', 
                         '-tune','zerolatency', 
                         '-crf', '17',
                         '-x264opts', 'opencl',
                         self.output_path
                        ]
            # using subprocess and pipe to fetch frame data
        self.p1 = subprocess.Popen(self.command1, stdin=subprocess.PIPE)
        self.loop = asyncio.new_event_loop()

        self.rawvideo         = Queue(loop = self.loop)
        self.read_frame_queue = Queue(loop = self.loop)
        self.crop_frame_queue = Queue(loop = self.loop)
        self.crop_list_queue  = Queue(loop = self.loop)
        self.blur_frame_queue = Queue(loop = self.loop)
        self.end              = torch.ones(1)
        

    def run(self):
        self.loop.run_until_complete(
            asyncio.gather(
                self.read_frame(),
                self.detect_and_crop(),
                self.model_blur(),
                self.write_stdin(), loop = self.loop
            )
        )

        self.p1.kill()
        self.p1.wait(timeout=3)
        return True

    def setGPUDevice(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {device}')  
        return device

    def setMTCNN(self):
        return MTCNN(device=self.setGPUDevice())

    def setARCFACE(self):
        model = resnet_face18(True)
        model = DataParallel(model).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'static/') + 'src/weights/resnet18_pretrain.pth'))
        model.eval()
        return model

    def boxminmax(self, box,frame):
        # frame : (640,480,3)
        box[0] = np.clip(box[0],0,frame.shape[1]) 
        box[1] = np.clip(box[1],0,frame.shape[0])
        box[2] = np.clip(box[2],0,frame.shape[1])
        box[3] = np.clip(box[3],0,frame.shape[0])
        return box

    def cropface(self, frame, boxes):
        crop_face = cv2.cvtColor(cv2.resize(frame[boxes[1]:boxes[3],boxes[0]:boxes[2]],(128,128),cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        crop_face = np.dstack((crop_face, np.fliplr(crop_face)))
        crop_face = crop_face.transpose((2, 0, 1))            
        crop_face = crop_face[:, np.newaxis, :, :]   
        return crop_face

    def convertArcface(self, frame,boxes):
        crop_list=[]
        for box in boxes:
            # if box[-1] > 0.9:
            box = box.astype(int)
            box = self.boxminmax(box,frame)
            crop_list.append(self.cropface(frame,box))           
            #frame = cv2.rectangle(frame,(int(box[0]),int(box[1]),(int(box[2]),int(box[3])),(0,0,255),1))
        return crop_list

    def getArcface(self, crop_list,model):
        crop_list = np.array(crop_list)
        crop_list = crop_list.astype(np.float32, copy=False)
        crop_list -= 127.5
        crop_list /= 127.5            
        crop_list = torch.from_numpy(crop_list) 
        crop_list = crop_list.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))     
        #print(crop_list.shape)
        crop_list = torch.reshape(crop_list,(crop_list.shape[0]*crop_list.shape[1],crop_list.shape[2],crop_list.shape[3],crop_list.shape[4]))
        #print(crop_list.shape)
        out = model(crop_list)
        #print("Before : ",out.shape)
        out = torch.reshape(out,(out.shape[0]//2,out.shape[1]*2))            
        return out

    def cosine_similarity_all(self, x1, x2): # 여러개의 Input에 대한 Cosine Metric
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


    def faceblur(self, frame, b,landmark):
        nose = landmark[2][0]
        difference = nose-b[0]
        b = b.astype(int)
        b[0], b[1], b[2], b[3] = b[0] - 10, b[1] - 10, b[2] + 10, b[3] + 10 
        b = self.boxminmax(b,frame)    
        fwidth = b[2]-b[0]    

        if difference < fwidth/4:
            face_img = self.right
        elif difference > fwidth*3/4:    
            face_img = self.left
        else :
            face_img = self.straight
        face_img = cv2.resize(face_img,(b[2]-b[0],b[3]-b[1]),cv2.INTER_AREA)
        frame[b[1]:b[3],b[0]:b[2]] = face_img

        return frame

    async def read_frame(self):
        while True:
            ret, frame = await self.loop.run_in_executor(None, self.cap.read)
            # frame = await read_frame_queue.get()
            ## read_frame_queue.task_done()
            if not ret:
                frame = self.end
                self.read_frame_queue.put_nowait(frame)
                print('!- - End read Frame!')
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
            #read_frame_queue.task_done()


    async def detect_and_crop(self):
        while True:
            frame = await self.read_frame_queue.get()
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
                crop_list = self.convertArcface(frame,box_probs) 

            ############
            await self.crop_frame_queue.put((frame, crop_list, bounding_boxes, landmark))
            #crop_frame_queue.task_done()

    async def model_blur(self):
        while True:
            info = await self.crop_frame_queue.get()
            ## crop_frame_queue.task_done()
            if info == self.end:
                self.blur_frame_queue.put_nowait(info)
                print(f'!- - - End blured!')
                break
            frame, crop_list, bounding_boxes, landmark = info
            ############
            # await asyncio.sleep(0.013)
            # print(f'- - - frame {frame.shape} blured!')
            out = self.OUT[:len(crop_list)]
            if len(crop_list) != 0:
                crop_sim_matrix = self.getArcface(crop_list,self.m_ARCFACE)
                sim_matrix = self.cosine_similarity_all(crop_sim_matrix,self.registerdEmbedding) # crop_sim_matrix : 비교할 대상, . 뒤 : 등록된 임베딩        
                out, inds = torch.max(sim_matrix,dim=1)

            ############
            await self.blur_frame_queue.put((frame, bounding_boxes, landmark, out))
            #blur_frame_queue.task_done()



    async def write_stdin(self):
        while True:
            info = await self.blur_frame_queue.get()
            if info == self.end:
                print('write done!')
                print(f'!- - - -  End writed!')
                break
            frame, bounding_boxes, landmark, out = info
            ############
            if bounding_boxes is not None:
                for i, o in enumerate(out):
                    print(o)
                    if o < 0.9: 
                        b = bounding_boxes[i].astype(int)
                        b = self.boxminmax(b,frame)
                        frame = self.faceblur(frame,b,landmark[i]) 
                    #frame = faceblur2(frame,b)

            self.p1.stdin.write(frame.tobytes())  
            #print(f'- - - - frame {frame.shape} writed!')
            ############
