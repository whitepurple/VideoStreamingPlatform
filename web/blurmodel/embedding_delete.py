from __future__ import print_function
from facenet_pytorch import MTCNN
from torch.nn import DataParallel
from .src.backbone import resnet_face18
import warnings
warnings.filterwarnings('ignore')
import os
import torch
import cv2
import numpy as np
from django.conf import settings
import datetime
import ffmpeg

# if __name__ == "__main__":

class Embedding:
    num_model = 0
    def __del__(self):
        Embedding.num_model = Embedding.num_model-1

    def __init__(self, filename):
        self.gpunum = (Embedding.num_model)%4
        Embedding.num_model = Embedding.num_model+1
        
        m_MTCNN = self.setMTCNN()
        ARCFACE = self.setARCFACE()    
        embedding = []
        print(filename)
        cap = cv2.VideoCapture(filename)
        cap.set(cv2.CAP_PROP_FRAME_COUNT, 50)
        cap.set(cv2.CAP_PROP_FPS, 10)
        # cap.set(cv2.CAP_PROP_AUTO_ROTATION, True)
        # print(cap.get(cv2.CAP_PROP_ROTATION))
        i=0
        ret = False
        
        rotateCode = self.check_rotation(filename)
        
        while not ret:
            ret, frame = cap.read()
            # frame = cv2.transpose(frame) # 행렬 변경 
            # frame = cv2.flip(frame, 0)   # 뒤집기
            if rotateCode is not None:
                frame = self.correct_rotation(frame, rotateCode)

            path = f'{settings.MEDIA_ROOT}/profile/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png'
            cv2.imwrite(path, frame)
            self.face = path
            # self.face = frame.tobytes()

        while ret:
            ret, frame = cap.read()
            # i+=1
            if ret:# and i%4==0:
                # frame = cv2.transpose(frame) # 행렬 변경 
                # frame = cv2.flip(frame, 0)   # 뒤집기
                if rotateCode is not None:
                    frame = self.correct_rotation(frame, rotateCode)

                bounding_boxes, probs, landmark = m_MTCNN.detect(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),True) 
                
                if bounding_boxes is not None:
                    probs = probs.reshape(probs.shape[0],1)  
                    box_probs = np.column_stack([bounding_boxes,probs])  
                    crop_list = self.convertArcface(frame,box_probs) 
                    if len(crop_list) != 0:
                        out = self.get_sim(crop_list[0],ARCFACE)
                        embedding.append(out)
            # if i >= 200:
            #     break
        print('video scan done!')
        # print(np.array(embedding,dtype=np.float32))
        self.embedding = np.array(embedding,dtype=np.float32).tobytes()
        # print(self.embedding)
        print(len(self.embedding))
        # print(embedding.tobytes())    

    def getEmbedding(self):
        return self.embedding
    
    def getFace(self):
        return self.face
    
    def check_rotation(self, path_video_file):
        # this returns meta-data of the video file in form of a dictionary
        rotateCode = None
        try:
            meta_dict = ffmpeg.probe(path_video_file)
        
        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
            
            if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
                rotateCode = cv2.ROTATE_90_CLOCKWISE
            elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
                rotateCode = cv2.ROTATE_180
            elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
                rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
        return rotateCode

    def correct_rotation(self, frame, rotateCode):  
        return cv2.rotate(frame, rotateCode) 

    def setDevice(self):
        device = torch.device(f'cuda:{self.gpunum}' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))  
        return device

    def setMTCNN(self):
        return MTCNN(margin=20,min_face_size=50,device=self.setDevice())

    def load_image(self, image): 
        if image is None:
            return None
        image = np.dstack((image, np.fliplr(image)))
        image = image.transpose((2, 0, 1))
        image = image[:, np.newaxis, :, :]
        image = image.astype(np.float32, copy=False)
        image -= 127.5
        image /= 127.5
        return image

    def boxminmax(self, box,frame):
        # frame : (640,480,3)
        box[0] = np.clip(box[0],0,frame.shape[1]) 
        box[1] = np.clip(box[1],0,frame.shape[0])
        box[2] = np.clip(box[2],0,frame.shape[1])
        box[3] = np.clip(box[3],0,frame.shape[0])
        return box

    def cropface(self, frame, boxes):
        # crop_face = cv2.cvtColor(cv2.resize(frame[boxes[1]:boxes[3],boxes[0]:boxes[2]],(128,128)),cv2.COLOR_BGR2GRAY)
        crop_face = cv2.resize(frame[boxes[1]:boxes[3],boxes[0]:boxes[2]],(128,128),cv2.INTER_AREA)
        crop_face = np.dstack((crop_face, np.fliplr(crop_face)))
        crop_face = crop_face.transpose((2, 0, 1))            
        crop_face = crop_face[:, np.newaxis, :, :]
        crop_face = crop_face.astype(np.float32, copy=False)  
        crop_face -= 127.5
        crop_face /= 127.5 
        return crop_face

    def convertArcface(self, frame, boxes):
        crop_list=[]
        for box in boxes:
            if box[-1] > 0.9:
                box = box.astype(int)
                box = self.boxminmax(box,frame)
                crop_list.append(self.cropface(frame,box))           
                #frame = cv2.rectangle(frame,(int(box[0]),int(box[1]),(int(box[2]),int(box[3])),(0,0,255),1))
        return crop_list

    def setARCFACE(self):
        model = resnet_face18(False)
        model = DataParallel(model).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'static/') + 'src/weights/resnet18_pretrain.pth'))
        # model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'static/') + 'src/weights/resnet18_KFace.pth'))
        model.eval()
        return model

    def get_sim(self, img, model): # Similarity 계산
        # image_data = self.load_image(img)
        data = torch.from_numpy(img)
        data = data.to(torch.device("cuda"))
        output = model(data)
        output = output.data.cpu().numpy()
        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1.reshape(1,-1), fe_2.reshape(1,-1)))
        return feature[0]