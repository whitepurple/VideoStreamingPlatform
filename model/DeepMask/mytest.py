# from __future__ import print_function
# from src.backbone import resnet_face18
# from src.config import Config
# from torch.nn import DataParallel
# import torch
# import numpy as np
# import time
# import os
# import cv2
# import time

# def load_image(image):
#     if image is None:
#         return None
#     image = np.dstack((image, np.fliplr(image)))
#     image = image.transpose((2, 0, 1))
#     image = image[:, np.newaxis, :, :]
#     image = image.astype(np.float32, copy=False)
#     image -= 127.5
#     image /= 127.5
#     return image

# def set_model():
#     model = resnet_face18(False)
#     model = DataParallel(model).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
#     model.load_state_dict(torch.load('src/weights/resnet18_pretrain.pth'))
#     model.eval()
#     return model


# def arcMargin(image, model):
#     print(image.shape)
#     image = load_image(image)
#     print(image.shape)
#     image = torch.from_numpy(image)
#     print(image.shape)
#     image = image.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
#     feature = model(image)
#     #feature = feature.data.cpu().numpy()
#     feature_1 = feature[::2]
#     feature_2 = feature[1::2]
#     feature = torch.cat((feature_1, feature_2),dim=1)  
#     return feature

# def cosine_similarity_all(x1, x2): # 여러개의 Input에 대한 Cosine Metric
#     """
#     ex) x1 size [128, 512], x2 size [1, 512]
#         similarity size [128, 1]
#     """
#     #assert len(x1.size()) == len(x2.size()) == 2
#     #assert x1.size(1) == x2.size(1)
#     x2t = torch.transpose(x2, 0, 1)
#     inner_product = torch.mm(x1, x2t)
#     normx1 = torch.norm(x1,dim=1).unsqueeze(1)
#     normx2 = torch.norm(x2,dim=1).unsqueeze(0)

#     return inner_product / (normx1*normx2)

# def cosin_metric(x1, x2): # Cosine Metric
#     return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

# if __name__ == '__main__':
    
#     image_1 = cv2.imread("blurtest.jpg",0)
#     image_1 = cv2.resize(image_1,(128,128))

#     image_2 = cv2.imread("blurtest.jpg",0)
#     image_2 = cv2.resize(image_2,(128,128))
#     model = set_model()
#     start = time.time()
#     feature_1 = arcMargin(image_1,model)   
#     print(feature_1)
#     feature_2 = arcMargin(image_2,model)
#     x = feature_1.detach().cpu().numpy().tobytes()
#     #print(x)
#     x = np.frombuffer(x,dtype=np.float32)
#     print(x)
#     similarity = cosine_similarity_all(feature_1,feature_2)
#     print(time.time()-start) 
