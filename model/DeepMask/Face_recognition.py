from __future__ import print_function
from .src.backbone import resnet_face18
from .src.config import Config
from torch.nn import DataParallel
import torch
import numpy as np
import time
import os
import cv2
import time

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

def set_arcFace():
    arcface = resnet_face18(False)
    arcface = DataParallel(model).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    arcface.load_state_dict(torch.load('src/weights/resnet18_pretrain.pth'))
    arcface.eval()
    return arcface


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

def cosin_metric(x1, x2): # Cosine Metric
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

