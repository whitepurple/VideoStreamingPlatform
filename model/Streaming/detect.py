from __future__ import print_function
from src2.backbone import resnet_face18
from src2.config import Config
from torch.nn import DataParallel
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import subprocess

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

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
    face_embedding = arcMargin(face_img,model)   
    #regist_embedding = arcMargin(regist_img,model)
    regist_embedding = torch.randn(200, 1024, dtype=torch.float).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))   
    similarity =cosine_similarity_all(face_embedding,regist_embedding)
    similarity = torch.max(similarity)
    #print(similarity)
    return similarity

def boxminmax(box):
    box[0] = np.clip(box[0],0,240)
    box[1] = np.clip(box[1],0,320)
    box[2] = np.clip(box[2],0,240)
    box[3] = np.clip(box[3],0,320)
    return box

import ffmpeg

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    # cap = cv2.VideoCapture('Test.mp4')
    width = 240
    height = 320
    output_path = 'rtmp://218.150.183.59:1935/encode/testkey'
    input_path = 'rtmp://218.150.183.59:1935/key/testkey'

    in_stream = ffmpeg.input(input_path, r=30, itsoffset='1500ms')

    audio_stream = in_stream.audio

    video_stream = ffmpeg.input(input_path, r=30)
    video_stream = ffmpeg.output(video_stream, 'pipe:', format='rawvideo', pix_fmt='bgr24',r=15)
    video_stream = ffmpeg.run_async(video_stream, pipe_stdout=True)

    out_stream = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height),r=15)
    # out_stream = ffmpeg.map_audio(out_stream, audio_stream)
    out_stream = ffmpeg.output(out_stream, audio_stream, 
                    output_path,
                    vcodec='libx264', 
                    acodec='copy', 
                    pix_fmt='yuv420p', 
                    preset='ultrafast', 
                    # r='20', 
                    # g='50', 
                    r=15,
                    video_bitrate='2500k', 
                    format='flv')
    out_stream = ffmpeg.run_async(out_stream, pipe_stdin=True)

    model = setARCFACE()
    while True:
        t1 = time.time()
        in_bytes = video_stream.stdout.read(width * height * 3)
        t2 = time.time()
        if not in_bytes:
            break
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        # ret, frame = cap.read()
        #print(frame.shape)
        # frame = cv2.resize(frame,(640,480))
        img_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #print(img_raw.shape)
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print(loc[0],conf[0])
        exit()
        #print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)
        # show image
        #if args.save_image:
        crop = []
        if dets is not None :
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                b = boxminmax(b)
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                #print(b[1], b[3], b[0], b[2])
                crop_frame = cv2.cvtColor(cv2.resize(frame[b[1]:b[3],b[0]:b[2]],(128,128)),cv2.COLOR_BGR2GRAY)
                crop_frame = np.dstack((crop_frame, np.fliplr(crop_frame)))
                crop_frame = crop_frame.transpose((2, 0, 1))            
                crop_frame = crop_frame[:, np.newaxis, :, :]
                crop.append(crop_frame)
                # landms
                # cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
                # cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
                # cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
                # cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)
                # save image
        t3 = time.time()
            #name = "test2.jpg"
            #cv2.imwrite(name, img_raw)
        #print(len(crop))
        # start = time.time()
        if len(crop) != 0:
            crop = np.array(crop)
            crop = crop.astype(np.float32, copy=False)
            crop -= 127.5
            crop /= 127.5            
            crop = torch.from_numpy(crop) 
            crop = crop.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))            
            #print("Origin  : ",crop.shape)
            crop = torch.reshape(crop,(crop.shape[0]*crop.shape[1],crop.shape[2],crop.shape[3],crop.shape[4]))
            #print("Reshape : ",crop.shape)
            out = model(crop)
            #print("Before : ",out.shape)
            out = torch.reshape(out,(out.shape[0]//2,out.shape[1]*2))            
            #print("After  : ",out.shape)
        # print("Time : ",time.time()-start)
        #exit()
        #print("Time : ",time.time()-start)
        # fps = 1/(time.time()-start)
        #frame = cv2.resize(frame,(200,360),cv2.INTER_CUBIC)
        # cv2.putText(frame,'FPS :'+str(int(fps)),(0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.FONT_HERSHEY_COMPLEX)        
        # cv2.imshow('frame',frame)
        # print(frame.shape)
        # p1.stdin.write(frame.tobytes())  
        t4 = time.time()
        out_stream.stdin.write(
            frame
            .astype(np.uint8)
            .tobytes()
        )
        t5 = time.time()
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
        # print(t2-t1, t3-t2, t4-t3, t5-t4)
    cap.release()
    cv2.destroyAllWindows()

    # for i in range(100):
    #     image_path = "./curve/test.jpg"
    #     img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     #print(img_raw.shape)
    #     img = np.float32(img_raw)

    #     im_height, im_width, _ = img.shape
    #     scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    #     img -= (104, 117, 123)
    #     img = img.transpose(2, 0, 1)
    #     img = torch.from_numpy(img).unsqueeze(0)
    #     img = img.to(device)
    #     scale = scale.to(device)

    #     tic = time.time()
    #     loc, conf, landms = net(img)  # forward pass
    #     print('net forward time: {:.4f}'.format(time.time() - tic))

    #     priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    #     priors = priorbox.forward()
    #     priors = priors.to(device)
    #     prior_data = priors.data
    #     boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    #     boxes = boxes * scale / resize
    #     boxes = boxes.cpu().numpy()
    #     scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    #     landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    #     scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                            img.shape[3], img.shape[2]])
    #     scale1 = scale1.to(device)
    #     landms = landms * scale1 / resize
    #     landms = landms.cpu().numpy()

    #     # ignore low scores
    #     inds = np.where(scores > args.confidence_threshold)[0]
    #     boxes = boxes[inds]
    #     landms = landms[inds]
    #     scores = scores[inds]

    #     # keep top-K before NMS
    #     order = scores.argsort()[::-1][:args.top_k]
    #     boxes = boxes[order]
    #     landms = landms[order]
    #     scores = scores[order]

    #     # do NMS
    #     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #     keep = py_cpu_nms(dets, args.nms_threshold)
    #     # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    #     dets = dets[keep, :]
    #     landms = landms[keep]

    #     # keep top-K faster NMS
    #     dets = dets[:args.keep_top_k, :]
    #     landms = landms[:args.keep_top_k, :]

    #     dets = np.concatenate((dets, landms), axis=1)

    #     # show image
    #     if args.save_image:
    #         for b in dets:
    #             if b[4] < args.vis_thres:
    #                 continue
    #             text = "{:.4f}".format(b[4])
    #             b = list(map(int, b))
    #             cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #             cx = b[0]
    #             cy = b[1] + 12
    #             cv2.putText(img_raw, text, (cx, cy),
    #                         cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    #             # landms
    #             cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
    #             cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
    #             cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
    #             cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
    #             cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    #         # save image

    #         name = "test2.jpg"
    #         cv2.imwrite(name, img_raw)

