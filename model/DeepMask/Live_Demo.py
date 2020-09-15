# import os
# from facenet_pytorch import MTCNN
# from arcFace import *

# prevTime = 0
# start = False
# register = []
# person = []
# number = 0
# regist = 0
# numofregist = 0

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # GPU가 있다면 GPU Mode 없으면 CPU Mode
# print('Running on device: {}'.format(device))
# mtcnn = MTCNN(keep_all=True, device=device)  # MTCNN 선언
# model = get_arcface()  # ArcFace 선언

# print('Delete Image')
# path = "./data/RealTime/"  # Register Face가 저장될 Path
# filelist = [f for f in os.listdir(path) if f.endswith(".jpg")]  # jpg 확장자를 가지고 있는 모든 File을 저장
# for f in filelist:  # Path에 있는 모든 File을 Delete함
#     img = path + '/' + f
#     os.remove(img)
# print('Delete Image Finish')

# print('Register Face Start')
# video = cv2.VideoCapture(0)  # Camera 연결
# while True:
#     ret, frames = video.read()  # webcam에서 frame 단위로 image 가져옴
#     origin_frames = frames  # 모자이크 처리하기 위해서 원본 frame copy
#     frames = Image.fromarray(cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))  # frame을 RGB로 변경
#     boxes, probs = mtcnn.detect(frames)  # Frame별로 MTCNN Model을 이용해 Box와 확률을 가져옴

#     #################### FPS ####################
#     curTime = time.time()
#     sec = curTime - prevTime
#     prevTime = curTime
#     fps = 1 / (sec)
#     string = "FPS : %0.1f" % fps
#     cv2.putText(origin_frames, string, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
#     #################### FPS ####################
#     if start == False:  # Start가 False라면 Register 단계
#         if boxes is not None:  # Box가 존재하면
#             boxes = boxes.astype(int)  # Box의 값을 int로 변경
#             for box in range(len(boxes)):  # Box의 길이 만큼 Box정보가져옴
#                 if boxes[box][0] < 0: boxes[box][0] = 0  # Camera 밖으로 벗어나는 Box의 좌표를 Camera Size로 맞춤
#                 if boxes[box][1] < 0: boxes[box][1] = 0  # Camera 밖으로 벗어나는 Box의 좌표를 Camera Size로 맞춤
#                 if boxes[box][2] > 640: boxes[box][2] = 640  # Camera 밖으로 벗어나는 Box의 좌표를 Camera Size로 맞춤
#                 if boxes[box][3] > 480: boxes[box][3] = 480  # Camera 밖으로 벗어나는 Box의 좌표를 Camera Size로 맞춤
#                 cx = boxes[box][0]  # Box의 왼쪽 상단 X위치
#                 cy = boxes[box][1] + 20  # Box의 왼쪽 상단 Y위치 + 20
#                 crop = origin_frames  # Register Face를 저장하기 위해 Crop할 변수 선언
#                 crop = crop[boxes[box][1]:boxes[box][3], boxes[box][0]:boxes[box][2]]  # Crop될 얼굴은 예측된 Box의 Size로 저장
#                 im = cv2.resize(crop, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)  # 저장될 jpg의 Size는 128 x 128
#                 cv2.rectangle(origin_frames, (boxes[box][0], boxes[box][1]), (boxes[box][2], boxes[box][3]),
#                               color=(0, 0, 255), thickness=2)
#                 cv2.putText(origin_frames, 'Press Space', (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255))  #
#                 cv2.putText(origin_frames, 'Regist : ' + str(regist) + ', num : ' + str(number), (200, 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
#                 if cv2.waitKey(10) == ord('o'):  # Keyboard의 O를 누르면 다른사람을 저장할 수 있게 Regist + 1을 함
#                     regist = regist + 1
#                 if cv2.waitKey(10) == ord('r'):  # Keyboard의 Space를 누르면 해당하는 사람의 얼굴을 jpg로 저장
#                     numofregist = numofregist + 1
#                     number = number + 1
#                     name = 'person' + str(regist) + '_' + str(numofregist) + '.jpg'
#                     path = './data/RealTime/' + name
#                     cv2.imwrite(path, im)
#                 elif cv2.waitKey(10) == ord('s'):  # Keyboard의 S를 누르면 Masking System 시작
#                     for root, dirs, files in os.walk(
#                             './data/RealTime/'):  # For문을 통해 RegisterFace의 Embedding과 Name을 list로 저장
#                         for fname in files:
#                             dir = fname.split('_')[0]
#                             list = root + fname
#                             Em = get_embedding_path(list, model)
#                             register.append(Em)
#                             person.append(fname)

#                     start = True
#                     register = torch.FloatTensor(register)
#                     print('Register Face Finish')

#     elif start == True:  # Masking System이 시작됬다면
#         if boxes is not None:  # Box가 존재하면
#             boxes = boxes.astype(int)  # Box의 Type을 int로 변경
#             # 위와 동일함 #
#             for box in range(len(boxes)):
#                 if boxes[box][0] < 0: boxes[box][0] = 0
#                 if boxes[box][1] < 0: boxes[box][1] = 0
#                 if boxes[box][2] > 640: boxes[box][2] = 640
#                 if boxes[box][3] > 480: boxes[box][3] = 480
#                 crop = origin_frames
#                 crop = crop[boxes[box][1]:boxes[box][3], boxes[box][0]:boxes[box][2]]
#                 im = cv2.resize(crop, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
#                 im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#                 sim = 0
#                 if len(register) > 0:
#                     value = get_sim_all(im, model, register)  # Register Face와ss Frame에 Detect된 Face와 Cosine Similarity 비교
#                     sim = max(value) # Max 값 저장
#                     acc = "{:.4f}".format(sim[0])
#                 else :
#                     acc = sim
#                 cx = boxes[box][0]
#                 cy = boxes[box][1] + 20

#                 cv2.putText(origin_frames, acc, (boxes[abox][2] - 60, boxes[box][3] + 15), cv2.FONT_HERSHEY_DUPLEX, 0.7,
#                             (0, 255, 0))
#                 if sim > 0.6:  # Sim이 0.4 이상이면 Register Face
#                     cv2.putText(origin_frames, "register Person", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.7,
#                                 (255, 255, 255))
#                     cv2.rectangle(origin_frames, (boxes[box][0], boxes[box][1]), (boxes[box][2], boxes[box][3]),
#                                   color=(0, 0, 255), thickness=2)
#                 else:  # 그렇지 않으면 UnKnown
#                     face_img = origin_frames[boxes[box][1]:boxes[box][3], boxes[box][0]:boxes[box][2]]
#                     w = boxes[box][2] - boxes[box][0]
#                     h = boxes[box][3] - boxes[box][1]
#                     if w / 10 > 1 and h / 10 > 1:
#                         face_img = cv2.resize(face_img, ((int)(w / 10), (int)(h / 10)))  # UnKnown의 Box에 있는 img를 10배 축소
#                         face_img = cv2.resize(face_img, (w, h),
#                                               interpolation=cv2.INTER_AREA)  # 10배 축소한 얼굴을 다시 원본 Size로 Resize
#                         origin_frames[boxes[box][1]:boxes[box][3],
#                         boxes[box][0]:boxes[box][2]] = face_img  # Frame에 Masking 처리된 얼굴을 덮어씀
#                         cv2.putText(origin_frames, 'Unknown', (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
#                         cv2.rectangle(origin_frames, (boxes[box][0], boxes[box][1]+30), (boxes[box][2], boxes[box][3]),
#                                       color=(0, 255, 0), thickness=2)

#     cv2.imshow('Video', origin_frames)  # Webcam Show
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Keyboard의 Q를 누르면 종료
#         break

# video.release()
# cv2.destroyAllWindows()
