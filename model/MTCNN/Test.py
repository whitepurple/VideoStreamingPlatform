import cv2
import numpy as np
#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer
video_path="http://218.150.183.58/live/testuser/index.m3u8"

def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()

    cv2.destroyAllWindows()
PlayVideo(video_path)