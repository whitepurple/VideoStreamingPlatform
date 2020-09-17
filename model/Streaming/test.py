import ffmpeg
import numpy as np
import cv2
width = 480
height = 640

# process1 = (
#     ffmpeg
#     .input('rtmp://218.150.183.59:1935/key/testkey')
#     .output('pipe:', format='rawvideo', pix_fmt='bgr24')
#     .run_async(pipe_stdout=True)
# )

in_stream = ffmpeg.input('rtmp://218.150.183.59:1935/key/testkey')

audio_stream = in_stream.audio

video_stream = in_stream
video_stream = ffmpeg.output(in_stream, 'pipe:', format='rawvideo', pix_fmt='bgr24')
video_stream = ffmpeg.run_async(video_stream, pipe_stdout=True)

out_stream = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
# out_stream = ffmpeg.map_audio(out_stream, audio_stream)
out_stream = ffmpeg.output(out_stream, audio_stream, 'rtmp://218.150.183.59:1935/encode/ssbkey',
                vcodec='libx264', 
                acodec='copy', 
                pix_fmt='yuv420p', 
                preset='ultrafast', 
                # r='20', 
                # g='50', 
                video_bitrate='2500k', 
                format='flv')

out_stream = ffmpeg.run_async(out_stream, pipe_stdin=True)

# process2 = (
#     ffmpeg
#     .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
#     .output('rtmp://218.150.183.59:1935/encode/ssbkey',vcodec='libx264', pix_fmt='yuv420p', preset='veryfast', r='20', g='50', video_bitrate='2500k', format='flv')
#     .run_async(pipe_stdin=True)
# )

while True:
    in_bytes = video_stream.stdout.read(width * height * 3)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([height, width, 3])
    )
    # cv2.imwrite("In.jpg",in_frame)
    # See examples/tensorflow_stream.py:
    #out_frame = deep_dream.process_frame(in_frame)
    out_frame = in_frame


    
    out_stream.stdin.write(
        out_frame
        .astype(np.uint8)
        .tobytes()
    )

out_stream.stdin.close()
in_stream.wait()
video_stream.wait()
audio_stream.wait()
out_stream.wait()