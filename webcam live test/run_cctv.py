import sys
import cv2
import numpy as np
import time
import requests
import os
from threading import Thread

second = int(sys.argv[1])
url = sys.argv[2]
video = 0 if sys.argv[3]=='0' else sys.argv[3]
class ThreadWithResult(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)

def send_frame(video_path, url):
    print(os.path.basename(video_path)+ ' video sent!!')
    resp = requests.post(url + '/predict', files={'file': open(video_path,'rb')})
    print(os.path.basename(video_path)+ ' video got answer------------')
    return resp.json()

def main():
    cap = cv2.VideoCapture(video)
    idx = 0
    total = 0
    frames = list()
    video_path = ''#'C:/Users/Seogki/GoogleDrive/데이터청년캠퍼스_고려대과정/child-abuse-detection/webcam live test/'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    filename = 'temp' + str(idx) + '.mp4'
    
    if os.path.isfile(filename):
        os.remove(filename)

    video_length = fps * second # 3초에 한번 영상 전송

    height = int(height/width * 640)
    width = 640
    out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print('video capture failed')
            break

        frame = cv2.resize(frame, dsize=(width,height), interpolation=cv2.INTER_LANCZOS4)

        out.write(frame)
        
        total += 1
        frames.append(frame)

        if total % video_length==0:
            out.release()

            request_thread = ThreadWithResult(target=send_frame, args=(video_path+filename, url))
            request_thread.start()
            

            idx = (idx + 1)%5
            filename = 'temp' + str(idx) + '.mp4'  
            if os.path.isfile(filename):
                os.remove(filename)
            out = cv2.VideoWriter(filename, fourcc,fps, (int(width), int(height)))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('CCTV',frame)

    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
