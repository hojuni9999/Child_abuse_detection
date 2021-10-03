import os
import sys
import cv2
import glob
import time
import ffmpeg
import imutils
import numpy as np

def video_resize_opencv(target, out, out_width=720):
    cap = cv2.VideoCapture(target)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 영상의 넓이(가로) 프레임
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 영상의 높이(세로) 프레임
    frameRate = int(cap.get(cv2.CAP_PROP_FPS)) 
    # print(width, height, frameRate)
    
    # 비디오 저장 변수
    writer = None
    
    while True:
        # ret : 성공적으로 불러왔는지 확인
        # frame : 읽어온 frame 정보    
        ret, frame = cap.read()
        
        # 읽은 Frame이 없는 경우 종료
        if not ret:
            break

        # Frame Resize
        frame = imutils.resize(frame, width=out_width, inter=cv2.INTER_LANCZOS4)
        # cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # 저장할 video 설정
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MJPG
            writer = cv2.VideoWriter(out, fourcc, frameRate, (frame.shape[1], frame.shape[0]), True)
        
        # 비디오 저장
        if writer is not None:
            writer.write(frame)
    
    cap.release()

def video_resize_ffmpeg(input_path, output_path, out_width=720):
    '''
    use ffmpeg to resize the input video to the width given, keeping aspect ratio
    '''

    stream = ffmpeg.input(input_path)
    stream = ffmpeg.filter(stream, 'scale', out_width, -2)
    #stream = ffmpeg.filter(stream, 'fps', fps=24, round='up')
    stream = ffmpeg.output(stream, output_path)
    ffmpeg.run(stream)

def main():
    '''
    finds mp4 file and create resized video with same aspect ratio and width
    resized video file name will have '_resized_width' at the end
    
    how to use :
    python video_resize.py "path" "width" "version"

    path : video path (search mp4 recursively)
    width : output width
    version: 'ffmpeg' or 'opencv' 
    '''

    path = sys.argv[1] # "D:/AIHub/이상행동 CCTV 영상/01.폭행(assult)" 
    width = int(sys.argv[2])
    # fps = int(sys.argv[3])
    version = sys.argv[3]
    mp4_path_list = glob.glob(path +'/**/*.mp4', recursive=True)
    video_count = len(mp4_path_list)

    print('number of mp4 files : ', video_count)
    for idx, filepath in enumerate(mp4_path_list):
        #print('FILE PATH : ', filepath)
        filename = filepath.split('\\')[-1].split('.')[0]
        file_format = filepath.split('.')[-1]
        file_path = filepath.split(filename)[0]
        #print(filename,file_format,file_path)
        if filename[-11:] == 'resized_720':
            continue
        if file_format == 'mp4' or file_format == 'avi':
            start_time = time.time()
            out_path = file_path + filename + '_resized_' + str(width) + '.' + file_format
            if not os.path.isfile(out_path):
                print('**** working on " '+ filename +' " ****')

                if version=='opencv':
                    video_resize_opencv(filepath, out_path, out_width=width)
                elif version == 'ffmpeg':
                    video_resize_ffmpeg(filepath, out_path, out_width=width)

            print('finished : {} out of {}'.format(idx+1,video_count))
            print('time took : ', round(time.time() - start_time,3) , ' sec')

if __name__=='__main__':
    main()