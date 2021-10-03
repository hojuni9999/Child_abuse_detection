import numpy as np
import cv2
import os
import glob
import imutils
import time
import re
import ffmpeg
from converter import Converter
from moviepy.tools import subprocess_call
from moviepy.config import get_setting

def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
    the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    cmd = [get_setting("FFMPEG_BINARY"),"-y",
           "-ss", "%0.2f"%t1,
           "-i", filename,
           "-t", "%0.2f"%(t2-t1),
           "-vcodec", "copy", "-acodec", "copy", targetname]

    subprocess_call(cmd)
def video_resize2(vid_path, output_path, width, overwrite = False):
    '''
    use ffmpeg to resize the input video to the width given, keeping aspect ratio
    '''
    if not( os.path.isdir(os.path.dirname(output_path))):
        raise ValueError(f'output_path directory does not exists: {os.path.dirname(output_path)}')

    stream = ffmpeg.input(vid_path)
    stream = ffmpeg.filter(stream, 'scale', width, -2)
    #stream = ffmpeg.filter(stream, 'fps', fps=24, round='up')
    stream = ffmpeg.output(stream, output_path)
    ffmpeg.run(stream)
    return output_path

def main():
    path = "D:/AIHub/이상행동 CCTV 영상/01.폭행(assult)" 
    video_path = os.listdir(path)
    
    ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'
    ffprobe_path = r'C:\ffmpeg\bin\ffprobe.exe'
    
    conv = Converter(ffmpeg_path, ffprobe_path)
    video_files = glob.glob(path +'/**/*.mp4', recursive=True)
    out_name = '.'.join(video_files[1].split('.')[:-1]) + '_TEST.mp4'
    info = conv.probe(video_files[0])

    width, height, frame_rate = re.findall(r'width=(\d+), height=(\d+), fps=(\d+)', str(info.streams[0]))[0]
    
    out_width = width/2
    out_height = height/2

    convert = conv.convert(video_files[0],out_name , {
        'format': 'mp4',
        'audio': {
            'codec': 'aac',
            'samplerate': 11025,
            'channels': 2
        },
        'video': {
            'codec': 'hevc',
            'width': out_width,
            'height': out_height,
            'fps': 24
        }})

    save_path = 'C:/Users/Seogki/Desktop/테스트영상/'
    for path in video_path:
        ffmpeg_extract_subclip(path,5.0,89,save_path)   