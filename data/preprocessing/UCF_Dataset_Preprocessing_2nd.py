import pandas as pd 
import re
import os
import cv2
import glob
import shutil
import numpy as np
from functools import reduce
from ffmpeg_subclip import ffmpeg_extract_subclip

def time_to_sec(time):
    start = re.find(r'\((\d+),',time)
    end = re.find(r',(\d+)\),',time)
    print(start,end)
    return [sum(np.array([60,1]) * np.array(start.split(':'), dtype=np.int8)), sum(np.array([60,1]) * np.array(end.split(':'), dtype=np.int8))]

def get_random_timesplit(s,e):
    time_arr = []
    tmp = 0
    while tmp<e:
        ran = np.random.rand() * 3 + 2
        tmp += ran
        time_arr.append([s,tmp])
        s += ran
    return time_arr[:-1]

def main():
    video_info_path = 'C:/Users/Seogki/Downloads/0819 UCF 데이터 재라벨링 - 시트1.csv'
    df = pd.read_csv(video_info_path)
    df.drop('Unnamed: 5', axis=1, inplace=True)
    null_df = df[df['info'].map(lambda x: str(x).startswith('x'))]
    child_df = df[df['child'].map(lambda x: str(x).startswith('o'))]
    no_x_df = df[df['info'].map(lambda x: not str(x).startswith('x'))]
    crop_df = no_x_df[no_x_df.child.map(lambda x: not str(x).startswith('o'))]

    crop_df.drop(50, inplace=True)
    crop_df.drop(columns=['child','info'],inplace=True)
    crop_df.fillna('x',inplace=True)

    for item in crop_df.iterrows():
        fi = item[1].fight
        nofi = item[1].nofight
        print(item[1].filename)
        if fi == 'x' and nofi == 'x':
            print(item)
        regex = r'(\(\d:\d{2}.\d:\d{2}\))'
        print('--------------')
        fi_time = re.findall(regex,fi)
        print('fi time', fi_time)
        nofi_time = re.findall(regex,nofi)
        print('no fi time', nofi_time)
        print(re.findall(r'\(([\d ]+:[\d ]+),([\d ]+:[\d ]+)\)',fi))
        print(re.findall(r'\(([\d ]+:[\d ]+),([\d ]+:[\d ]+)\)',nofi))

    full_vid= 'C:/Users/Seogki/Downloads/Anomaly-Videos-Part-1_Abuse/'
    fight_path = 'C:/Users/Seogki/Downloads/UCF_Cropped/fight/' 
    c_fight_path = 'C:/Users/Seogki/Downloads/UCF_Cropped/cropped_fight/' 
    nofight_path = 'C:/Users/Seogki/Downloads/UCF_Cropped/noFight/' 
    c_nofight_path = 'C:/Users/Seogki/Downloads/UCF_Cropped/cropped_noFight/' 

    fight_header = 'AA-V-UCF-'
    nofight_header = 'AA-N-UCF-'

    vid_list = glob.glob(full_vid + '**/*.mp4', recursive=True)


    pattern = r'\(([\d ]+:[\d ]+),([\d ]+:[\d ]+)\)'
    out_path = []
    for item in crop_df.iterrows():
        #print(item[1].filename)
        video = None
        for vid in vid_list:
            if item[1].filename in vid:
                video = vid
        
        source = video
        print('-----------------\n',video,'\n-------------------')
        #print(item[1].fight, item[1].nofight)
        dst_name = os.path.basename(video).split('.mp4')[0]
        # fight
        index = 1
        for start, end in re.findall(pattern,item[1].fight):
            start = float(start.split(':')[0]) * 60 + float(start.split(':')[1])
            end = float(end.split(':')[0]) * 60 + float(end.split(':')[1])
            print(start,end)
            assert end > start
            dst = fight_path + fight_header + dst_name + '_' + str(index)+ '.mp4'
            out_path.append(dst)
            ffmpeg_extract_subclip(source, int(start), int(end), targetname=dst)
            #print(dst)
            index += 1
        index = 1
        # nofight
        for start, end in re.findall(pattern,item[1].nofight):
            start = float(start.split(':')[0]) * 60 + float(start.split(':')[1])
            end = float(end.split(':')[0]) * 60 + float(end.split(':')[1])
            print(start, end)
            assert end > start
            dst = nofight_path + nofight_header + dst_name + '_' + str(index)+ '.mp4'
            out_path.append(dst)
            #print(dst)
            ffmpeg_extract_subclip(source, int(start), int(end), targetname=dst)
            index += 1

    fight_vid = glob.glob(fight_path + '**/*.mp4', recursive=True)

    for fight in fight_vid:
        cap = cv2.VideoCapture(fight)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cropped = c_fight_path + os.path.basename(fight).split('.mp4')[0]
        print(fight, int(frame_count/fps))
        if frame_count / fps < 7:
            #print(int(frame_count/fps),' copy ', cropped)
            shutil.copy(fight,cropped +'.mp4')
        else:
            timesplit = get_random_timesplit(0,frame_count/fps)
            index = 1
            for tt in timesplit:
                start = tt[0]
                end = tt[1]
                print(start,end,cropped + '_' + str(index) + '.mp4')
                ffmpeg_extract_subclip(fight,int(start),int(end),cropped + '_' + str(index) + '.mp4')
                index+=1
        print('------')
            
    nofight_vid = glob.glob(nofight_path + '**/*.mp4', recursive=True)
    for fight in nofight_vid:
        cap = cv2.VideoCapture(fight)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cropped = c_nofight_path + os.path.basename(fight).split('.mp4')[0]
        print(fight, int(frame_count/fps))
        if frame_count / fps < 7:
            #print(int(frame_count/fps),' copy ', cropped)
            shutil.copy(fight,cropped+'.mp4')
        else:
            timesplit = get_random_timesplit(0,frame_count/fps)
            index = 1
            for tt in timesplit:
                start = tt[0]
                end = tt[1]
                if end-start < 1:
                    break
                print(start,end,cropped + '_' + str(index) + '.mp4')
                ffmpeg_extract_subclip(fight,int(start),int(end),cropped + '_' + str(index) + '.mp4')
                index+=1
        print('------')
        