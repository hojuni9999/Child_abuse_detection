import glob
import re
import math
import pandas as pd
from ffmpeg_subclip import ffmpeg_extract_subclip

def isNaN(string):
    return string == 'na' or string == 'x'

def find_video(path, filename):
    file_list = os.listdir(path)
    filefound = list(filter(lambda x:x.startswith(filename), file_list))
    print(filefound)
    if len(filefound) == 0:
        print('{} / file not found in path / {}'.format(filename,path))
        return -1
    if len(filefound) > 1:
        print(str(filefound) , ' / more than one file matched')
        return -1
    return path + filefound[0]

def ucf_data_subclip_extract(abuse_path, assault_path, save_path, label, dst_name):
    out_path = list()
    for video in label.itertuples():
        pattern = r'\(([\d ]+:[\d ]+),([\d ]+:[\d ]+)\)'
        index = 1
        if video.filename.startswith('Abuse'):
            source = find_video(abuse_path, video.filename)
        elif video.filename.startswith('Assault'):
            source = find_video(assault_path, video.filename)
        else:
            break

        if source == -1:
            break

        print(source)
        for start, end in re.findall(pattern,video.body):
            start = float(start.split(':')[0]) * 60 + float(start.split(':')[1])
            end = float(end.split(':')[0]) * 60 + float(end.split(':')[1])
            print(start, end)
            assert end > start
            dst = save_path + video.filename + dst_name + '.mp4'
            out_path.append(dst)

            ffmpeg_extract_subclip(source, start, end, targetname=dst)
            index += 1

        for start, end in re.findall(pattern,video.tools):
            start = float(start.split(':')[0]) * 60 + float(start.split(':')[1])
            end = float(end.split(':')[0]) * 60 + float(end.split(':')[1])
            print(start, end)
            assert end > start
            dst = save_path + video.filename + dst_name + '.mp4' 
            out_path.append(dst)

            ffmpeg_extract_subclip(source, start, end, targetname=dst)
            index += 1
        
        print('Sucessfully Finished - ',video.filename)
    return out_path

def main():
    label_path = '/content/mnt/Shareddrives/2021청년인재_고려대과정_10조/데이터 라벨링/괴도키즈_데이터_라벨링 - UCF-Crime Data.csv'
    abuse_path = '/content/mnt/Shareddrives/2021청년인재_고려대과정_10조/Data/UCF-Crime/Anomaly-Videos-Part-1_Abuse/'
    assault_path = '/content/mnt/Shareddrives/2021청년인재_고려대과정_10조/Data/UCF-Crime/Anomaly-Videos-Part-1_Assault/'
    save_path = '/content/mnt/Shareddrives/2021청년인재_고려대과정_10조/Data/UCF-Crime/UCF-Crime_Dataset/'

    # Crop Fight Subclip

    label = pd.read_csv(label_path)
    label.columns = ['filename','body','tools','information','seen_by']
    label.drop('seen_by',axis=1, inplace=True)
    label = label.fillna('na')
    label.head()

    label = label[label['information'].map(lambda info: not str(info).startswith('x'))]                 # information이 x로 시작하는 row 제거
    label = label[label['filename'].map(lambda info: not isNaN(info))]                                  # filename 없는 row 제거
    label['body'] = label['body'].map(lambda x: str(x).strip())                                         # body column 속 whitespace 제거
    label = label[~((label['body'].map(lambda x:isNaN(x))) & (label['tools'].map(lambda x:isNaN(x))))]  # body와 tools 모두 nan인 row 제거

    for video in label.itertuples():
        if isNaN(video.body) and isNaN(video.tools): 
            print(video.Index,video.filename,video.body, video.tools, video.information)

    for video in label.itertuples():
        print('---------------------------- ',video.Index)
        print(video.body, video.tools)
        if not isNaN(video.body):
            print(re.findall(r'\(([\d ]+:[\d ]+),([\d ]+:[\d ]+)\)',video.body))
        if not isNaN(video.tools):
            print(re.findall(r'\(([\d ]+:[\d ]+),([\d ]+:[\d ]+)\)',video.tools))
    
    ucf_subclip_list = ucf_data_subclip_extract(abuse_path, assault_path, save_path, label, '_'.join(['',str(index)]))

    # Crop Non-Fight Subclip

    non_fight_label_path = '/content/mnt/Shareddrives/2021청년인재_고려대과정_10조/데이터 라벨링/괴도키즈_데이터_라벨링 - UCF-Crime_nonfight_Data.csv'
    non_fight_label = pd.read_csv(non_fight_label_path)
    non_fight_label.columns = ['filename', 'nonfight', 'maybe', 'nonfightinformation','body','tools','information']

    non_fight_label.drop(['body','tools','information'], axis=1, inplace=True)
    non_fight_label = non_fight_label.fillna('na')

    non_fight_label = non_fight_label[non_fight_label['nonfightinformation'].map(lambda info: not str(info).startswith('x'))]                       # information이 x로 시작하는 row 제거
    non_fight_label = non_fight_label[non_fight_label['filename'].map(lambda info: not isNaN(info))]                                                # filename 없는 row 제거   
    non_fight_label['nonfight'] = non_fight_label['nonfight'].map(lambda x: str(x).strip())                                                         # nonfight column 속 whitespace 제거
    non_fight_label = non_fight_label[~((non_fight_label['nonfight'].map(lambda x:isNaN(x))) & (non_fight_label['maybe'].map(lambda x:isNaN(x))))]  # body와 tools 모두 nan인 row 제거
    non_fight_label['nonfight'] = non_fight_label['nonfight'].map(lambda x: re.sub(';', ':', x))                                                    # 오타 제거
    non_fight_label['maybe'] = non_fight_label['maybe'].map(lambda x: re.sub(';', ':', x))

    for video in non_fight_label.itertuples():
        #print(video.body, video.tools)

        print('---------------------------- ',video.Index)
        print(video.nonfight, video.maybe)
        if not isNaN(video.nonfight):
            print(re.findall(r'\(([\d ]+:[\d ]+),([\d ]+:[\d ]+)\)',video.nonfight))
        if not isNaN(video.maybe):
            print(re.findall(r'\(([\d ]+:[\d ]+),([\d ]+:[\d ]+)\)',video.maybe))

    ucf_data_subclip_extract(abuse_path, assault_path, save_path, non_fight_label, '_'.join(['','nofi',str(index)]))