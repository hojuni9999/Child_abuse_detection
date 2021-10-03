import glob
import re
import random

import numpy as np
import cv2
from torch.utils.data import Dataset

from data.data_label_factory import label_factory
def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()  
        if not ret:
            break
        frames.append(frame)
        #print(np.array(frame).shape)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames, we stop
            break
    cap.release()
    video = np.stack(frames)
    return video

class DatasetReader(Dataset):
    def __init__(self, root_dir, data_name):
        super(DatasetReader, self).__init__()

        self.root_dir = root_dir

        # regex = re.compile(r'.*\.mp4$')
        # data_files = list(filter(regex.search, glob.glob(self.root_dir + '\**', recursive=True)))
        # data_files = glob.glob(self.root_dir + '/**.mp4', recursive=True)
        # data_files.append(glob.glob(self.root_dir + '/**.avi', recursive=True)) #append하시면 안됩니다!
        data_files = glob.glob(root_dir + '/**/*.mp4', recursive=True)
        data_files += (glob.glob(root_dir + '/**/*.avi', recursive=True))
        #print('printing  data_Files \n',len(data_files),'\n',data_files)
        random.shuffle(data_files)
        data_labeler = label_factory(data_name)
        self.labeled_data = data_labeler(data_files)

    def __len__(self):
        return len(self.labeled_data)

    def __getitem__(self, idx):
        data_file, label = self.labeled_data[idx]
        #print('run read video of ',data_file)
        videodata = read_video(data_file)
        #print('finished read video')
        
        return (videodata, label)
