import glob
import re
import random
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

from data.data_label_factory import label_factory

def read_video(filename):
    frames = []
    if not os.path.isfile(filename):
        print('file not found')
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()  
        if not ret:
            break
        frames.append(frame)
    cap.release()
    video = np.stack(frames)
    return video
class DatasetReader(Dataset):
    def __init__(self, data):
        super(DatasetReader, self).__init__()
        self.data = [data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        videodata = self.data[idx]
        return videodata