import os
import random
import torch
import numpy as np 

import torch.nn as nn
import torch.optim
import torch.utils.data
from model import *
from main import network_factory
from torchvision import transforms

from data.data_reader import DatasetReader, read_video
from data.data_splitter import DatasetSplit
from data.data_transformer import DatasetTransform
from data.transforms import SelectFrames, FrameDifference, Downsample, TileVideo, RandomCrop, Resize, RandomHorizontalFlip, Normalize, ToTensor

def load_model(args, model_path):
    seed = 250

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model '{}'".format(args['arch']))
    VP = network_factory(args['arch'])

    model = VP()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args['lr'],
                                    weight_decay=args['weight_decay'])

    # optionally resume from a checkpoint
    if os.path.isfile(args['evalmodel']):
        print("=> loading checkpoint '{}'".format(args['evalmodel']))
        checkpoint = torch.load(args['evalmodel'])
        args['start_epoch'] = checkpoint['epoch']
        print('start_epoch : ', args['start_epoch'])
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args['evalmodel'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['evalmodel']))

    return model

def data_loader(args, video):
    val_dataset = DatasetReader(video)
    val_transformations = transforms.Compose([Resize(size=224), SelectFrames(num_frames=args['frames']), FrameDifference(dim=0), Normalize(), ToTensor()])

    val_dataset = DatasetTransform(val_dataset, val_transformations)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    return val_loader