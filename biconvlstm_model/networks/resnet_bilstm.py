from networks.RESNET_encoder import ConvEncoder
from networks.classification import Classification
import torch.nn as nn

from networks.BiConvLSTM import BiConvLSTM
import numpy as np

class VP(nn.Module):
    def __init__(self, num_classes=2):
        super(VP, self).__init__()
        self.convenc = ConvEncoder()
        self.biconvlstm = BiConvLSTM(input_size=(7, 7), input_dim=2048, hidden_dim=2048, kernel_size=(3, 3), num_layers=1)
        self.classification = Classification(in_size=(7, 7), in_channels=2048, num_classes=num_classes)

    def forward(self, clips):
        # print('-------------------clips', np.shape(clips))

        clips_feature_maps = self.convenc(clips)
        # print('-------------------after enc clips_feature_maps', np.shape(clips_feature_maps),type(clips_feature_maps))
    
        clips_feature_maps = self.biconvlstm(clips_feature_maps)
        # print('-------------------clips_feature_maps : ', np.shape(clips_feature_maps),type(clips_feature_maps))


        # Max pool :)
        classification = self.classification(clips_feature_maps.max(dim=1)[0])
        return {'classification': classification}
