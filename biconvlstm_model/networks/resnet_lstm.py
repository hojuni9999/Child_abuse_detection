from networks.RESNET_encoder import ConvEncoder
from networks.classification import Classification
import torch.nn as nn

from networks.ConvLSTM import ConvLSTM


class VP(nn.Module):
    def __init__(self, num_classes=2):
        super(VP, self).__init__()
        self.convenc = ConvEncoder()
        self.convlstm = ConvLSTM(input_size=(7, 7), input_dim=2048, hidden_dim=2048, kernel_size=(3, 3), num_layers=1)
        self.classification = Classification(in_size=(7, 7), in_channels=2048, num_classes=num_classes)

    def forward(self, clips):
        clips_feature_maps = self.convenc(clips)

        clips_feature_maps = self.convlstm(clips_feature_maps)

        # Max pool :)
        classification = self.classification(clips_feature_maps.max(dim=1)[0])
        return {'classification': classification}
