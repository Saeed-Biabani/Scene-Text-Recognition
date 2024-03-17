from torch import nn
from .feature_extractor import VGGFeatureExtractor
from .layers import BidirectionalLSTM

class RecognizerNetwork(nn.Module):
    def __init__(self, cfg):
        super(RecognizerNetwork, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(cfg.img_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 512, 1024),
            BidirectionalLSTM(512, len(cfg.dict_), 1024),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.avgpool(features.permute(0, 3, 1, 2)).squeeze(3)
        rnn_out =  self.rnn(features)
        return rnn_out