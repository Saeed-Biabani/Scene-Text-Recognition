from torch import nn

class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels
    ):
        super(VGGFeatureExtractor, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(True)
        )

    def forward(self, input):
        return self.ConvNet(input)