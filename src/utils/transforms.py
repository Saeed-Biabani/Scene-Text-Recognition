import torch
from torchvision.transforms import functional as F

class Normalization(object):
    def __call__(self, image):
        image = (
            image - image.mean()
        ) / image.std()
        return image
    
class Rescale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image):
        return image * self.scale

class Resize(object):
    def __init__(self, to_size):
        self.to_size = to_size

    def __resizeimage__(self, image : torch.Tensor):
        return F.resize(
            image,
            self.to_size,
            interpolation = F.InterpolationMode.BICUBIC,
            antialias = True
        )

    def __call__(self, image):
        return self.__resizeimage__(image)