from src.utils.misc import printParams
from src.nn.layers import BidirectionalLSTM
import config as cfg
import torch
device = cfg.device

model = BidirectionalLSTM(512, 1024, 512).to(device)
printParams(model, "OCR model trainable params : {:,}")
# inp = torch.rand((1, 1, 64, 192), device = device)

# out = model(inp)

# print(out.shape)