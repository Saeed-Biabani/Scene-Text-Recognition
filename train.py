from src.utils.trainUtils import trainOneEpoch, testOneEpoch
from src.utils.misc import printParams, WeightInitializer
from src.utils.transforms import Resize, Normalization
from src.utils.labelConverter import CTCLabelConverter
from src.nn.ocr_model import RecognizerNetwork
from src.utils.dataset import DataGenerator
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import config as cfg
import torch
import tqdm

def printConfigVars(module, fname):
    pa = [item for item in dir(module) if not item.startswith("__")]
    for item in pa:
        value = eval(f'{fname}.{item}')
        if str(type(value)) not in ("<class 'module'>", "<class 'function'>"):
            print(f"{fname}.{item} : {eval(f'{fname}.{item}')}")

printConfigVars(cfg, 'cfg')

device = cfg.device

trainds = DataGenerator(
    root = cfg.ds_path["train_ds"],
    transforms = Compose([
        Resize((cfg.img_h, cfg.img_w)),
        Normalization()
    ])
); trian_dataloader = DataLoader(trainds, cfg.batch_size, True)

testds = DataGenerator(
    root = cfg.ds_path["test_ds"],
    transforms = Compose([
        Resize((cfg.img_h, cfg.img_w)),
        Normalization()
    ])
); test_dataloader = DataLoader(testds, cfg.batch_size, True)

model = RecognizerNetwork(cfg).to(device)
WeightInitializer(model)
opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=cfg.betas)
printParams(model, "OCR model trainable params : {:,}")

converter = CTCLabelConverter(cfg)

for epoch in range(cfg.epochs):
    trainOneEpoch(
        model,
        trian_dataloader,
        converter, opt,
        device, epoch
    )
    testOneEpoch(
        model,
        test_dataloader,
        converter, device, epoch
    )
torch.save(model.state_dict(), "ocr.pth")