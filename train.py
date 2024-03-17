from src.utils.transforms import Resize, Normalization
from src.utils.labelConverter import CTCLabelConverter
from src.utils.loss_calculator import calc_ctc_loss
from src.nn.ocr_model import RecognizerNetwork
from src.utils.dataset import DataGenerator
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from src.utils.misc import printParams
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

model = RecognizerNetwork(cfg).to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=cfg.betas)
printParams(model, "OCR model trainable params : {:,}")

converter = CTCLabelConverter(cfg)

for epoch in range(cfg.epochs):
    loop = tqdm.tqdm(trian_dataloader)
    for batch_indx, (img, label) in enumerate(loop):
        label, len_gt = converter.encode(label)
        img = img.to(device)
        label = label.to(device)
        len_gt = len_gt.to(device)
        
        
        model.zero_grad()
        
        preds = model(img)
        
        loss = calc_ctc_loss(preds, label, len_gt)
        
        loss.backward()
        opt.step()
        
        __log = {
            "epoch" : epoch + 1,
            "loss" : loss.item(),
        }
        loop.set_postfix(__log)
torch.save(model.state_dict(), "ocr.pth")