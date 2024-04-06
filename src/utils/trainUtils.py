from .loss_calculator import calc_ctc_loss
import numpy as np
import tqdm

def trainOneEpoch(model, loader, converter, opt, device, epoch):
    model.train()
    losses = []
    loop = tqdm.tqdm(loader, colour = "yellow")
    for batch_indx, (img, label) in enumerate(loop):
        label, len_gt = converter.encode(label)
        img = img.to(device)
        label = label.to(device)
        len_gt = len_gt.to(device)
        
        
        model.zero_grad()
        
        preds = model(img)
        
        loss = calc_ctc_loss(preds, label, len_gt)
        losses.append(loss.cpu().detach().item())
        
        loss.backward()
        opt.step()
        
        __log = {
            "epoch" : epoch + 1,
            "loss" : loss.item(),
        }
        loop.set_postfix(__log)
    return np.mean(losses)


def testOneEpoch(model, loader, converter, device, epoch):
    model.eval()
    losses = []
    loop = tqdm.tqdm(loader, colour = "green")
    for batch_indx, (img, label) in enumerate(loop):
        label, len_gt = converter.encode(label)
        img = img.to(device)
        label = label.to(device)
        len_gt = len_gt.to(device)
        
        
        model.zero_grad()
        
        preds = model(img)
        
        loss = calc_ctc_loss(preds, label, len_gt)
        losses.append(loss.cpu().detach().item())

        __log = {
            "epoch" : epoch + 1,
            "loss" : loss.item(),
        }
        loop.set_postfix(__log)
    return np.mean(losses)