import torch
from torch.nn import functional as nnF

def calc_ctc_loss(preds, targets, tg_len):
    preds_size = torch.IntTensor([preds.size(1)] * preds.size(0))
    preds = preds.log_softmax(2).permute(1, 0, 2)
    cost = nnF.ctc_loss(preds, targets, preds_size, tg_len, zero_infinity = True, blank = 0)
    return cost