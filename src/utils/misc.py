import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
import json

def printParams(model, text):
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))
    print(text.format(sum(params_num)))

def WeightInitializer(model):
    for name, param in model.named_parameters():
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:
            if 'weight' in name:
                param.data.fill_(1)
            continue

def loadJson(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def plotHistory(history):
    plt.figure(figsize = (10, 5))
    plt.title("Learning Curve")

    plt.plot(history["train"], 'red')
    plt.plot(history["val"], 'green')

    plt.ylabel("CTC Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc = "upper right")
    plt.savefig("figures/LearningCurve.png")
    plt.close()