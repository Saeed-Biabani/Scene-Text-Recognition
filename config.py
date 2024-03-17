import torch
import os
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_ds_info(ds_path_list):
    max_str_len = 0
    total = []
    for path_ in ds_path_list:
        with open(
            os.path.join(
                path_, "labels.json"
            ), 'r'
        ) as file_:
            labels = json.load(file_)
            total += list(labels.values())

    max_str_len = len(
        max(total, key = lambda x : len(x))
    )
    dict_ = ''.join(
        np.unique(list(''.join(total)))
    ).replace(' ', '') + '_'

    return dict_, max_str_len

ds_path = {
    "train_ds" : "path/to/train/dataset",
    "test_ds" : "path/to/test/dataset",
    "val_ds" : "path/to/val/dataset",
}
dict_, max_str_len = extract_ds_info(ds_path.values())
batch_size = 32
img_h = 64
img_w = 192
img_channel = 1
learning_rate = 3e-4
betas = (0, 0.999)
epochs = 500