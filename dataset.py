import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

class MedMnist():
    def __init__(self):
        data_flag = 'pathmnist'
        download = True

    # NUM_EPOCHS = 3
        self.BATCH_SIZE = 128
    # lr = 0.001

        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])

        # preprocessing:数据变换为tensor，归一化
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data：训练集、测试集、未预处理的数据集用于可视化
        self.train_dataset = DataClass(split='train', transform=data_transform, download=download)
        self.test_dataset = DataClass(split='test', transform=data_transform, download=download)
        self.pil_dataset = DataClass(split='train', download=download)

        # encapsulate data into dataloader form
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.train_loader_at_eval = data.DataLoader(dataset=self.train_dataset, batch_size=2*self.BATCH_SIZE, shuffle=False)
        self.test_loader = data.DataLoader(dataset=self.test_dataset, batch_size=2*self.BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    ds = MedMnist()
    print(ds.train_dataset)
    print("=========")
    print(ds.test_dataset)