import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset

class CIFAR10_train(Dataset):
    def __init__(self):
        super().__init__()
        self.trainset = datasets.CIFAR10(root='./data/', train=True, download=True)
        self.img_convert = transforms.Compose([
            transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, index):
        img, label = self.trainset[index]
        return self.img_convert(img)/255.0, label

class CIFAR10_test(Dataset):
    def __init__(self):
        super().__init__()
        self.trainset = datasets.CIFAR10(root='./data/', train=True, download=True)
        self.img_convert = transforms.Compose([
            transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, index):
        img, label = self.trainset[index]
        return self.img_convert(img)/255.0, label

if __name__=='__main__':

    ds = CIFAR10_train()
    print(len(ds))
    img, label = dsc[0]
    print(img.shape)
    print(label)




