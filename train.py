import os
import torch
import numpy as np
from vit import ViT
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import MedMnist
from tqdm import tqdm

# 设备
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

# 数据集
dataset = MedMnist()

# 模型
model = ViT().to(DEVICE)

# 加载模型
try:    
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 

# 优化器
optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)

'''
训练模型
'''

EPOCH=5
BATCH_SIZE=dataset.BATCH_SIZE

for epoch in range(EPOCH):
    for imgs,label in tqdm(dataset.train_loader):
        label = np.transpose(label).reshape(-1)
        # 真实输出
        logits = model(imgs.to(DEVICE))

        # 交叉熵函数
        loss = F.cross_entropy(logits, label.to(DEVICE))
        # 反向传播前梯度清零
        optimzer.zero_grad()
        # 反向传播得梯度
        loss.backward()
        # 根据梯度更新优化器参数
        optimzer.step()

    print('epoch:{} loss:{}'.format(epoch, loss))
    torch.save(model.state_dict(), '.model.pth')
    os.replace('.model.pth', 'model.pth')
    


