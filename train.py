import os
import torch
from vit import ViT
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CIFAR10_train

# 设备
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

# 数据集
dataset = CIFAR10_train()

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

EPOCH=100
BATCH_SIZE=64

# 数据加载器
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, persistent_workers=True)  # 使用数据加载器迭代后

iter_count = 0
for epoch in range(EPOCH):
    for imgs,label in dataloader:
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

        if iter_count%1000 == 0:
            print('epoch:{} iter:{} loss:{}'.format(epoch, iter_count, loss))
            torch.save(model.state_dict(), '.model.pth')
            os.replace('.model.pth', 'model.pth')
        iter_count += 1



