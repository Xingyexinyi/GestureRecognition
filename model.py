# model.py
# author r,y,x,m
import torch.nn as nn
import torchvision.models as models

class GestureClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(GestureClassifier, self).__init__()
        # 使用预训练的Resnet-18模型
        self.model = models.resnet18(pretrained=True)
        # 获取全连接层的输入特征数
        num_ftrs = self.model.fc.in_features
        # 替换全连接层以适应新的分类任务
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # 定义向前传播
        return self.model(x)
