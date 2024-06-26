# data_loader.py
# author r,y,x,m
import torch.utils.data
from torchvision import datasets, transforms

def get_data_loaders(data_dir, batch_size=32):
    # 定义数据集的变换
    train_transforms = transforms.Compose([
        # 随机裁剪成244x244的大小
        transforms.RandomResizedCrop(224),
        # 随即水平翻转
        transforms.RandomHorizontalFlip(),
        # 转换为多维矩阵(张量)
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 定义验证集的变换
    val_transforms = transforms.Compose([
        # 调整图片大小为256x256
        transforms.Resize(256),
        # 中心裁剪成224x224的大小
        transforms.CenterCrop(224),
        # 转换为张量
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建训练集数据集
    train_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=train_transforms)
    # 创建验证集数据集
    val_dataset = datasets.ImageFolder(root=data_dir + '/validation', transform=val_transforms)
    # 创建训练集加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 创建验证集加载器
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
