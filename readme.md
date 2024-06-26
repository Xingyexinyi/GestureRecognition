# 手势分类项目

这是一个用于训练和评估手势分类模型的项目。该项目使用PyTorch和Torchvision库，通过迁移学习的方法，基于预训练的ResNet-18模型进行训练。
（本项目数据集来自AiStudio中的免费数据集，网址https://aistudio.baidu.com/datasetdetail/87430）
## 目录结构

```
.
├── dataloader.py         # 数据加载器脚本
├── model.py              # 模型定义脚本
├── train.py              # 训练脚本
├── gesture_classifier.pth # 已训练模型的权重（如果有,没有第一次运行会自动生成在根目录）
└── dataset               # 数据集目录
    ├── train             # 训练数据集
    └── validation        # 验证数据集
```

## 安装依赖

在运行本项目之前，请确保你已经安装了以下依赖：

```bash
pip install torch torchvision matplotlib
```

## 数据准备

将你的图像数据集组织如下结构：

```
dataset
├── train
│   ├── class1
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2
│   └── ...
└── validation
    ├── class1
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── class2
    └── ...
```

每个子目录（如`class1`，`class2`）对应一个类别，目录中包含该类别的所有图像。

## 使用方法

### 训练模型

你可以运行`train.py`脚本来训练模型。确保你已经在`train.py`中设置了正确的数据集目录和类别数量。

```bash
python train.py
```

训练过程中，脚本会打印每个epoch的训练损失和验证准确率，并在训练结束后保存模型权重到`gesture_classifier.pth`。

### 加载并继续训练

如果你之前已经训练过模型并保存了权重，可以加载已有的模型权重继续训练。`train.py`会自动尝试加载`gesture_classifier.pth`文件。

## 文件说明

### dataloader.py

`dataloader.py`定义了数据加载器函数`get_data_loaders`，该函数负责应用数据增强并加载训练和验证数据集。

### model.py

`model.py`定义了`GestureClassifier`类，这是一个基于预训练ResNet-18的模型，最后一层全连接层被替换以适应新的分类任务。

### train.py

`train.py`负责训练和验证模型。脚本定义了训练过程，包括损失计算、梯度更新和验证准确率计算。训练结束后，模型权重会被保存，并绘制训练损失和验证准确率的曲线图。

## 注意事项

- 确保你的数据集路径和类别数量设置正确。
- 确保你的数据集组织结构与示例一致。
- 如果没有GPU，请确保训练参数适当，以免训练时间过长。

## 贡献

欢迎对该项目提出建议和改进。如果你有任何问题或建议，请提交Issue或Pull Request。