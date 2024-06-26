# train.py
# author r,y,x,m
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loader import get_data_loaders
from model import GestureClassifier


def train_model(m_model, t_loader, v_loader, num_epochs=10, learning_rate=0.001):
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(m_model.parameters(), lr=learning_rate)
    # 使用GPU如果GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型移动到我们的设备上
    m_model.to(device)
    # 存储训练损失
    train_losses = []
    # 存储验证正确率
    val_accuracies = []
    # 设置模型为训练模型
    for epoch in range(num_epochs):
        m_model.train()
        running_loss = 0.0
        for inputs, labels in t_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 清除梯度
            optimizer.zero_grad()
            outputs = m_model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            # 计算每个epoch的平均损失
        epoch_loss = running_loss / len(t_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validation
        # 验证
        # 设置模型为评估模型
        m_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in v_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = m_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # 计算验证准确率

        accuracy = correct / total
        val_accuracies.append(accuracy)
        print(f"Validation Accuracy: {accuracy:.4f}")

    # 保存训练后的模型
    torch.save(m_model.state_dict(), 'gesture_classifier.pth')
    print('Model saved to gesture_classifier.pth')

    # 绘制损失，准确率的折线统计图
    # Plotting training loss and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    data_dir = 'dataset'  # 文件路径
    train_loader, val_loader = get_data_loaders(data_dir)
    # 根据需求调整num_classes，我们的项目验证的是（剪刀，石头，布）所以这里是3
    model = GestureClassifier(num_classes=3)

    # Load the model if a saved model exists
    try:
        model.load_state_dict(torch.load('gesture_classifier.pth'))
        print('Model loaded from gesture_classifier.pth')
    except FileNotFoundError:
        print('No saved model found, starting from scratch')

    train_model(model, train_loader, val_loader)
