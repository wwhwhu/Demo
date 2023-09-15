# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import csv
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CNN


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


def cal_mean_and_std(dataset):
    data = [np.array(image) / 255 for image, _ in dataset]
    data = np.stack(data, axis=0)
    # 计算三维数组 data 沿着前三个维度的平均值，即计算所有元素的平均值
    mean = np.mean(data, axis=(0, 1, 2))
    # 计算三维数组 data 沿着前三个维度的平均值，即计算所有元素的平均值
    std = np.std(data, axis=(0, 1, 2))
    print("mean:", mean, "std:", std)
    return mean, std

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyTorch')
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    mean, std = cal_mean_and_std(train_dataset)
    # 定义数据预处理转换，定义数据预处理转换，将图像转换为张量并进行标准化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean, std)  # 标准化图像数据 output = (input - mean) / std
    ])

    # 加载MNIST训练集和测试集，应用预处理转换
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建数据加载器，用于批量加载数据
    batch_size = 64
    print("batch_size: ", batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("训练集数据大小", len(train_loader))
    print("验证集数据大小", len(test_loader))
    # 获取一个小批量数据并打印其形状
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print("图片大小", images.shape)  # 打印图像数据的形状，如 (64, 1, 28, 28)
    print("标签数据大小", labels.shape)  # 打印标签数据的形状，如 (64)

    file_name = 'res/acc.csv'  # csv 文件名
    file_path = os.path.abspath(os.path.join(os.getcwd(), file_name))  # csv 文件路径
    if os.path.exists(file_path):  # 判断文件是否存在
        os.remove(file_path)  # 如果文件存在则删除
    with open(file_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch','train_acc','test_acc'])
    # 创建CNN模型实例
    model = CNN()

    # 定义交叉熵损失函数和Adam优化器
    criterion = nn.CrossEntropyLoss()
    print("损失函数", criterion)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("优化器", optimizer)

    # 训练模型
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备", device)
    model.to(device)

    best_accuracy = 0
    best_epoch = 0

    # epoch大循环
    for epoch in range(num_epochs):
        model.train()
        # batch小循环: images[64, 1, 28, 28], labels[64]
        start_time = time.time()
        for images, labels in tqdm(train_loader, desc='训练ing...'):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_time = time.time() - start_time

        # 在每个epoch结束后计算模型在训练集上的准确率
        model.eval()
        total0 = 0
        correct0 = 0
        start_time = time.time()
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc='训练集准确性测试ing...'):
                images = images.to(device)
                labels = labels.to(device)
                # outputs_size: 64,10
                outputs = model(images)
                # predicted_size: 64, 每个image输出的10个元素的tensor中最大的元素的下标，只关注下标！！
                # outputs.data 是模型的输出张量，它的形状为 (batch_size, num_classes)，其中 batch_size=64 是每个小批量数据的大小，num_classes 是分类问题的类别数。因此，torch.max(outputs.data, 1) 将返回 (max_values, max_indices) 二元组，其中 max_values 是每一行中的最大值，max_indices 是每一行中最大值所在的列的下标
                confidence_values, predicted = torch.max(outputs.data, 1)  # confidence_values是最大值的大小，即置信度值
                # 计算总数目
                total0 += labels.size(0)
                # 计算正确数目
                correct0 += (predicted == labels).sum().item()
        train_accuracy0 = 100 * correct0 / total0
        test_time0 = time.time() - start_time

        # 在每个epoch结束后计算模型在测试集上的准确率
        model.eval()
        total = 0
        correct = 0
        start_time = time.time()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='验证集准确性测试ing...'):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        test_time1 = time.time() - start_time
        tqdm.write(f"Time [{time.time()}], Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy0:.2f}%, "
                   f"Test Accuracy: {test_accuracy:.2f}%, Train Time: {train_time:.2f}s, "
                   f"Test_train_dateset Time: {test_time0:.2f}s, "
                   f"Test_test_dateset Time: {test_time1:.2f}s")
        torch.save(model.state_dict(), f'res/process/model_epoch_{epoch+1}.pkl')
        # 保存loss
        with open(file_path, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, train_accuracy0, test_accuracy])

        # 在每个epoch结束时，检查模型在验证集上的准确性是否最好
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if os.path.exists(f'res/best_model_epoch_{best_epoch}.pkl'):
                # 如果存在，则删除该文件
                os.remove(f'res/best_model_epoch_{best_epoch}.pkl')
            best_epoch = epoch + 1
            print(f"save new best_model_epoch: {best_epoch}")
            # 保存当前最好的模型
            torch.save(model.state_dict(), f'res/best_model_epoch_{best_epoch}.pkl')
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
