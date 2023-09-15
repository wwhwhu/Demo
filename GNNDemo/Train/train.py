import csv
import datetime
import os
import time

import torch
from torch import nn
from tqdm import tqdm

from GNN_Model.model import GCNNet, GATNet, GINNet, SAGENet
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, epoch, lr, node_feature, hidden_channels, num_class, model_name, batch_size, dataset,
                 dataset_name, num_head):
        # epoch: 训练轮次
        # lr: 学习率
        # node_feature: 单个节点维度（单个节点特征数）
        # num_class: 节点分类数目
        # model: 模型类别
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        print(f"训练设备: {self.device}")
        self.epoch = epoch
        self.lr = lr
        self.model_name = model_name
        self.node_feature = node_feature
        self.hidden_channels = hidden_channels
        self.num_class = num_class
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.num_head = num_head

    def train(self):
        if self.model_name == 'GCN':
            # 模型选择
            Model = GCNNet(self.node_feature, self.hidden_channels, self.num_class).to(self.device)
            optimizer = torch.optim.Adam(Model.parameters(), lr=self.lr)
            loss_function = nn.CrossEntropyLoss()
        elif self.model_name == 'GAT':
            # 模型选择
            Model = GATNet(self.node_feature, self.hidden_channels, self.num_class, self.num_head).to(self.device)
            optimizer = torch.optim.Adam(Model.parameters(), lr=self.lr)
            loss_function = nn.CrossEntropyLoss()
        elif self.model_name == 'GIN':
            # 模型选择
            Model = GINNet(self.node_feature, self.hidden_channels, self.num_class).to(self.device)
            optimizer = torch.optim.Adam(Model.parameters(), lr=self.lr)
            loss_function = nn.CrossEntropyLoss()
        elif self.model_name == 'SAGE_MEAN':
            # 模型选择
            Model = SAGENet(self.node_feature, self.hidden_channels, self.num_class, aggr_type='mean').to(self.device)
            optimizer = torch.optim.Adam(Model.parameters(), lr=self.lr)
            loss_function = nn.CrossEntropyLoss()
        elif self.model_name == 'SAGE_LSTM':
            # 模型选择
            Model = SAGENet(self.node_feature, self.hidden_channels, self.num_class, aggr_type='lstm').to(self.device)
            optimizer = torch.optim.Adam(Model.parameters(), lr=self.lr)
            loss_function = nn.CrossEntropyLoss()
        elif self.model_name == 'SAGE_POOL':
            # 模型选择
            Model = SAGENet(self.node_feature, self.hidden_channels, self.num_class, aggr_type='max').to(self.device)
            optimizer = torch.optim.Adam(Model.parameters(), lr=self.lr)
            loss_function = nn.CrossEntropyLoss()
        else:
            # 模型选择
            Model = GCNNet(self.node_feature, self.hidden_channels, self.num_class).to(self.device)
            optimizer = torch.optim.Adam(Model.parameters(), lr=self.lr)
            loss_function = nn.CrossEntropyLoss()

        print("Model Structure:\n", Model)
        best_accuracy = 0
        best_epoch = 0

        train_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

        file_name = f'res/{self.model_name}/{self.dataset_name}/acc.csv'  # csv 文件名
        file_path = os.path.abspath(os.path.join(os.getcwd(), file_name))  # csv 文件路径
        if os.path.exists(file_path):  # 判断文件是否存在
            os.remove(file_path)  # 如果文件存在则删除
        with open(file_path, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'train_loss', 'train-acc', 'test-acc', 'train-f1', 'test-f1'])

        for epoch in range(1, self.epoch + 1):
            # 训练模式
            Model.train()
            Total_loss = 0
            start_time = time.time()
            for data in tqdm(train_loader, desc="GCN训练ing..."):
                data = data.to(self.device)
                optimizer.zero_grad()
                out = Model(data)
                loss = loss_function(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                Total_loss += loss.item()
            # 这里的loss是单个batch的损失值
            loss = Total_loss / len(train_loader)
            train_time = time.time() - start_time

            # 在每个epoch结束后计算模型在训练集上的准确率
            Model.eval()
            start_time = time.time()
            # 生成的预测结果
            logits = []
            # 真实label
            ys = []
            for data in tqdm(train_loader, desc='训练集准确性测试ing...'):
                data = data.to(self.device)
                out = Model(data)
                logits.append(out[data.train_mask])
                ys.append(data.y[data.train_mask])
            logits = torch.cat(logits, dim=0)
            ys = torch.cat(ys, dim=0)
            pred = logits.argmax(dim=1)
            acc0 = pred.eq(ys).sum().item() / ys.size(0)
            f1_0 = f1_score(y_true=ys.cpu(), y_pred=pred.cpu(), average='weighted')
            test_time0 = time.time() - start_time

            # 在每个epoch结束后计算模型在测试集上的准确率
            Model.eval()
            start_time = time.time()
            # 生成的预测结果
            logits1 = []
            # 真实label
            ys1 = []
            for data in tqdm(train_loader, desc='训练集准确性测试ing...'):
                data = data.to(self.device)
                out1 = Model(data)
                logits1.append(out1[data.test_mask])
                ys1.append(data.y[data.test_mask])
            logits1 = torch.cat(logits1, dim=0)
            ys1 = torch.cat(ys1, dim=0)
            pred1 = logits1.argmax(dim=1)
            acc1 = pred1.eq(ys1).sum().item() / ys1.size(0)
            f1_1 = f1_score(y_true=ys1.cpu(), y_pred=pred1.cpu(), average='weighted')
            test_time1 = time.time() - start_time

            tqdm.write(
                f"Time [{datetime.datetime.now()}], Epoch [{epoch}/{self.epoch}], Train Accuracy and F1_Score: {acc0:.3f} {f1_0:.3f}, "
                f"Test Accuracy and F1_Score: {acc1:.3f} {f1_1:.3f}, Train Time: {train_time:.2f}s, "
                f"Test_train_dateset Time: {test_time0:.2f}s, "
                f"Test_test_dateset Time: {test_time1:.2f}s")
            torch.save(Model.state_dict(), f'res/{self.model_name}/{self.dataset_name}/model_epoch_{epoch}.pkl')

            # 保存loss
            with open(file_path, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, loss, acc0, acc1, f1_0, f1_1])

            # 在每个epoch结束时，检查模型在验证集上的准确性是否最好
            if acc1 > best_accuracy:
                best_accuracy = acc1
                if os.path.exists(f'res/{self.model_name}/{self.dataset_name}/best_model_epoch_{best_epoch}.pkl'):
                    # 如果存在，则删除该文件
                    os.remove(f'res/{self.model_name}/{self.dataset_name}/best_model_epoch_{best_epoch}.pkl')
                best_epoch = epoch
                print(f"save new best_model_epoch: {best_epoch}")
                # 保存当前最好的模型
                torch.save(Model.state_dict(),
                           f'res/{self.model_name}/{self.dataset_name}/best_model_epoch_{best_epoch}.pkl')
