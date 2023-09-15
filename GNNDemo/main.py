# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import os

import torch

from DataSet.dataset import get_dataset
from GNN_Model.model import GCNNet
from Train.train import Trainer


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    # 清理显存
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
    GNN_model = ['GCN', 'GAT', 'GIN', 'SAGE_MEAN', 'SAGE_LSTM', 'SAGE_POOL']
    DS = ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics', 'Computers', 'Photo']
    msg = input('请输入GNN模型类型：1.GCN 2.GAT 3.GIN 4.SAGE_MEAN 5.SAGE_LSTM 6.SAGE_POOL\n')
    msg2 = input('请输入数据集名称：1.Cora 2.CiteSeer 3.PubMed 4.CS 5.Physics 6.Amazon Computers 7.Amazon Photo\n')
    dataset = get_dataset(DS[int(msg2) - 1])
    num_heads = 1
    if msg == '2':
        num_heads = 8
    train = Trainer(epoch=400, lr=0.0003, node_feature=dataset.num_node_features, hidden_channels=32,
                    num_class=dataset.num_classes, model_name=GNN_model[int(msg) - 1], batch_size=16, dataset=dataset,
                    dataset_name=DS[int(msg2) - 1], num_head=num_heads)
    train.train()
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
