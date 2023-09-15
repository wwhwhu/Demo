# GCN论文: https://arxiv.org/abs/1609.02907
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv, GINConv, SAGEConv


# GCN模型定义
class GCNNet(nn.Module):
    # GCNConv 的执行方式可以概括为以下几个步骤：
    #
    # 邻居聚合：对于每个节点，根据 edge_index 中的边索引信息，将其自身特征与其邻居节点的特征进行加权聚合。
    # 加权聚合：使用邻接矩阵中的权重，将邻居节点的特征进行加权求和，以获得每个节点的聚合信息。
    # 更新节点特征：将聚合后的信息与当前节点的特征进行融合，得到节点的新表示。
    # 非线性变换：通常，在更新节点特征后，会应用非线性激活函数（如 ReLU）对新的节点表示进行非线性变换，以增强模型的表达能力。
    # GCNConv 在执行过程中会使用图卷积的数学定义，利用邻接矩阵来实现节点特征的更新和聚合。
    # 通过堆叠多个 GCNConv 层，我们可以构建更深的 GCN 模型，以学习更复杂的节点表示。
    # 这些表示可以用于图数据的各种任务，如节点分类、图分类、链接预测等。

    # 1. 邻居节点特征聚合：
    # 对于图中的每个节点v，GCN首先通过聚合其邻居节点的特征来获取节点v的邻居特征表示。假设节点v的邻居节点集合为N(v)，则邻居节点特征聚合过程如下：
    # h_v = (1 + ε) * ∑(h_u / sqrt(1 + | N(u) |))
    # 其中，h_u表示邻居节点u的特征表示， | N(u) | 表示节点u的邻居节点数，ε是一个可学习的参数。
    # 在这个聚合过程中，节点v的邻居特征h_u被除以sqrt(1 + | N(u) |)，以缩放邻居特征的权重，避免特征在不同节点之间的度数差异导致的偏差。
    #
    # 2.自身特征更新：
    # 在邻居节点特征聚合后，节点v将其自身的特征表示h_v与聚合后的邻居特征进行合并或叠加，以更新节点v的最终特征表示。
    # 具体的更新方式可以是将自身特征h_v与聚合后的邻居特征相加，也可以是将其进行拼接。
    #
    # 3.馈神经网络变换：
    # 在自身特征更新后，节点v的最终特征表示可以通过一个前馈神经网络（例如单层的全连接层）来进行变换和映射，以生成节点的最终表征。
    # 这个前馈神经网络的权重在训练过程中被学习，用于调整节点特征的表征能力。
    #
    # 4.节点分类：
    # GCN模型的输出通常用于节点分类任务。通过训练过程，模型学习到节点特征的表示，并将节点的表征输入到一个softmax层中，生成节点的分类概率分布。
    # 最终，将这些分类概率与真实标签进行比较，并使用交叉熵损失函数来优化模型参数，以提高节点分类性能。
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_prob=0.1):
        super(GCNNet, self).__init__()
        # 第一层GCN，输入特征维度为in_channels，输出特征维度为hidden_channels
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        # 在两个 GCN 层之间添加 Dropout 层
        self.dropout = nn.Dropout(p=dropout_prob)
        # 第二层GCN，输入特征维度为hidden_channels，输出特征维度为out_channels
        self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        # x：x 是一个二维张量，形状为 [num_nodes, num_features]。
        # 其中，num_nodes 是图中节点的数量，num_features 是每个节点的特征维度。
        # 例如，如果一个图有 100 个节点，每个节点的特征维度为 32，则 x 的形状将为 [100, 32]。

        # edge_index：edge_index 是一个二维张量，形状为 [2, num_edges]。它存储了图中所有边的索引信息。
        # 具体来说，每一列包含两个元素，分别表示一条边的源节点和目标节点的索引。
        # 例如，如果一个图有 200 条边，edge_index 的形状将为 [2, 200]。
        x, edge_index = data.x, data.edge_index

        # 第一层GCN, 使用ReLU激活函数进行非线性变换
        x = F.relu(self.conv1(x, edge_index))

        # 在两个 GCN 层之间应用 Dropout
        x = self.dropout(x)

        # 第二层GCN, 不使用激活函数，直接得到节点的新表示
        x = self.conv2(x, edge_index)

        # 最后使用log_softmax得到节点的分类概率
        return F.log_softmax(x, dim=1)


# GAT模型定义
class GATNet(nn.Module):
    # GATConv层的计算方法：
    # 假设我们有一个输入图数据，其中包含节点特征矩阵X和边索引矩阵edge_index。
    # 节点特征矩阵X的形状为（num_nodes, input_dim），表示num_nodes个节点的输入特征，input_dim为输入特征的维度。
    # 边索引矩阵edge_index的形状为（2, num_edges），其中第一行表示源节点的索引，第二行表示目标节点的索引，num_edges为图中边的数量。
    # GATConv层的计算过程如下：
    # 对节点特征进行线性变换和自注意力计算：
    # 1.首先，我们通过对节点特征矩阵X进行线性变换得到节点表示矩阵Z：
    # Z = X * W
    # 其中，W是可学习的权重矩阵，形状为（input_dim, hidden_channels * num_heads）。
    # hidden_channels是GATConv层的隐藏特征维度，num_heads是头注意力的数量。

    # 2.接下来，我们计算节点之间的注意力分数，用于表示节点之间的关系和重要性。
    # 注意力分数是通过计算节点表示矩阵Z的每一对节点之间的相似度得到的。这里，我们采用一种常用的注意力计算方式，使用内积（点积）操作来计算相似度：
    #
    # e_ij = Z_i * Z_j
    # 其中，Z_i表示节点i的表示，Z_j表示节点j的表示，e_ij表示节点i和节点j之间的注意力分数。
    #
    # 3.计算注意力权重：
    # 为了得到节点i对节点j的注意力权重a_ij，我们对e_ij进行softmax操作，将注意力分数归一化为概率值：
    # a_ij = softmax(e_ij)
    #
    # 4.特征聚合：
    # 使用注意力权重a_ij对节点i的邻居节点的特征进行加权聚合，得到节点i的聚合特征：
    # h_i = sum(a_ij * Z_j)
    # 其中，h_i是节点i的聚合特征，Z_j是节点j的表示，注意力权重a_ij表示节点i对节点j的关注程度。
    #
    # 多头注意力聚合：
    # GATConv层引入了多头注意力机制，通过并行计算多个注意力头，可以捕捉不同方向和层次的节点关系。
    # 多头注意力的计算方式和上述步骤相同，但是使用不同的权重矩阵W，并将所有头的输出在特征维度上进行拼接：
    # h_i' = [h_i^1, h_i^2, ..., h_i^k]
    # 其中，h_i^k表示第k个头注意力的输出，k取值从1到num_heads。h_i'是节点i的最终聚合特征，其形状为（num_heads * hidden_channels）。
    #
    # GATConv层的参数定义：
    #
    # 在定义GATConv层时，我们需要指定一些参数，这些参数会影响层的计算和效果。
    #
    # in_channels：输入特征的维度，即节点特征的维度。
    # out_channels：输出特征的维度，即节点聚合后的特征维度。
    # heads：注意力头的数量，用于并行计算多头注意力。
    # concat：是否将多头注意力的输出在特征维度上进行拼接，如果为True，则输出的维度是out_channels * heads；如果为False，则输出的维度是out_channels。
    # negative_slope：LeakyReLU激活函数的负斜率，用于增加模型的非线性性，默认为0.2。
    # dropout：Dropout的概率，用于在特征聚合过程中进行随机失活，默认为0，表示不使用Dropout。
    # add_self_loops：是否在边索引矩阵中添加自环边，默认为False。自环边表示节点与自身的连接。
    def __init__(self, in_channels, hidden_channels, num_classes, num_heads, dropout_prob=0.1):
        # in_channels：输入特征的维度。在Cora数据集中，每个节点的特征维度是词频统计的向量的维度（即1433维）。
        # hidden_channels：GATConv层的隐藏特征维度。
        # num_classes：输出类别的数量。
        # num_heads：头注意力的数量。设置为8表示有8个头注意力机制。
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        # 在两个 GAT 层之间添加 Dropout 层
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv2 = GATConv(hidden_channels * num_heads, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# GIN模型定义
class GINNet(nn.Module):
    # GIN模型的核心思想是通过对节点和其邻居的特征进行可交换的聚合，来捕捉图中节点之间的全局信息。
    # 可交换性是指无论节点的排列方式如何，聚合的结果都应该是相同的。GIN模型通过将图中每个节点的特征与其邻居的特征进行可交换的聚合，并迭代多次来获取节点的最终表示。
    # GIN模型的层次结构：
    # GIN模型由多个GINConv层组成，每个GINConv层都是一个节点特征的聚合模块。GINConv层的计算方式如下：
    #
    # 1.对邻居节点特征进行聚合：
    # 对于节点v的邻居节点u，我们使用一个全连接层来对其特征进行线性变换，并对变换后的特征进行求和聚合：
    # z_u = MLP(X_u)
    ## 其中，X_u表示节点u的输入特征，MLP是一个全连接层。
    #
    # 2.聚合节点自身的特征：
    # 对于节点v自身的特征，我们也进行线性变换：
    ## z_v = MLP(X_v)
    #
    # 3.合并邻居节点特征和节点自身特征：
    # 将聚合后的邻居节点特征和节点自身特征进行求和，得到节点v的最终表示：
    # x_v' = z_v + ∑z_u
    # 其中，z_v表示节点v自身的特征，z_u表示节点v的邻居节点u的特征。
    #
    # 4.更新节点特征：
    # 将节点v的最终表示x_v'作为节点v的新特征，用于下一层的计算。
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GINNet, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        # self.conv2 = GINConv(nn.Sequential(
        #     nn.Linear(hidden_channels, hidden_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_channels, hidden_channels)
        # ))
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = self.conv2(x, edge_index)
        x = self.final_mlp(x)
        return F.log_softmax(x, dim=1)


# SAGE模型定义
class SAGENet(nn.Module):
    # SAGE（GraphSAGE，Graph Sample and Aggregated）是一种用于图数据的卷积神经网络方法，用于节点表征学习和节点分类任务。
    # SAGE的计算原理涉及图数据的采样和特征聚合，以下是SAGE方法的主要计算原理：
    #
    # 1.图数据采样：
    # 在SAGE方法中，为了处理大规模的图数据，通常会对图进行采样操作，从而减少计算和内存开销。
    # 采样时会选取每个节点的一部分邻居节点，或者直接采样整个图的一个子图。采样操作可以使得模型对大规模图数据进行有效处理，同时保持对图结构的局部性特征的捕捉。
    #
    # 2.邻居节点特征聚合：
    # 对于采样得到的节点及其邻居子图，SAGE首先对邻居节点特征进行聚合操作，得到每个节点的邻居特征表示。SAGE中常用的邻居特征聚合方法包括：
    # Mean Aggregation（均值聚合）、LSTM Aggregation（LSTM聚合）和Pooling Aggregation（池化聚合）等。
    # 这些方法会将邻居节点的特征进行平均、使用LSTM进行序列建模或者进行最大值池化，得到每个节点的邻居特征。
    #
    # 3.自身特征更新：
    # 在邻居节点特征聚合后，每个节点将其自身的特征与邻居特征进行合并或叠加，以更新节点的最终特征表示。
    # 具体的更新方式可以是将自身特征与邻居特征进行拼接或者求和。
    # 通过将邻居特征与自身特征进行融合，SAGE可以将节点的全局信息和局部信息相结合，从而生成节点的新特征表示。
    #
    # 4.前馈神经网络变换：
    # 在自身特征更新后，节点的最终特征表示可以通过一个前馈神经网络（例如单层的全连接层）来进行变换和映射，以生成节点的最终表征。
    # 这个前馈神经网络的权重在训练过程中被学习，用于调整节点特征的表征能力。

    # SAGEConv的参数及其含义如下：
    #
    # in_channels：输入特征的维度，即节点特征的维度。
    #
    # out_channels：输出特征的维度，即节点表征的维度。SAGEConv层会将输入特征映射为输出特征维度。
    #
    # normalize：是否对聚合后的邻居特征进行归一化。默认为True，表示对聚合后的邻居特征进行L2归一化。
    #
    # bias：是否使用偏置项。默认为True，表示在特征聚合时使用偏置项。
    #
    # aggr：邻居特征聚合方法，可以是'mean'、'lstm'或'max_pool'之一。默认为'mean'，表示使用均值聚合。在构造SAGEConv层时，我们可以根据任务需求选择不同的聚合方法。
    def __init__(self, in_channels, hidden_channels, out_channels, aggr_type='mean'):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr_type)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr_type)
        self.aggr_type = aggr_type

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.aggr_type == 'lstm':
            edge_index, _ = data.edge_index.sort(dim=1)  # 对edge_index进行排序
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
