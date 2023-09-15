import torch.nn as nn
# 定义用于进行训练的CNN模型,模型包含两个卷积层和两个全连接层。
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 因为输入图片为1*28*28，说明为单通道图片，大小为28*28。
        # 1: 输入通道数（input channels）表示输入数据的通道数，这里的值为1，说明输入是单通道的图像。
        # 32: 输出通道数（output channels）表示卷积层的输出通道数，这里的值为32，说明该卷积层会产生32个不同的特征图作为输出。
        # kernel_size = 3: 卷积核大小 表示卷积核的尺寸大小。在这种情况下，卷积核的大小为3x3，即3行3列。
        # stride = 1: 步幅（stride）表示卷积操作时滑动卷积核的步幅大小。在这里，步幅为1，意味着卷积核每次在水平和垂直方向上都以1个像素的距离滑动。
        # padding = 1: 填充（padding）表示在输入图像周围添加额外的零值像素来控制输出图像的尺寸。在这里，填充为1，意味着在输入图像的周围添加1个像素宽度的零值填充。
        # 所以输出为 28+2(padding1+1=2)-(3-1)(kernek_size-1)=28,即64,32,28,28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        #表示在CNN模型中定义了一个ReLU激活函数层。
        # ReLU（Rectified Linear Unit）是一种常用的非线性激活函数，它的定义是 f(x) = max(0, x)，即将小于零的输入值变为零，而大于等于零的输入值保持不变。
        # 在深度学习中，ReLU激活函数被广泛应用于神经网络的隐藏层，作为引入非线性性质的关键组件。ReLU的主要作用是引入非线性映射，使得神经网络能够学习复杂的非线性关系。
        # 输出大小与之前一样
        self.relu = nn.ReLU()

        # 最大池化是一种用于降低特征图维度的操作，常用于卷积神经网络中。它将输入的特征图划分为不重叠的矩形区域（通常是2x2的窗口），然后在每个区域中选择最大的值作为输出。这样可以减少特征图的空间维度，并保留最显著的特征。
        # kernel_size = 2：池化窗口大小：表示池化操作使用的窗口大小。在这里，池化窗口的大小为2x2，即2行2列的窗口。
        # stride = 2：步幅：表示在池化操作时滑动池化窗口的步幅大小。在这里，步幅为2，意味着池化窗口每次在水平和垂直方向上都以2个像素的距离滑动。
        # 最大池化层通常紧跟在卷积层之后，用于减小特征图的尺寸，同时保留主要的特征。它有助于减少模型的参数数量，提高计算效率，并具有一定的平移不变性。
        # 输出大小为原图片大小长宽除以2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 32: 输入通道数（input channels）表示输入数据的通道数，为之前产生的32个不同的特征图
        # 64: 输出通道数（output channels）表示卷积层的输出通道数，这里的值为64，说明该卷积层会产生64个不同的特征图作为输出。
        # kernel_size = 3: 卷积核大小 表示卷积核的尺寸大小。在这种情况下，卷积核的大小为3x3，即3行3列。
        # stride = 1: 步幅（stride）表示卷积操作时滑动卷积核的步幅大小。在这里，步幅为1，意味着卷积核每次在水平和垂直方向上都以1个像素的距离滑动。
        # padding = 1: 填充（padding）表示在输入图像周围添加额外的零值像素来控制输出图像的尺寸。在这里，填充为1，意味着在输入图像的周围添加1个像素宽度的零值填充。
        # 所以输出为 14+2(padding1+1=2)-(3-1)(kernek_size-1)=28,即64,64,14,14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 全连接层是深度学习模型中常用的一种层类型，也称为线性层或密集连接层。它的作用是将前一层的所有输入与当前层的每个神经元进行连接，通过权重和偏置进行线性变换，然后将结果传递给激活函数进行非线性映射。
        # 参数解释如下：
        # 7 * 7 * 64：输入特征的维度 表示前一层的输出特征的维度。在这里，这个维度的值是7 * 7 * 64，说明输入特征是一个7x7的图像，具有64个通道。
        # 10：输出特征的维度 表示当前层输出特征的维度。在这里，这个维度的值是10，说明全连接层将产生一个包含10个元素的输出。
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        # print(x.shape) torch.Size([64, 1, 28, 28])

        out = self.conv1(x)
        # print(out.shape) torch.Size([64, 32, 28, 28])

        out = self.relu(out)
        # print(out.shape) torch.Size([64, 32, 28, 28])

        out = self.maxpool(out)
        # print(out.shape) torch.Size([64, 32, 14, 14])

        out = self.conv2(out)
        # print(out.shape) torch.Size([64, 64, 14, 14])

        out = self.relu(out)
        # print(out.shape) torch.Size([64, 64, 14, 14])

        out = self.maxpool(out)
        # print(out.shape) torch.Size([64, 64, 7, 7])

        # out.view(out.size(0), -1)的含义是将张量 out 进行形状变换。其中，第一个维度的大小保持不变（即 out.size(0)），而剩余维度的大小会根据张量的元素数量自动计算得出。即变成64*7*7=3136
        out = out.view(out.size(0), -1)
        # print(out.shape) torch.Size([64, 3136])

        out = self.fc(out)
        # print(out.shape) torch.Size([64, 10])
        return out
