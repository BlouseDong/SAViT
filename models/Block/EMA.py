import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        """
        初始化EMA模块。

        参数：
        - channels: 输入特征图的通道数。
        - c2: 可选的参数，当前实现中未使用。
        - factor: 分组的因子，用于将通道数划分为多个组。
        """
        super(EMA, self).__init__()

        self.groups = factor  # 分组因子，通道数将被划分为groups个组
        assert channels // self.groups > 0  # 确保分组后的通道数大于0

        # 定义一个Softmax层，用于在最后的加权计算中使用
        self.softmax = nn.Softmax(-1)

        # 定义自适应平均池化层，将特征图池化到 (1, 1) 大小
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        # 定义自适应平均池化层，按宽度进行池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))

        # 定义自适应平均池化层，按高度进行池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 定义GroupNorm层，按分组归一化通道
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        # 定义1x1卷积层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

        # 定义3x3卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入特征图，形状为 (batch_size, channels, height, width)

        返回：
        - 处理后的特征图
        """
        b, c, h, w = x.size()  # 获取输入特征图的维度

        # 将特征图按照分组因子重新排列，形状变为 (batch_size * groups, channels // groups, height, width)
        group_x = x.reshape(b * self.groups, -1, h, w)

        # 对重新排列后的特征图进行高度方向的池化，得到 (batch_size * groups, channels // groups, 1, width)
        x_h = self.pool_h(group_x)

        # 对重新排列后的特征图进行宽度方向的池化，并进行转置，得到 (batch_size * groups, channels // groups, height, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 将高度和宽度方向的池化结果在通道维度上拼接，并通过1x1卷积层
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))

        # 将卷积结果拆分为高度和宽度方向的特征图
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 对特征图进行归一化、加权（通过sigmoid激活函数）并重新排列
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # 使用3x3卷积对重新排列后的特征图进行卷积操作
        x2 = self.conv3x3(group_x)

        # 对归一化后的特征图进行平均池化，并计算权重
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups,
                         -1)  # 展开为 (batch_size * groups, channels // groups, height * width)

        # 对卷积后的特征图进行平均池化，并计算权重
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups,
                         -1)  # 展开为 (batch_size * groups, channels // groups, height * width)

        # 计算加权系数，并将其应用到输入特征图上
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # 将加权后的特征图重新排列为原始形状并返回
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return out


# 测试代码
if __name__ == '__main__':
    # 创建一个随机张量作为测试输入，假设batch_size=1, channels=8, height=64, width=64
    x = torch.randn(1, 8, 64, 64).to('cuda:0')  # 使用CUDA设备

    # 打印输入张量的形状
    print("Input shape:", x.shape)

    # 实例化EMA模型，假设输入通道数为8，分组因子为4
    ema_model = EMA(channels=8, factor=4).to('cuda:0')

    # 执行前向传播
    y = ema_model(x)

    # 打印输出张量的形状
    print("Output shape:", y.shape)
