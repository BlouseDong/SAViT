import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math


class Onefortwo(nn.Module):
    def __init__(self, H=24, W=24, hh=4, ww=4, box_num=3, batch=16, n=16):
        super(Onefortwo, self).__init__()
        self.H = H
        self.W = W
        self.hh = hh
        self.ww = ww
        self.box_num = box_num
        self.batch = batch
        self.n = n

    def forward(self, x):
        B, N, C = x.shape  # B: 批大小, N: 特征数, C: 通道数

        # Divide x into x_restored and y
        x_restored = x[:, :576, :]
        y = x[:, 576:, :]

        # Reshape x_restored to (B, C, H, W)
        x_restored = x_restored.permute(0, 2, 1).reshape(B, C, self.H, self.W)

        # Rearrange y to (box_num, batch, n, d)
        y_restored = rearrange(y, 'batch (box_num n) d -> box_num batch n d', box_num=self.box_num, n=self.n)

        return x_restored, y_restored







class tongdao(nn.Module):  #处理通道部分   函数名就是拼音名称
    # 通道模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出大小为1x1
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于降维
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，就地操作以节省内存

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 提取批次大小和通道数
        y = self.avg_pool(x)  # 应用自适应平均池化
        y = self.fc(y)  # 应用1x1卷积
        y = self.relu(y)  # 应用ReLU激活
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # 调整y的大小以匹配x的空间维度
        return x * y.expand_as(x)  # 将计算得到的通道权重应用到输入x上，实现特征重校准

class kongjian(nn.Module):
    # 空间模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于产生空间激励
        self.norm = nn.Sigmoid()  # Sigmoid函数用于归一化

    # 前向传播函数
    def forward(self, x):
        y = self.Conv1x1(x)  # 应用1x1卷积
        y = self.norm(y)  # 应用Sigmoid函数
        return x * y  # 将空间权重应用到输入x上，实现空间激励

class hebing(nn.Module):    #函数名为合并, 意思是把空间和通道分别提取的特征合并起来
    # 合并模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)  # 创建通道子模块
        self.kongjian = kongjian(in_channel)  # 创建空间子模块

    # 前向传播函数
    def forward(self, U):
        U_kongjian = self.kongjian(U)  # 通过空间模块处理输入U
        U_tongdao = self.tongdao(U)  # 通过通道模块处理输入U
        return torch.max(U_tongdao, U_kongjian)  # 取两者的逐元素最大值，结合通道和空间激励



class MDFA(nn.Module):                       ##多尺度空洞卷积融合空间及通道特征。
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):# 初始化多尺度空洞卷积结构模块，dim_in和dim_out分别是输入和输出的通道数，rate是空洞率，bn_mom是批归一化的动量
        super(MDFA, self).__init__()
        self.branch1 = nn.Sequential(# 第一分支：使用1x1卷积，保持通道维度不变，不使用空洞
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential( # 第二分支：使用3x3卷积，空洞率为6，可以增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential( # 第三分支：使用3x3卷积，空洞率为12，进一步增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(# 第四分支：使用3x3卷积，空洞率为18，最大化感受野的扩展
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True) # 第五分支：全局特征提取，使用全局平均池化后的1x1卷积处理
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential( # 合并所有分支的输出，并通过1x1卷积降维
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing=hebing(in_channel=dim_out*5)# 整合通道和空间特征的合并模块
        self.onefortwo = Onefortwo()
    def forward(self, x):
        B, N, C = x.shape
        hh = 4
        ww = 4
        x_restored, y_restored = self.onefortwo(x)

        [b, c, row, col] = x_restored.size()
        # 应用各分支
        conv1x1 = self.branch1(x_restored)
        conv3x3_1 = self.branch2(x_restored)
        conv3x3_2 = self.branch3(x_restored)
        conv3x3_3 = self.branch4(x_restored)
        # 全局特征提取
        global_feature = torch.mean(x_restored, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # 合并所有特征
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 应用合并模块进行通道和空间特征增强
        larry=self.Hebing(feature_cat)
        larry_feature_cat=larry*feature_cat
        # 最终输出经过降维处理
        x_out = self.conv_cat(larry_feature_cat)
        x_out = x_out.reshape(B, C, 576).permute(0, 2, 1)



        y_list = []
        for i in range(y_restored.size(0)):  # 遍历第0维，即示例数
            example = y_restored[i]  # 获取当前示例，形状为 (16, 16, 768)
            _,temp,C = example.shape

            example = example.permute(0, 2, 1).reshape(16, C, hh, ww)
            [b, c, row, col] = example.size()
            # 应用各分支
            conv1x1 = self.branch1(example)
            conv3x3_1 = self.branch2(example)
            conv3x3_2 = self.branch3(example)
            conv3x3_3 = self.branch4(example)
            # 全局特征提取
            global_feature = torch.mean(example, 2, True)
            global_feature = torch.mean(global_feature, 3, True)
            global_feature = self.branch5_conv(global_feature)
            global_feature = self.branch5_bn(global_feature)
            global_feature = self.branch5_relu(global_feature)
            global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
            # 合并所有特征
            feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
            # 应用合并模块进行通道和空间特征增强
            larry = self.Hebing(feature_cat)
            larry_feature_cat = larry * feature_cat
            # 最终输出经过降维处理
            result = self.conv_cat(larry_feature_cat)
            example_out = result.reshape(16, C, 16).permute(0, 2, 1)
            y_list.append(example_out)

        y = torch.stack(y_list, dim=0)
        box_num, _, n, d = y.shape
        y_out = rearrange(y, 'box_num batch n d->batch (box_num n) d')
        x_y = torch.cat((x_out, y_out), dim=1)


        return x_y


if __name__ == '__main__':
    input = torch.load(r"D:\zhiwei\CACViT\x_y.pth")

    model = MDFA(dim_in=768,dim_out=768).cuda()  # 实例化模块
    output = model(input)  # 将输入通过模块处理
    print(output.shape)  # 输出处理后的数据形状


