import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torchsummary import summary

'''
B站论文解说：MAE中掩码的部分也是要经过处理后成为嵌入向量的？不是直接分块后遮盖住就不管的。
'''

# 原论文中在多头注意力之后 使用的是dropout层，有其他大神用droppath实现，效果更好
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class 三维图像块嵌入(nn.Module):
    """
    现在认为输入的图像都是已经划分成图像块之后的结果。
    """
    def __init__(self,图像块嵌入向量的维度):
        """
        3维输入数据形状BCDHW 1*2*256*16*16，在256中随机选取部分用来训练，输出1*256的向量，然后整形为1*16*16。

        :param 图像块嵌入向量的维度:
        """
        super().__init__()
        self.图像块的大小 = 16
        self.标准化 = nn.LayerNorm(图像块嵌入向量的维度)

    def forward(self, x):
        # 输入1*2*64*16*16，展平后1*2*64*256。形状[B, C, D, H, W] -> [B, C, D , HW].交换维度后，1*64*256*2，形状[B, D, HW, C]
        x = torch.flatten(x, start_dim=3).permute(0, 2, 3, 1)
        # 再展平后输出1*64*512，形状[B, D, HWC]
        x = torch.flatten(x, 2)
        x = self.标准化(x)
        return x


class 自注意力(nn.Module):
    """
    多头自注意力模块
    """
    def __init__(self,
                 嵌入向量维度,
                 注意力头数量,
                 qkv_偏差=False,
                 自注意力丢弃率=0.,
                 全连接层丢弃率=0.):
        super(自注意力, self).__init__()
        self.注意力头数量 = 注意力头数量
        注意力头维度 = 嵌入向量维度 // 注意力头数量  # 每个注意力头的维度大小
        self.qk_缩小因子 = 注意力头维度 ** -0.5
        # 通过一个全连接层生成qkv三个向量，有助于并行化计算
        # 小数据集中qkv直接false，但是原版用参数控制true
        self.qkv = nn.Linear(嵌入向量维度, 嵌入向量维度 * 3, bias=qkv_偏差)
        self.自注意力丢弃 = nn.Dropout(自注意力丢弃率) # 自注意力信息经过softmax后的Dropout

        # 多头注意力的输出拼接后与Wo相乘得到最终的输出。Wo矩阵通过全连接层实现
        self.线性投影 = nn.Linear(嵌入向量维度, 嵌入向量维度)
        self.线性投影丢弃 = nn.Dropout(全连接层丢弃率)

    def forward(self, x):
        批量大小, 图像块数量, 嵌入向量维度 = x.shape
        # [batch_size, num_patches + 1, embed_dim] 每一批图片数。图像块的数量加1是因为算上class token，我的方法按照纵向的深度计算
        # qkv(): -> [批量大小, 图像块数量 + 1, 3 * 嵌入向量维度] 我不分类可能不加分类嵌入向量。
        # 3对应qkv三个向量。后面两个维度的数据可以理解为将嵌入向量按照注意力头的数量平均划分之后分别送入不同的注意力头中计算
        # reshape: -> [批量大小, 图像块数量 + 1, 3, 注意力头数量, 每个注意力头的嵌入向量维度]。
        # permute: -> [3, 批量大小, 注意力头数量, 图像块数量 + 1, 每个注意力头的嵌入向量维度]
        qkv = self.qkv(x).reshape(批量大小, 图像块数量, 3, self.注意力头数量, 嵌入向量维度 // self.注意力头数量).permute(2, 0, 3, 1, 4)

        # 分别取出qkv向量。向量的形状[批量大小, 注意力头数量, 图像块数量 + 1, 每个注意力头的嵌入向量维度]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # k最后两个维度调换位置，k.transpose：-> [批量大小, 注意力头数量, 每个注意力头的嵌入向量维度, 图像块数量 + 1]
        # q和k矩阵相乘: -> [批量大小, 注意力头数量, 图像块数量 + 1, 图像块数量 + 1]
        自注意力信息 = (q @ k.transpose(-2, -1)) * self.qk_缩小因子  # q和k进行多维矩阵乘法时，实际只有最后两个维度相乘
        自注意力信息 = 自注意力信息.softmax(dim=-1)  # 沿着最后一个维度进行柔性最大值处理。
        自注意力信息 = self.自注意力丢弃(自注意力信息)

        # 自注意力信息和v矩阵相乘：-> [批量大小, 注意力头数量, 图像块数量 + 1, 每个注意力头的嵌入向量维度]
        # transpose: -> [批量大小, 图像块数量 + 1, 注意力头数量, 每个注意力头的嵌入向量维度]
        # reshape: -> [批量大小, 图像块数量 + 1, 嵌入向量维度]
        自注意力信息 = (自注意力信息 @ v).transpose(1, 2).reshape(批量大小, 图像块数量, 嵌入向量维度)  # 和v矩阵相乘 加权求和
        自注意力信息 = self.线性投影(自注意力信息)
        自注意力信息 = self.线性投影丢弃(自注意力信息)
        return 自注意力信息


class 多层感知机(nn.Module):
    """
    transformer编码块中的MLP模块
    """
    def __init__(self, 输入大小, 隐藏层大小=None, 激活函数=nn.GELU, 丢弃率=0.):
        """

        :param 输入大小: (int) 第一个全连接层的输入大小,默认和嵌入向量维度大小一致。
        :param 隐藏层大小: (int)
        :param 激活函数:
        :param 丢弃率:
        """
        super(多层感知机, self).__init__()
        隐藏层大小 = 隐藏层大小 or 输入大小
        self.全连接1 = nn.Linear(输入大小, 隐藏层大小)
        self.激活函数 = 激活函数()
        self.全连接2 = nn.Linear(隐藏层大小, 输入大小)
        self.随机丢弃 = nn.Dropout(丢弃率)

    def forward(self, x):
        x = self.全连接1(x)
        x = self.激活函数(x)
        x = self.随机丢弃(x)
        x = self.全连接2(x)
        x = self.随机丢弃(x)
        return x


def 随机掩码(x, 掩码率):
    """
    随着遮盖部分图像块
    x: [批量, 块数量, 嵌入向量维度]
    """

    批量, 块数量, 嵌入向量维度 = x.shape  # 获取输入数据的形状，[批量, 块数量, 维度]
    保留块数量 = int(块数量 * (1 - 掩码率))
    noise = torch.rand(批量, 块数量, device=x.device)  # 二维随机噪声矩阵，数值在[0, 1]

    # sort noise for each sample。argsort()返回的是元素对应的索引
    # ids_shuffle是用来选择哪些元素做掩码
    ids_shuffle = torch.argsort(noise, dim=1)  # 升序排列ascend: small is keep, large is remove
    # ids_restore是用于在编码之后对图像块的顺序进行还原并输入解码器？
    ids_restore = torch.argsort(ids_shuffle, dim=1) # 对上一步得到的索引排序

    # keep the first subset
    ids_keep = ids_shuffle[:, :保留块数量]
    # 没有被掩码的图像块序列？
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 嵌入向量维度))

    # generate the binary mask: 0 is keep, 1 is remove
    #
    mask = torch.ones([批量, 块数量], device=x.device)
    mask[:, :保留块数量] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class 二维编码块(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(二维编码块, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out



class 编码块(nn.Module):
    """
    transformer编码器模块，将自注意力模块和MLP模块结合起来
    """
    def __init__(self,
                 特征维度, # 输入向量的维度
                 注意力头数量, # 注意力头的数量
                 多层感知机扩增率=4., # 第一个全连接层的节点是输入节点的四倍，对应768变为2304
                 qkv_偏差=False, # 是否使用偏差
                 自注意力丢弃率=0.,  # qkv矩阵计算输出之后经过softmax层后的dropout
                 全连接层丢弃率=0., # 对应多头注意力模块中最后一个全连接层
                 drop_path_ratio=0., # 对应的编码块中droppath
                 激活函数=nn.GELU, # 默认激活函数
                 标准化=nn.LayerNorm): # 默认归一化方式
        super(编码块, self).__init__()
        self.标准化1 = 标准化(特征维度) # 编码块中的第一个LN层
        self.多头自注意力 = 自注意力(特征维度, 注意力头数量=注意力头数量, qkv_偏差=qkv_偏差, 自注意力丢弃率=自注意力丢弃率, 全连接层丢弃率=全连接层丢弃率)
        # 如果drop_path_ratio大于0，就采用droppath否则直接用恒等映射。作者认为droppath比dropout效果好
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.标准化2 = 标准化(特征维度) # 对应MLP块中最前面的的LN层
        隐藏层维度 = int(特征维度 * 多层感知机扩增率) # MLP模块中第一个全连接层对应的隐藏层节点数量，这里相当于维度增加。
        self.多层感知机 = 多层感知机(输入大小=特征维度, 隐藏层大小=隐藏层维度, 激活函数=激活函数, 丢弃率=全连接层丢弃率)

    def forward(self, x):
        x = x + self.drop_path(self.多头自注意力(self.标准化1(x)))
        x = x + self.drop_path(self.多层感知机(self.标准化2(x))) # 不能从这里直接修改多层感知机的输出维度，因为要和原输入x相加
        return x


class 三维掩码编码网络(nn.Module):
    """
    注意三维B-scan OCTA图像深度的选择
    """
    def __init__(self, 图像深度, 嵌入向量维度=512, 图像块大小=16, 掩码率=0.5, 深度=12, 注意力头数量=8, 多层感知机扩增率=4.0,
                 qkv_偏差=True, 全连接丢弃率=0., 自注意力丢弃率=0., drop_path_ratio=0., 图像嵌入层=三维图像块嵌入, 标准化=None,):
        """

        :param 嵌入向量维度: (int) 大小为图像块的宽度乘高度再乘模态数。单模态用256，双模态用512
        :param 图像深度: 从上往下截取三维B-scan OCTA图像的大小
        :param 图像块大小:
        :param 掩码率:
        :param 深度: transformer编码器重复的次数。
        :param 注意力头数量: (int): 必须是嵌入向量维度的因数。
        :param 多层感知机扩增率: (int): 多层感知机模块中第一个全连接层的隐藏层节点数目和嵌入向量维度的比值
        :param qkv_偏差:
        :param 全连接丢弃率:
        :param 自注意力丢弃率: 多头自注意力中输出经过softmax层之后的dropout层的丢弃率。
        :param drop_path_ratio:
        :param 图像嵌入层: (nn.Module): 将图像划分成若干块之后生成嵌入向量
        :param 标准化:
        """
        super().__init__()
        self.图像块大小 = 图像块大小
        self.掩码率 = 掩码率
        标准化 = 标准化 or partial(nn.LayerNorm, eps=1e-6)
        self.图像块嵌入向量 = 图像嵌入层(嵌入向量维度)

        # 第一个维度的1对应的是批量
        self.位置嵌入向量 = nn.Parameter(torch.zeros(1, 图像深度, 嵌入向量维度))

        掩码后图像块数量 = int(图像深度 * (1 - 掩码率))
        # 构建了一个等差数列，保存后续transformer编码块中的drop率。最后一个transformer编码块需要令输出形状为1*256，方便调整为16*16
        丢弃率 = [x.item() for x in torch.linspace(0, drop_path_ratio, 深度)]  # stochastic depth decay rule

        # transformer编码块重复指定的次数
        self.编码块堆叠 = nn.Sequential(*[
            编码块(特征维度=嵌入向量维度, 注意力头数量=注意力头数量, 多层感知机扩增率=多层感知机扩增率, qkv_偏差=qkv_偏差,
                自注意力丢弃率=自注意力丢弃率, 全连接层丢弃率=全连接丢弃率, drop_path_ratio=丢弃率[i], 标准化=标准化)
                # 自注意力丢弃率 = 自注意力丢弃率, 全连接层丢弃率 = 全连接丢弃率, drop_path_ratio = 0.1, 标准化 = 标准化)
            for i in range(深度)
        ])
        self.标准化 = 标准化(嵌入向量维度) # 在所有的transformer编码块之后的LN层。输出形状[1, 256， 512]
        self.通道融合 = nn.Linear(嵌入向量维度, 256) # 输出形状[1, 256， 256]
        self.激活函数 = nn.GELU()
        self.特征降维 = nn.Linear(掩码后图像块数量, 1)

        # 权重初始化
        nn.init.trunc_normal_(self.位置嵌入向量, std=0.02)
        self.apply(self._初始化权重)

    # 使用掩码思想时初始化
    def _初始化权重(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # [B, C, D, H, W] -> [批量, 图像块数量, 嵌入向量维度] 我的图像深度相当于图像块数量
        x = self.图像块嵌入向量(x)  # 输出形状[B, 256, 512]
        # 这里报错说明实际输入的嵌入向量维度和参数传入的嵌入向量维度大小不一致。
        x = x + self.位置嵌入向量 # 加上位置嵌入信息，准备输入transformer编码器
        x, mask, ids_restore = 随机掩码(x, self.掩码率) #
        x = self.编码块堆叠(x)
        x = self.标准化(x) # 输出形状[批量, 块数量， 维度]
        x = self.通道融合(x) # 输出形状[批量, 块数量， 维度]
        x = self.激活函数(x)
        x = torch.transpose(x, 1, 2)
        x = self.特征降维(x) # 输出形状[批量, 维度， 1]
        x = torch.reshape(x, (-1, self.图像块大小, self.图像块大小)) # 输出形状[批量, 图像块大小, 图像块大小]。考虑批量维度方便输入真实数据
        # print(x.shape)
        return x


class 二维图像块嵌入6M(nn.Module):
    """
    输入原始的6M大小的二维en-face OCTA图像，分块之后转化为图像块嵌入向量。
    """
    def __init__(self, 图像块嵌入向量的维度=256):
        """

        :param 图像块嵌入向量的维度:
        """
        super().__init__()
        # 输入是灰度图，单通道
        self.裁剪缩放1 = nn.Conv2d(1, 8, kernel_size=29, stride=1, padding=14, bias=False)
        self.裁剪缩放2 = nn.Conv2d(8, 16, kernel_size=29, stride=1, padding=14, bias=False)
        self.裁剪缩放3 = nn.Conv2d(16, 32, kernel_size=29, stride=2, bias=False)  # 得到420大小

        self.裁剪缩放4 = nn.Conv2d(32, 16, kernel_size=21, stride=1, padding=10, bias=False)
        self.裁剪缩放5 = nn.Conv2d(16, 8, kernel_size=21, stride=1, padding=10, bias=False)
        self.裁剪缩放6 = nn.Conv2d(8, 1, kernel_size=21, stride=1, bias=False)  # 得到400大小

        self.图像分块 = nn.Conv2d(1, 图像块嵌入向量的维度, kernel_size=16, stride=16)
        self.标准化 = nn.LayerNorm(图像块嵌入向量的维度)

    def forward(self, x):
        # 输入真实图像
        x = self.裁剪缩放1(x)
        x = self.裁剪缩放2(x)
        x = self.裁剪缩放3(x)

        x = self.裁剪缩放4(x)
        x = self.裁剪缩放5(x)
        x = self.裁剪缩放6(x)

        x = self.图像分块(x)  # 输出[批量，嵌入向量维度，高度，宽度]
        x = torch.flatten(x, start_dim=2)  # 将高度和宽度方向的数据展平
        x = torch.transpose(x, 1, 2)  # 输出[批量，块数量，嵌入向量维度]
        x = self.标准化(x)
        return x


class 二维图像块嵌入3M(nn.Module):
    """
    用于3M图像的嵌入模块，现在基本不考虑输入原始的二维图像，分块之后转化为图像块嵌入向量。
    """
    def __init__(self, 图像形状, 图像块嵌入向量的维度=256):
        """
        :param 图像形状:
        :param 图像块嵌入向量的维度:
        """
        super().__init__()
        self.图像形状 = 图像形状
        self.图像块的大小 = 16
        # TODO 将原图随机裁剪之后再输入模型
        self.图像块数量 = (图像形状 // 16) ** 2 # 使用有标签的图，那么图像块有2916个
        # 输入是灰度图，单通道
        # 868首先缩放到618，然后卷积得到304
        self.裁剪 = nn.Conv2d(1, 1, kernel_size=11, stride=2)
        self.图像分块 = nn.Conv2d(1, 图像块嵌入向量的维度, kernel_size=16, stride=16)
        self.标准化 = nn.LayerNorm(图像块嵌入向量的维度)

    def forward(self, x):
        # 输入真实图像
        x = self.裁剪(x)
        x = self.图像分块(x) # 输出[批量，嵌入向量维度，高度，宽度]
        x = torch.flatten(x, start_dim=2) # 将高度和宽度方向的数据展平
        x = torch.transpose(x, 1, 2) # 输出[批量，块数量，嵌入向量维度]
        x = self.标准化(x)
        return x


class 已分块二维图像块嵌入(nn.Module):
    """
    现在认为输入的图像都是已经划分成图像块之后的结果。有监督训练的时候在另一个类里对有标签的图像划分。
    """
    def __init__(self, 图像块嵌入向量的维度=256):
        """
        :param 图像块嵌入向量的维度:
        """
        super().__init__()
        self.标准化 = nn.LayerNorm(图像块嵌入向量的维度)

    def forward(self, x):
        # 立体图像融合之后的结果进行分块，输入应该是[1*361*16*16]，[批量，宽度，高度，通道数]
        # 现在只考虑从立体数据融合来的。输入1*361*16*16，展平后1*361*256。
        x = torch.flatten(x, start_dim=2)
        x = self.标准化(x) # 输出形状[1*361*256]
        return x


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class 二维掩码编码网络(nn.Module):
    """
    现在考虑传进来的数据是已经划分块之后的结果，图像是单通道。目前优先考虑400
    """
    def __init__(self, 图像形状, 图像块大小=16, 嵌入向量维度=256, 深度=1, 注意力头数量=1, 解码器嵌入向量维度=256, 解码器深度=1,
                 解码器注意力头数量=1, 多层感知机扩增率=4., 标准化=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE编码器实现
        self.图像块嵌入向量 = 已分块二维图像块嵌入(嵌入向量维度)
        # 由三维图像数据融合得到的二维图像数据，图像块的数量在三维数据划分时就已确定。对于400形状是625块
        图像块数量 = (图像形状 // 图像块大小) ** 2
        # 第一个轴的1表示批量维度
        self.分类嵌入 = nn.Parameter(torch.zeros(1, 1, 嵌入向量维度))
        self.位置嵌入向量 = nn.Parameter(torch.zeros(1, 图像块数量 + 1, 嵌入向量维度), requires_grad=False)  # fixed sin-cos embedding
        # 作者没给transformer编码块传入dropout的丢弃率
        self.编码块堆叠 = nn.Sequential(*[
            编码块(特征维度=嵌入向量维度, 注意力头数量=注意力头数量, 多层感知机扩增率=多层感知机扩增率, qkv_偏差=True, 标准化=标准化)
            for i in range(深度)])
        self.编码器标准化 = 标准化(嵌入向量维度)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE解码器实现
        self.__解码嵌入 = nn.Linear(嵌入向量维度, 解码器嵌入向量维度, bias=True)
        # 替换被遮掩的图像块
        self.__mask_token = nn.Parameter(torch.zeros(1, 1, 解码器嵌入向量维度))
        # 加1是因为解码也需要cls_token
        self.__解码器位置嵌入向量 = nn.Parameter(torch.zeros(1, 图像块数量 + 1, 解码器嵌入向量维度),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.__解码块堆叠 = nn.Sequential(*[
            编码块(特征维度=解码器嵌入向量维度, 注意力头数量=解码器注意力头数量, 多层感知机扩增率=多层感知机扩增率, qkv_偏差=True, 标准化=标准化)
            for i in range(解码器深度)])

        self.__解码器标准化 = 标准化(解码器嵌入向量维度)
        # 这个就是输出时的最后一层了
        self.__解码器预测值 = nn.Linear(解码器嵌入向量维度, 图像块大小 ** 2, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # 是否使用归一化像素损失
        self.__norm_pix_loss = norm_pix_loss
        # 初始化编码器位置嵌入向量。 (and freeze) pos_embed by sin-cos embedding
        位置嵌入向量 = get_2d_sincos_pos_embed(self.位置嵌入向量.shape[-1], int(图像块数量 ** .5),
                                            cls_token=True)
        self.位置嵌入向量.data.copy_(torch.from_numpy(位置嵌入向量).float().unsqueeze(0))
        # 初始化解码器位置嵌入向量。
        解码器位置嵌入向量 = get_2d_sincos_pos_embed(self.__解码器位置嵌入向量.shape[-1],
                                                    int(图像块数量 ** .5), cls_token=True)
        self.__解码器位置嵌入向量.data.copy_(torch.from_numpy(解码器位置嵌入向量).float().unsqueeze(0))
        torch.nn.init.normal_(self.分类嵌入, std=.02)
        torch.nn.init.normal_(self.__mask_token, std=.02)
        # 初始化 nn.Linear 和 nn.LayerNorm
        self.apply(self._初始化权重)

    # 使用掩码思想时初始化
    def _初始化权重(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 将图片划分成块，计算损失函数的时候用到，我现在已经就是块了所以不需要
    def patchify(self, imgs):
        """
        块大小 = self.图像块嵌入向量.图像块的大小 # 每个patch的长宽
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % 块大小 == 0

        # 原图像的高度和宽度对块的大小做整数除法
        高度数量 = 宽度数量 = imgs.shape[2] // 块大小
        x = imgs.reshape(shape=(imgs.shape[0], 3, 高度数量, 块大小, 宽度数量, 块大小))
        # batchsize，通道数，patch每一边的数量，每个patch的宽高，patch每一边的数量，每个patch的长宽
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], 高度数量 * 宽度数量, 块大小 ** 2 * 3)) # 
        # 其实直接转换过来就行， （batchsize，patch总数base=196，path平方x通道数也就是每个patch块内的数据）
        """
        x = torch.flatten(imgs, start_dim=2)
        # print(x.shape)

        return x

    # 将图像块还原成完整的图像
    def unpatchify(self, x):
        """
        x: (批量, L, patch_size**2 *3)
        imgs: (批量, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        # （batchsize，patch总数base=196，path平方x通道数也就是每个patch块内的数据）->（图像宽高，通道数，hxp=patch[0]xp的宽高）
        return imgs

    def forward_encoder(self, x, 掩码率):
        # print("x的形状", x.shape)
        x = self.图像块嵌入向量(x) # 图像块转化为嵌入向量 [1*361*256]
        # 加上位置嵌入向量
        x = x + self.位置嵌入向量[:, 1:, :] # 从1开始是因为0对应cls_token
        # masking: length -> length * 掩码率。
        x, mask, ids_restore = 随机掩码(x, 掩码率)
        # 添加分类嵌入向量
        分类嵌入 = self.分类嵌入 + self.位置嵌入向量[:, :1, :]
        分类嵌入 = 分类嵌入.expand(x.shape[0], -1, -1)
        x = torch.cat((分类嵌入, x), dim=1)
        x = self.编码块堆叠(x)
        x = self.编码器标准化(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # 编码器的输出转化为解码器的输入嵌入向量
        x = self.__解码嵌入(x)

        # 加上被掩码的图像
        mask_tokens = self.__mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # 添加解码器位置嵌入向量
        x = x + self.__解码器位置嵌入向量
        x = self.__解码块堆叠(x)
        x = self.__解码器标准化(x)
        x = self.__解码器预测值(x)
        x = x[:, 1:, :] # 取出没有分类嵌入向量的部分形状[批量, 361, 256]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [批量, 3, H, W]
        pred: [批量, L, p*p*3]
        mask: [批量, L], 0 is keep, 1 is remove,
        """

        target = self.patchify(imgs)
        if self.__norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # pred和target维度不一致
        # print('target形状', target.shape)
        # print('pred形状', pred.shape)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [批量, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, 掩码率=0.5):
        # x是已经分块转化为嵌入向量的形式了
        latent, mask, ids_restore = self.forward_encoder(x, 掩码率)
        pred = self.forward_decoder(latent, ids_restore)  # [批量, L, p*p*3]
        loss = self.forward_loss(x, pred, mask)
        return loss
        # return loss, pred, mask


class 有监督分类阶段6MR多批量(nn.Module):

    def __init__(self,
                 类别数,编码块 = 二维编码块, groups=1, width_per_group=64, 图像块大小=16, 嵌入向量维度=256, 深度=1, 注意力头数量=1, 多层感知机扩增率=4.,
                 标准化=nn.LayerNorm):
        super().__init__()
        self.include_top = True
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # 将图像调整至合适的大小
        self.前置卷积1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.前置卷积2 = nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        blocks_num = [3, 4, 6, 3]

        self.layer1 = self._make_layer(编码块, 64, blocks_num[0])
        self.layer2 = self._make_layer(编码块, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(编码块, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(编码块, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * 编码块.expansion, 类别数)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, 编码块, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * 编码块.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * 编码块.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * 编码块.expansion))

        layers = [编码块(self.in_channel,
                        channel,
                        downsample=downsample,
                        stride=stride,
                        groups=self.groups,
                        width_per_group=self.width_per_group)]
        self.in_channel = channel * 编码块.expansion

        for _ in range(1, block_num):
            layers.append(编码块(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.前置卷积1(x)
        x = self.前置卷积2(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x




















class 有监督分类模型6M多批量(nn.Module):
    """
    有监督训练微调阶段。现在考虑传进来的数据是未分块的6M图像，单通道，大小400。
    """
    def __init__(self, 批量大小, 图像形状, 类别数 = 2, 图像块大小=16, 嵌入向量维度=256, 深度=1, 注意力头数量=1, 多层感知机扩增率=4.,
                 标准化=nn.LayerNorm):
        """
        默认编码块深度和注意力头数量全是1，实例化的时候注意修改成和自监督阶段的数目一致。

        :param 批量大小:
        :param 图像形状: 3M图像是304,6M图像是400
        :param 类别数:
        :param 图像块大小:
        :param 嵌入向量维度:
        :param 深度:
        :param 注意力头数量:
        :param 多层感知机扩增率:
        :param 标准化:
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # 编码器实现
        self.图像块嵌入向量 = 二维图像块嵌入6M(嵌入向量维度)
        # 对原图随机裁剪之后的图像形状
        图像块数量 = (图像形状 // 图像块大小) ** 2
        # 自监督按照批量为1定义的，当批量大于1时这里不能直接定义为1
        self._分类嵌入 = nn.Parameter(torch.zeros(批量大小, 1, 嵌入向量维度))
        self._位置嵌入向量 = nn.Parameter(torch.zeros(批量大小, 图像块数量 + 1, 嵌入向量维度), requires_grad=False)  # fixed sin-cos embedding

        # 作者没给transformer编码块传入dropout的丢弃率
        self.编码块堆叠 = nn.Sequential(*[
            编码块(特征维度=嵌入向量维度, 注意力头数量=注意力头数量, 多层感知机扩增率=多层感知机扩增率, qkv_偏差=True, 标准化=标准化)
            for i in range(深度)])
        self.编码器标准化 = 标准化(嵌入向量维度)
        # --------------------------------------------------------------------------
        self.截取 = nn.Identity()
        self.分类头 = nn.Linear(嵌入向量维度, 类别数)
        torch.nn.init.normal_(self._分类嵌入, std=.02)
        位置嵌入向量 = get_2d_sincos_pos_embed(self._位置嵌入向量.shape[-1], int(图像块数量 ** .5),
                                            cls_token=True)
        self._位置嵌入向量.data.copy_(torch.from_numpy(位置嵌入向量).float().unsqueeze(0))
        self.apply(self._初始化权重)

    def _初始化权重(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        x = self.图像块嵌入向量(x) # 图像块转化为嵌入向量 [1*361*256]
        # 加上位置嵌入向量
        x = x + self._位置嵌入向量[:, 1:, :] # 从1开始是因为0对应cls_token
        # 添加分类嵌入向量
        分类嵌入 = self._分类嵌入 + self._位置嵌入向量[:, :1, :]
        分类嵌入 = 分类嵌入.expand(x.shape[0], -1, -1)
        x = torch.cat((分类嵌入, x), dim=1)
        x = self.编码块堆叠(x)
        x = self.编码器标准化(x)
        x = self.截取(x[:, 0])
        x = self.分类头(x)

        return x



class 有监督分类模型6M(nn.Module):
    """
    有监督训练微调阶段。现在考虑传进来的数据是未分块的6M图像，单通道，大小400。
    """
    def __init__(self, 图像形状, 类别数 = 2, 图像块大小=16, 嵌入向量维度=256, 深度=1, 注意力头数量=1, 多层感知机扩增率=4.,
                 标准化=nn.LayerNorm):
        """
        默认编码块深度和注意力头数量全是1，实例化的时候注意修改成和自监督阶段的数目一致。

        :param 图像形状: 3M图像是304,6M图像是400
        :param 类别数:
        :param 图像块大小:
        :param 嵌入向量维度:
        :param 深度:
        :param 注意力头数量:
        :param 多层感知机扩增率:
        :param 标准化:
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # 编码器实现
        self.图像块嵌入向量 = 二维图像块嵌入6M(嵌入向量维度)
        # self.图像块嵌入向量 = 图像块嵌入(图像形状, 嵌入向量维度)
        # 对原图随机裁剪之后的图像形状
        图像块数量 = (图像形状 // 图像块大小) ** 2
        # 第一个轴的1表示批量维度
        self.分类嵌入 = nn.Parameter(torch.zeros(1, 1, 嵌入向量维度))
        self.位置嵌入向量 = nn.Parameter(torch.zeros(1, 图像块数量 + 1, 嵌入向量维度), requires_grad=False)  # fixed sin-cos embedding

        # 作者没给transformer编码块传入dropout的丢弃率
        self.编码块堆叠 = nn.Sequential(*[
            编码块(特征维度=嵌入向量维度, 注意力头数量=注意力头数量, 多层感知机扩增率=多层感知机扩增率, qkv_偏差=True, 标准化=标准化)
            for i in range(深度)])
        self.编码器标准化 = 标准化(嵌入向量维度)
        # --------------------------------------------------------------------------
        self.截取 = nn.Identity()
        self.分类头 = nn.Linear(嵌入向量维度, 类别数)
        torch.nn.init.normal_(self.分类嵌入, std=.02)
        位置嵌入向量 = get_2d_sincos_pos_embed(self.位置嵌入向量.shape[-1], int(图像块数量 ** .5),
                                            cls_token=True)
        self.位置嵌入向量.data.copy_(torch.from_numpy(位置嵌入向量).float().unsqueeze(0))
        self.apply(self._初始化权重)

    def _初始化权重(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        x = self.图像块嵌入向量(x) # 图像块转化为嵌入向量 [1*361*256]
        # 加上位置嵌入向量
        x = x + self.位置嵌入向量[:, 1:, :] # 从1开始是因为0对应cls_token
        # 添加分类嵌入向量
        分类嵌入 = self.分类嵌入 + self.位置嵌入向量[:, :1, :]
        分类嵌入 = 分类嵌入.expand(x.shape[0], -1, -1)
        x = torch.cat((分类嵌入, x), dim=1)
        x = self.编码块堆叠(x)
        x = self.编码器标准化(x)
        x = self.截取(x[:, 0])
        x = self.分类头(x)

        return x


class 有监督分类模型3M(nn.Module):
    """
    输入批量大于1
    有监督训练微调阶段。现在考虑传进来的数据是未分块图像，图像是单通道。
    """
    def __init__(self, 批量, 图像形状, 类别数 = 2, 图像块大小=16, 嵌入向量维度=256, 深度=1, 注意力头数量=1, 多层感知机扩增率=4.,
                 标准化=nn.LayerNorm):
        """

        :param 图像形状: 3M图像是304,6M图像是400
        :param 类别数:
        :param 图像块大小:
        :param 嵌入向量维度:
        :param 深度:
        :param 注意力头数量:
        :param 多层感知机扩增率:
        :param 标准化:
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # 编码器实现
        self.图像块嵌入向量_微调 = 二维图像块嵌入6M(嵌入向量维度)
        # 对原图随机裁剪之后的图像形状
        图像块数量 = (图像形状 // 图像块大小) ** 2
        # 第一个轴的1表示批量维度
        self.分类嵌入_微调 = nn.Parameter(torch.zeros(批量, 1, 嵌入向量维度))
        self.位置嵌入向量_微调 = nn.Parameter(torch.zeros(批量, 图像块数量 + 1, 嵌入向量维度), requires_grad=False)  # fixed sin-cos embedding

        # 作者没给transformer编码块传入dropout的丢弃率
        self.编码块堆叠 = nn.Sequential(*[
            编码块(特征维度=嵌入向量维度, 注意力头数量=注意力头数量, 多层感知机扩增率=多层感知机扩增率, qkv_偏差=True, 标准化=标准化)
            for i in range(深度)])
        self.编码器标准化 = 标准化(嵌入向量维度)
        # --------------------------------------------------------------------------
        self.截取 = nn.Identity()
        self.分类头 = nn.Linear(嵌入向量维度, 类别数)
        torch.nn.init.normal_(self.分类嵌入_微调, std=.02)
        位置嵌入向量 = get_2d_sincos_pos_embed(self.位置嵌入向量_微调.shape[-1], int(self.图像块嵌入向量_微调.图像块数量 ** .5),
                                            cls_token=True)
        self.位置嵌入向量_微调.data.copy_(torch.from_numpy(位置嵌入向量).float().unsqueeze(0))
        self.apply(self._初始化权重)

    def _初始化权重(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        x = self.图像块嵌入向量_微调(x) # 图像块转化为嵌入向量 [1*361*256]
        # 加上位置嵌入向量
        x = x + self.位置嵌入向量_微调[:, 1:, :] # 从1开始是因为0对应cls_token
        # 添加分类嵌入向量
        分类嵌入 = self.分类嵌入_微调 + self.位置嵌入向量_微调[:, :1, :]
        分类嵌入 = 分类嵌入.expand(x.shape[0], -1, -1)
        x = torch.cat((分类嵌入, x), dim=1)
        x = self.编码块堆叠(x)
        x = self.编码器标准化(x)
        x = self.截取(x[:, 0])
        x = self.分类头(x)

        return x


if __name__ == '__main__':
    # 测试模型 = 自注意力(嵌入向量维度=512, 注意力头数量=8, 自注意力丢弃率=0.2, 全连接层丢弃率=0.2)
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(256,512), batch_size=3, device='cuda') # [3, 256, 512]

    # 测试模型 = 多层感知机(512, 200, 2048, 丢弃率=0.2)
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(256,512), batch_size=3, device='cuda') # [3, 256, 200]

    # 测试模型 = 编码块(特征维度=512, 注意力头数量=8, 自注意力丢弃率=0.2, 全连接层丢弃率=0.2, drop_path_ratio=0.2)
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(256,512), batch_size=3, device='cuda') # [3, 256, 512]

    # 测试模型 = 二维图像块嵌入6M()
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(1, 868, 868), batch_size=1, device='cuda') # [3, 256, 512]

    # 测试模型 = 三维掩码投影网络() # 输出[3, 256, 512]
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(2, 256, 16, 16), batch_size=1, device='cuda')

    # 测试模型 = 三维掩码投影网络() # 输出[1, 256, 1]
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(2, 64, 16, 16), batch_size=1, device='cuda')

    # 测试模型 = 二维掩码网络自监督阶段()
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(361, 16, 16), batch_size=1, device='cuda')

    # 测试模型 = 二维掩码网络自监督阶段()
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(625, 16, 16), batch_size=1, device='cuda')

    # 测试模型 = 有监督分类模型6M(图像形状=400, 深度=1, 注意力头数量=1)
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(1, 868, 868), batch_size=1, device='cuda')

    测试模型 = 有监督分类模型6M多批量(批量大小=2, 图像形状=400, 深度=1, 注意力头数量=1)
    测试模型.to(torch.device('cuda:0'))
    print('\n')
    summary(测试模型, input_size=(1, 868, 868), batch_size=2, device='cuda')