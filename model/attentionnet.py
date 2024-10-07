
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
import math
import json
import copy
import torch.nn.functional as F
from os import path, makedirs

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class Transblock(nn.Module):
    def __init__(self):
        super(Transblock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv_1 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn_1 = norm_layer(128)
        self.relu_1 = nn.ReLU(inplace=False)
    def forward(self, x):
        x0 = self.conv_1(x)  # x：1024*14*14
        x0 = self.bn_1(x0)
        x0 = self.relu_1(x0)  # 128*7*7
        x0 = x0.flatten(1)
        x0 = x0.unsqueeze(1)

        return x0






class FusionBlock(nn.Module):
    def __init__(self, global_fer, local_fer):
        super(FusionBlock, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(6)
        self.relu2 = nn.ReLU(6)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(128)
        self.softmax = nn.Softmax(dim=1)
        self.GlobalAvgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6272, 7)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, global_fer, local_fer):
        local_fer = self.conv1(local_fer)
        cat_all = torch.cat([global_fer, local_fer], dim=2)
        x_0 = self.conv2(cat_all)
        x = self.bn1(x_0)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x + cat_all
        x = self.softmax(x)
        x1 = global_fer * x
        x2 = local_fer * (1-x)
        x = x1 + x2
        x = x.permute(0,3,1,2)
        x = self.GlobalAvgpool(x).squeeze(3).squeeze(2)
        out = self.fc(x)
        out = self.dropout(out)

        return out



class eca_layer(nn.Module):


    def __init__(self, channel, k_size=5):
        super(eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Split_channel(nn.Module):
    def __init__(self, inplanes=1024):
        super(Split_channel, self).__init__()
        norm_layer = nn.BatchNorm2d
        #inplanes = int(x.size()[1])
        scale_width = int(inplanes / 4)  # 通道数planes/4，将X划分为4个映射子集，N=4
        self.scale_width = scale_width
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        # up_conv
        self.conv1_1 = conv3x3(scale_width, scale_width)
        self.bn1_1 = norm_layer(scale_width)
        self.conv1_2 = conv3x3(scale_width, scale_width)
        self.bn1_2 = norm_layer(scale_width)
        self.conv1_3 = conv3x3(scale_width, scale_width)
        self.bn1_3 = norm_layer(scale_width)
        self.conv1_4 = conv3x3(scale_width, scale_width)
        self.bn1_4 = norm_layer(scale_width)
        # down_conv
        self.conv2_1 = conv3x3(scale_width, scale_width)
        self.bn2_1 = norm_layer(scale_width)
        self.conv2_2 = conv3x3(scale_width, scale_width)
        self.bn2_2 = norm_layer(scale_width)
        self.conv2_3 = conv3x3(scale_width, scale_width)
        self.bn2_3 = norm_layer(scale_width)
        self.conv2_4 = conv3x3(scale_width, scale_width)
        self.bn2_4 = norm_layer(scale_width)
        self.bn_ds = norm_layer(128)

        self.eca = eca_layer(channel=inplanes, k_size=5)
    def forward(self, x):
        spatial_list = []
        sp_x = torch.split(x, self.scale_width, 1)
        out_1_1 = self.conv1_1(sp_x[0])
        out_1_1 = self.bn1_1(out_1_1)
        out_1_1_relu = self.relu(out_1_1)
        avgout = torch.mean(out_1_1, dim=1, keepdim=True)
        maxout, _ = torch.max(out_1_1, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out_1 = out * out_1_1
        out_1 = self.relu(out_1)
        spatial_list.append(out_1)

        out_1_2 = self.conv1_2(out_1_1_relu + sp_x[1])
        out_1_2 = self.bn1_2(out_1_2)
        out_1_2_relu = self.relu(out_1_2)
        avgout = torch.mean(out_1_2, dim=1, keepdim=True)
        maxout, _ = torch.max(out_1_2, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out_2 = out * out_1_2
        out_2 = self.relu(out_2)
        spatial_list.append(out_2)

        out_1_3 = self.conv1_3(out_1_2_relu + sp_x[2])
        out_1_3 = self.bn1_3(out_1_3)
        out_1_3_relu = self.relu(out_1_3)
        avgout = torch.mean(out_1_3, dim=1, keepdim=True)
        maxout, _ = torch.max(out_1_3, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out_3 = out * out_1_3
        out_3 = self.relu(out_3)
        spatial_list.append(out_3)

        out_1_4 = self.conv1_4(out_1_3_relu + sp_x[3])
        out_1_4 = self.bn1_4(out_1_4)
        avgout = torch.mean(out_1_4, dim=1, keepdim=True)
        maxout, _ = torch.max(out_1_4, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out_4 = out * out_1_4
        out_4 = self.relu(out_4)
        spatial_list.append(out_4)

        out_2_1 = self.conv2_1(spatial_list[3])
        out_2_1 = self.bn2_1(out_2_1)
        out_2_1_relu = self.relu(out_2_1)
        out_e1 = self.eca(out_2_1)

        out_2_2 = self.conv2_2(spatial_list[2]+out_2_1_relu)
        out_2_2 = self.bn2_2(out_2_2)
        out_2_2_relu = self.relu(out_2_2)
        out_e2 = self.eca(out_2_2)

        out_2_3 = self.conv2_3(spatial_list[1]+out_2_2_relu)
        out_2_3 = self.bn2_3(out_2_3)
        out_2_3_relu = self.relu(out_2_3)
        out_e3 = self.eca(out_2_3)

        out_2_4 = self.conv2_4(spatial_list[0]+out_2_3_relu)
        out_2_4 = self.bn2_4(out_2_4)
        out_e4 = self.eca(out_2_4)

        out_cat = torch.cat([out_e1, out_e2, out_e3, out_e4], dim=1)
        out_cat = self.conv5(out_cat)
        out_cat = self.bn_ds(out_cat)
        out_cat = self.relu(out_cat)
        return out_cat



class RegionInBlock(nn.Module):
    def __init__(self, inplanes_f=1024):
        super(RegionInBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        inplanes = int(inplanes_f/8)
        self.relu = nn.ReLU(inplace=False)
        self.conv3_1 = conv3x3(inplanes, inplanes)
        self.bn3_1 = norm_layer(inplanes)
        self.conv3_2 = conv3x3(inplanes, inplanes)
        self.bn3_2 = norm_layer(inplanes)
        self.conv3_3 = conv3x3(inplanes, inplanes)
        self.bn3_3 = norm_layer(inplanes)
        self.conv3_4 = conv3x3(inplanes, inplanes)
        self.bn3_4 = norm_layer(inplanes)
        self.split_channel = Split_channel(inplanes=1024)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(6272),
            nn.Linear(6272, 8)
        )
    def forward(self, attention_branch_feat):

        x = attention_branch_feat

        patch_1_spatial = x[:, :, 0:7, 0:7]
        
        patch_1 = self.split_channel(patch_1_spatial)
        patch_1_c = self.conv6(patch_1_spatial)
        patch_1_c = self.bn3_1(patch_1_c)
        patch_1_c = self.relu(patch_1_c)

        patch_1 = patch_1_c + patch_1


        patch_1_unsuq = patch_1.unsqueeze(1)
        patch_1_flat = patch_1_unsuq.flatten(2)


        patch_2_spatial = x[:, :, 0:7, 7:14]

        patch_2 = self.split_channel(patch_2_spatial)
        patch_2_c = self.conv6(patch_2_spatial)
        patch_2_c = self.bn3_2(patch_2_c)
        patch_2_c = self.relu(patch_2_c)
        patch_2 = patch_2_c + patch_2
        patch_2_unsuq = patch_2.unsqueeze(1)
        patch_2_flat = patch_2_unsuq.flatten(2)

        patch_3_spatial = x[:, :, 7:14, 0:7]
        patch_3 = self.split_channel(patch_3_spatial)
        patch_3_c = self.conv6(patch_3_spatial)
        patch_3_c = self.bn3_3(patch_3_c)
        patch_3_c = self.relu(patch_3_c)
        patch_3 = patch_3_c + patch_3
        patch_3_unsuq = patch_3.unsqueeze(1)
        patch_3_flat = patch_3_unsuq.flatten(2)

        patch_4_spatial = x[:, :, 7:14, 7:14]
        patch_4 = self.split_channel(patch_4_spatial)
        patch_4_c = self.conv6(patch_4_spatial)
        patch_4_c = self.bn3_4(patch_4_c)
        patch_4_c = self.relu(patch_4_c)
        patch_4 = patch_4_c + patch_4

        patch_up = torch.cat([patch_1, patch_2], dim=3)
        patch_down = torch.cat([patch_3, patch_4], dim=3)        
        patch_cat = torch.cat([patch_up, patch_down], dim=2)
        patch_cat = self.maxpool(patch_cat)
        rf_unseq = patch_cat.unsqueeze(1)
        rf = rf_unseq.flatten(2)

        patch_4_unsuq = patch_4.unsqueeze(1)
        patch_4_flat = patch_4_unsuq.flatten(2)
        patch_into_att = torch.cat([patch_1_flat, patch_2_flat, patch_3_flat, patch_4_flat, rf], dim=1)
        
        return patch_into_att

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

'''FeedForeward'''
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Cross_Attention(nn.Module):
    def __init__(self, dim, num_head):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.qkv = nn.Linear(dim, dim * 3)  # extend the dimension for later spliting

    def forward(self, x, f):
        B, N, C = x.shape   # B 16    C 6272   N 5
        b, n, c = f.shape   # c 6272  b 16    n 1
        qkv_f = self.qkv(f).reshape(b, n, 3, self.num_head, c // self.num_head).permute(2, 0, 3, 1, 4)
        # qkv_f: Tensor 3,16,1,5,6272
        k, v = qkv_f[1], qkv_f[2]
        # k, v :16,1,5,6272
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        # qkv: 3,16,1,1,6272
        q = qkv[0]
        # q:16,1,1,6272
        att = q @ k.transpose(-1, -2) / math.sqrt(C)
        #
        att = att.softmax(dim=1)

        x = (att @ v).transpose(1, 2)

        x = x.reshape(B, N, C)
        # print(x.shpape)
        return x

class Attention(nn.Module):
    def __init__(self, dim,  heads=1, dim_head=6272, dropout=0.2):
        super().__init__()
        inner_dim = dim_head * heads
        # project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_q = nn.Linear(dim, inner_dim * 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads    # b:16 n:5  _:6272   h:1 单头
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # qkv:tuple:3   (16,6,6272)  # (b, n(5), dim*3) ---> 3 * (b, n, dim)
        # q_fk_fv_f = self.to_q(f).chunk(3, dim=-1)
        # q_f, k_f, v_f = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), q_fk_fv_f)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(6272))
        # q,k,v: (16,1,6,6272)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale     # dots: (16,1,5,5)
        # dots_f = einsum('b h i d, b h j d -> b h i j', q_f, k) * self.scale  # dots_f: (16,1,5,5)
        # attn_f = self.attend(dots_f)

        attn = self.attend(dots)
        # (16,1,5,5)
        # out_f = einsum('b h i j, b h j d -> b h i d', attn_f, v)  # 应该为 out_f :(16,1,6,64)
        # out_f = rearrange(out_f, 'b h n d -> b n (h d)')    # 应该为 out_f :(16,1,64)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # 自注意力机制加权特征  # out:(16,1,6,6272)
        out = rearrange(out, 'b h n d -> b n (h d)')  # out:(16,5,6272)
        out = out + x
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x_out = attn(x) + x
            #x = ff(x) + x
        return x_out



class ViT(nn.Module):
    def __init__(self, *,  num_classes, dim, depth, heads, mlp_dim, pool='mean', channels=1024, dim_head=6272, dropout=0.2, emb_dropout=0.2):
        super().__init__()


        num_patches = 5   # N：num_patches=4
        patch_dim = channels * 14 * 14   # D：512*14*14
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=14, p2=14),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape           # b表示batchSize

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)                                                 # (b, 65, dim)
        return x                                                 #  (b, num_classes)



class F_to_F(nn.Module):
    def __init__(self, dim=6272, num_classes=7):
        super(F_to_F, self).__init__()
        self.vit_model = ViT(num_classes=7, dim=6272, depth=2, heads=1, mlp_dim=1024, dropout=0.1, emb_dropout=0.1)
        self.cross_model = Cross_Attention(dim=6272, num_head=1)
        self.fusion = FusionBlock(global_fer=6, local_fer=1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Dropout(0.5)
        )
    def forward(self, x, f):
        x_out = self.vit_model(x)
        f_out = self.cross_model(f, x)
        out = torch.cat([x_out, f_out, f], dim=1)
        out = torch.mean(out, dim=1, keepdim=True)
        out = out.squeeze(1)
        out = self.mlp_head(out)


        return out


class Attentiomasknet(nn.Module):
    def __init__(self, inputdim=512):
        super(Attentiomasknet, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(inputdim, inputdim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(inputdim),     # PReLU -----ReLU
            nn.BatchNorm2d(inputdim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.attention(x)
        return y

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out





class eca_layer(nn.Module):

    def __init__(self, channel, k_size=5):
        super(eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1,  dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

# @wy 定义1X1卷积函数
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AttentionBlock(nn.Module):

    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(channel=planes, k_size=5)
        self.spatial_att = SpatialAttentionModule()
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.eca(out)
        out = self.spatial_att(out) * out


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# =============================trying-first
class RegionBranch(nn.Module):
    def __init__(self, block_a, inputdim=512, num_classes=7):
        super(RegionBranch, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes

        self.patch_1_1 = self._make_layer(block_a, 512, 128, stride=1)
        self.patch_1_2 = self._make_layer(block_a, 128, 128, stride=1)

        self.patch_2_1 = self._make_layer(block_a, 512, 128, stride=1)
        self.patch_2_2 = self._make_layer(block_a, 128, 128, stride=1)

        self.patch_3_1 = self._make_layer(block_a, 512, 128, stride=1)
        self.patch_3_2 = self._make_layer(block_a, 128, 128, stride=1)

        self.patch_4_1 = self._make_layer(block_a, 512, 128, stride=1)
        self.patch_4_2 = self._make_layer(block_a, 128, 128, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, inplanes, planes, stride=1):
        norm_layer =self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes,planes, stride, downsample))
        return nn.Sequential(*layers)



    def _forward_impl(self, attention_branch_feat):
        patches_list = []
        x = attention_branch_feat

        patch_1 = x[:, :, 0:14, 0:14]
        patches_list.append(patch_1)

        patch_2 = x[:, :, 0:14, 14:28]
        patches_list.append(patch_2)

        patch_3 = x[:, :, 14:28, 0:14]

        patches_list.append(patch_3)

        patch_4 = x[:, :, 14:28, 14:28]
        patches_list.append(patch_4)

        branch_1_1_out = self.patch_1_1(patch_1)
        branch_1_1_out = self.patch_1_2(branch_1_1_out)

        branch_2_1_out = self.patch_2_1(patch_2)
        branch_2_1_out = self.patch_2_2(branch_2_1_out)

        branch_3_1_out = self.patch_3_1(patch_3)
        branch_3_1_out = self.patch_3_2(branch_3_1_out)

        branch_4_1_out = self.patch_4_1(patch_4)
        branch_4_1_out = self.patch_4_2(branch_4_1_out)

        branch_out_1 = torch.cat([branch_1_1_out,branch_2_1_out], dim=3)
        branch_out_2 = torch.cat([branch_3_1_out, branch_4_1_out], dim=3)
        branch_out = torch.cat([branch_out_1, branch_out_2], dim=2)

        return branch_out

    def forward(self, x):
        return self._forward_impl(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



