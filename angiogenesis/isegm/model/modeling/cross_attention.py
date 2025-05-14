import torch.nn as nn

from .models_vit import Mlp
import torch
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """ Multi-head cross-attention operation
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask = None

    def forward(self, query, target):
        assert query.shape == target.shape

        B, N, C = query.shape
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k(target).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.v(target).reshape(B, N, self.num_heads, C // self.num_heads)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        scores = (q @ k.transpose(-2, -1)) * self.scale
            #如果mask为控或其尺寸与输入图像大小不匹配，则生成新的随机二值mask
        if self.mask is None or self.mask.size(-1) != N:
            #创建一个二值mask,随机生成0或1的值
            binary_mask = torch.randint(0, 2, (B, 1, N), device=query.device).float()
            binary_mask =binary_mask.view(B,-1)

            #将大于0.5的值设为0，小于0.5的值设置为负无穷（即遮蔽这些区域,特征全部抹掉）
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0).to(query.device),torch.tensor(float('-inf')).to(query.device))
            self.mask = processed_mask.unsqueeze(1).expand(-1, N, -1)  # 扩展为(btach_size, height * width, height * width)
            if self.mask.ndim < 4:
                self.mask = self.mask.unsqueeze(1)
        # # 将mask添加到注意力分数中
        # # 将 mask 的形状扩展到与 scores 一致
        # scores = scores + self.mask
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1,2).contiguous().reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class CrossBlock(nn.Module):
    """ Multi-head cross-attention block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_drop=0., qkv_bias=False,
            attn_drop=0., proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=proj_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=mlp_drop)

    def forward(self, query, target):
        query = query + self.attn(self.norm1(query), target)
        query = query + self.mlp(self.norm2(query))
        return query



class MaskAttention(nn.Module):
    def __init__(self, channels, size):
        super(MaskAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.mask = None
        self.norm = nn.LayerNorm(channels)

    def forward(self, prompt_feat, x):
        batch_size, channels, height, width = x.size()
        if channels != self.channels:
            raise ValueError(f"Expected input with {self.channels} channels, but got {channels} channels.")
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
        prompt_feat = prompt_feat.view(batch_size, channels, height * width).permute(0, 2, 1)
        #计算query, key, value
        Q = self.query(prompt_feat)
        K = self.key(x)
        V = self.value(x)

        #计算注意力分数(scaled dot-product)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.channels ** 0.5) #对分数进行缩放

        #如果mask为控或其尺寸与输入图像大小不匹配，则生成新的随机二值mask
        if self.mask is None or self.mask.size(-1) != height * width:
            #创建一个二值mask,随机生成0或1的值
            binary_mask = torch.randint(0, 2, (batch_size, 1, height * width), device=x.device).float()
            binary_mask =binary_mask.view(batch_size,-1)

            #将大于0.5的值设为0，小于0.5的值设置为负无穷（即遮蔽这些区域,特征全部抹掉）
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0).to(x.device),torch.tensor(float('-inf')).to(x.device))
            self.mask = processed_mask.unsqueeze(1).expand(-1, height * width, -1)  # 扩展为(btach_size, height * width, height * width)

        # 将mask添加到注意力分数中
        scores = scores + self.mask

        # 对分数进行softmax归一化，得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 使用注意力权重对值（V），进行加权求和，得到注意力输出
        attention_output = torch.matmul(attention_weights, V)

        # 将输入与注意力输出相加，并进行标准化
        attention_output = attention_output + x
        attention_output = self.norm(attention_output)

        #恢复为原始的(batch_size, channels, height, width)形状
        return attention_output.view(batch_size, channels,height, width)



class Attention(nn.Module):
    """ Multi-head self-attention operation
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask = None
                # 预生成二值 mask 模板（0/1）
        #self.binary_mask_template = nn.Parameter(torch.randint(0, 2, (1, 1, 1024)), requires_grad=False)  # 假设最大 N=1024

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # if self.mask is None or self.mask.size(-1) != N:
        #     #创建一个二值mask,随机生成0或1的值
        #     binary_mask = torch.randint(0, 2, (B, 1, N), device=x.device).float()
        #     binary_mask =binary_mask.view(B,-1)

        #     #将大于0.5的值设为0，小于0.5的值设置为负无穷（即遮蔽这些区域,特征全部抹掉）
        #     processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0).to(x.device),torch.tensor(float('-inf')).to(x.device))
        #     self.mask = processed_mask.unsqueeze(1).expand(-1, N, -1)  # 扩展为(btach_size, height * width, height * width)
        #     if self.mask.ndim < 4:
        #         self.mask = self.mask.unsqueeze(1)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        # 将mask添加到注意力分数中
        # 将 mask 的形状扩展到与 scores 一致
        # if self.mask is not None:
        #     scores = scores + self.mask
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block_mask(nn.Module):
    """ Multi-head self-attention block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_drop=0., qkv_bias=False,
            attn_drop=0., proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=proj_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=mlp_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
