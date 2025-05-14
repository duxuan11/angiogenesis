import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MaskBlock(nn.Module):
    """ Multi-head self-attention block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_drop=0., qkv_bias=False,
            attn_drop=0., proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = MaskAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=proj_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=mlp_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MaskAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(MaskAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.mask = None
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #计算注意力分数(scaled dot-product)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        #如果mask为控或其尺寸与输入图像大小不匹配，则生成新的随机二值mask
        if self.mask is None or self.mask.size(-1) != N:
            #创建一个二值mask,随机生成0或1的值
            binary_mask = torch.randint(0, 2, (batch_size, 1, N), device=x.device).float()
            binary_mask =binary_mask.view(batch_size,-1)

            #将大于0.5的值设为0，小于0.5的值设置为负无穷（即遮蔽这些区域,特征全部抹掉）
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0).to(x.device),torch.tensor(float('-inf')).to(x.device))
            self.mask = processed_mask.unsqueeze(1).expand(-1,N, -1)  # 扩展为(btach_size, height * width, height * width)

        # 将mask添加到注意力分数中
        scores = scores + self.mask

        # 对分数进行softmax归一化，得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 使用注意力权重对值（V），进行加权求和，得到注意力输出
        attention_output = torch.matmul(attention_weights, v)

        # 将输入与注意力输出相加，并进行标准化
        attention_output = attention_output + x
        attention_output = self.norm(attention_output)

        #恢复为原始的(batch_size, channels, height, width)形状
        return attention_output.view(batch_size, channels,height, width)

# class MaskAttention(nn.Module):
#     def __init__(self, channels, size):
#         super(MaskAttention, self).__init__()
#         self.channels = channels
#         self.size = size
#         self.query = nn.Linear(channels, channels)
#         self.key = nn.Linear(channels, channels)
#         self.value = nn.Linear(channels, channels)

#         self.mask = None
#         self.norm = nn.LayerNorm(channels)

#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         if channels != self.channels:
#             raise ValueError(f"Expected input with {self.channels} channels, but got {channels} channels.")
#         x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

#         #计算query, key, value
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)

#         #计算注意力分数(scaled dot-product)
#         scores = torch.matmul(Q, K.transpose(-2, -1))
#         scores = scores / (self.channels ** 0.5) #对分数进行缩放

#         #如果mask为控或其尺寸与输入图像大小不匹配，则生成新的随机二值mask
#         if self.mask is None or self.mask.size(-1) != height * width:
#             #创建一个二值mask,随机生成0或1的值
#             binary_mask = torch.randint(0, 2, (batch_size, 1, height * width), device=x.device).float()
#             binary_mask =binary_mask.view(batch_size,-1)

#             #将大于0.5的值设为0，小于0.5的值设置为负无穷（即遮蔽这些区域,特征全部抹掉）
#             processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0).to(x.device),torch.tensor(float('-inf')).to(x.device))
#             self.mask = processed_mask.unsqueeze(1).expand(-1, height * width, -1)  # 扩展为(btach_size, height * width, height * width)

#         # 将mask添加到注意力分数中
#         scores = scores + self.mask

#         # 对分数进行softmax归一化，得到注意力权重
#         attention_weights = F.softmax(scores, dim=-1)

#         # 使用注意力权重对值（V），进行加权求和，得到注意力输出
#         attention_output = torch.matmul(attention_weights, V)

#         # 将输入与注意力输出相加，并进行标准化
#         attention_output = attention_output + x
#         attention_output = self.norm(attention_output)

#         #恢复为原始的(batch_size, channels, height, width)形状
#         return attention_output.view(batch_size, channels,height, width)
if __name__ == "__main__":
    # # 测试代码
    # batch_size = 1
    # channels = 64
    # height = 16
    # width = 16

    # x = torch.randn(batch_size, channels, height, width)
    # x[:,:,:3,:3] = 1
    # print(x)
    # mask_attention = MaskAttention(channels, (height, width))
    # output = mask_attention(x)

    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)

    # 生成模拟输入
    batch_size, num_objects, H, W = 2, 1, 64, 64
    logits = torch.randn(batch_size, num_objects, H, W)  # 随机logits
    #selector = torch.tensor([[[1]], [[1]], [[0]]]).expand(batch_size, num_objects, 1, 1).float()  # 屏蔽第3个对象

    # 调用函数生成掩码
    aux_mask = get_aux_mask(logits, selector= None)

    # 输出形状验证
    print(aux_mask.shape)  # 预期输出: (2 * 3 * 8=48, 16, 16)
