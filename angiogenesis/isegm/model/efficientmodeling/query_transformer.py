
from typing import Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn

from mask_attention import aggregate
from transformer_layers import *
import math
import numpy as np
def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding(nn.Module):
    def __init__(self,
                 dim: int,
                 scale: float = math.pi * 2,
                 temperature: float = 10000,
                 normalize: bool = True,
                 channel_last: bool = True,
                 transpose_output: bool = False):
        super().__init__()
        dim = int(np.ceil(dim / 4) * 2)
        self.dim = dim
        inv_freq = 1.0 / (temperature**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.normalize = normalize
        self.scale = scale
        self.eps = 1e-6
        self.channel_last = channel_last
        self.transpose_output = transpose_output

        self.cached_penc = None  # the cache is irrespective of the number of objects

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A 4/5d tensor of size
            channel_last=True: (batch_size, h, w, c) or (batch_size, k, h, w, c)
            channel_last=False: (batch_size, c, h, w) or (batch_size, k, c, h, w)
        :return: positional encoding tensor that has the same shape as the input if the input is 4d
                 if the input is 5d, the output is broadcastable along the k-dimension
        """
        if len(tensor.shape) != 4 and len(tensor.shape) != 5:
            raise RuntimeError(f'The input tensor has to be 4/5d, got {tensor.shape}!')

        if len(tensor.shape) == 5:
            # take a sample from the k dimension
            num_objects = tensor.shape[1]
            tensor = tensor[:, 0]
        else:
            num_objects = None

        if self.channel_last:
            batch_size, h, w, c = tensor.shape
        else:
            batch_size, c, h, w = tensor.shape

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            if num_objects is None:
                return self.cached_penc
            else:
                return self.cached_penc.unsqueeze(1)

        self.cached_penc = None

        pos_y = torch.arange(h, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_x = torch.arange(w, device=tensor.device, dtype=self.inv_freq.dtype)
        if self.normalize:
            pos_y = pos_y / (pos_y[-1] + self.eps) * self.scale
            pos_x = pos_x / (pos_x[-1] + self.eps) * self.scale

        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_x = get_emb(sin_inp_x)

        emb = torch.zeros((h, w, self.dim * 2), device=tensor.device, dtype=tensor.dtype)
        emb[:, :, :self.dim] = emb_x
        emb[:, :, self.dim:] = emb_y

        if not self.channel_last and self.transpose_output:
            # cancelled out
            pass
        elif (not self.channel_last) or (self.transpose_output):
            emb = emb.permute(2, 0, 1)

        self.cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if num_objects is None:
            return self.cached_penc
        else:
            return self.cached_penc.unsqueeze(1)



class GConv2d(nn.Conv2d):
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])

class QueryTransformerBlock(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_transformer
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries
        self.ff_dim = this_cfg.ff_dim

        self.read_from_pixel = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=this_cfg.read_from_pixel.add_pe_to_qkv)
        self.self_attn = SelfAttention(self.embed_dim,
                                       self.num_heads,
                                       add_pe_to_qkv=this_cfg.query_self_attention.add_pe_to_qkv)
        self.ffn = FFN(self.embed_dim, self.ff_dim)
        self.read_from_query = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=this_cfg.read_from_query.add_pe_to_qkv,
                                              norm=this_cfg.read_from_query.output_norm)
        self.pixel_ffn = PixelFFN(self.embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            pixel: torch.Tensor,
            query_pe: torch.Tensor,
            pixel_pe: torch.Tensor,
            attn_mask: torch.Tensor,
            need_weights: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        # x: (bs*num_objects)*num_queries*embed_dim
        # pixel: bs*num_objects*C*H*W
        # query_pe: (bs*num_objects)*num_queries*embed_dim
        # pixel_pe: (bs*num_objects)*(H*W)*C
        # attn_mask: (bs*num_objects*num_heads)*num_queries*(H*W)

        # bs*num_objects*C*H*W -> (bs*num_objects)*(H*W)*C
        pixel_flat = pixel.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        x, q_weights = self.read_from_pixel(x,
                                            pixel_flat,
                                            query_pe,
                                            pixel_pe,
                                            attn_mask=attn_mask,
                                            need_weights=need_weights)
        x = self.self_attn(x, query_pe)
        x = self.ffn(x)

        pixel_flat, p_weights = self.read_from_query(pixel_flat,
                                                     x,
                                                     pixel_pe,
                                                     query_pe,
                                                     need_weights=need_weights)
        pixel = self.pixel_ffn(pixel, pixel_flat)

        if need_weights:
            bs, num_objects, _, h, w = pixel.shape
            q_weights = q_weights.view(bs, num_objects, self.num_heads, self.num_queries, h, w)
            p_weights = p_weights.transpose(2, 3).view(bs, num_objects, self.num_heads,
                                                       self.num_queries, h, w)

        return x, pixel, q_weights, p_weights


class QueryTransformer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_transformer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries

        # query initialization and embedding
        self.query_init = nn.Embedding(self.num_queries, self.embed_dim)
        self.query_emb = nn.Embedding(self.num_queries, self.embed_dim)

        # projection from object summaries to query initialization and embedding
        self.summary_to_query_init = nn.Linear(self.embed_dim, self.embed_dim)
        self.summary_to_query_emb = nn.Linear(self.embed_dim, self.embed_dim)

        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature
        self.pixel_init_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.pixel_emb_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.spatial_pe = PositionalEncoding(self.embed_dim,
                                             scale=self.pixel_pe_scale,
                                             temperature=self.pixel_pe_temperature,
                                             channel_last=False,
                                             transpose_output=True)

        # transformer blocks
        self.num_blocks = this_cfg.num_blocks
        self.blocks = nn.ModuleList(
            QueryTransformerBlock(model_cfg) for _ in range(self.num_blocks))
        self.mask_pred = nn.ModuleList(
            nn.Sequential(nn.ReLU(), GConv2d(self.embed_dim, 1, kernel_size=1))
            for _ in range(self.num_blocks + 1))

        self.act = nn.ReLU(inplace=True)

    def forward(self,
                pixel: torch.Tensor,
                obj_summaries: torch.Tensor,
                selector: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> (torch.Tensor, Dict[str, torch.Tensor]):
        # pixel: B*num_objects*embed_dim*H*W
        # obj_summaries: B*num_objects*T*num_queries*embed_dim
        T = obj_summaries.shape[2]
        bs, num_objects, _, H, W = pixel.shape

        # normalize object values
        # the last channel is the cumulative area of the object
        obj_summaries = obj_summaries.view(bs * num_objects, T, self.num_queries,
                                           self.embed_dim + 1)
        # sum over time
        # during inference, T=1 as we already did streaming average in memory_manager
        obj_sums = obj_summaries[:, :, :, :-1].sum(dim=1)
        obj_area = obj_summaries[:, :, :, -1:].sum(dim=1)
        obj_values = obj_sums / (obj_area + 1e-4)
        obj_init = self.summary_to_query_init(obj_values)
        obj_emb = self.summary_to_query_emb(obj_values)

        # positional embeddings for object queries
        query = self.query_init.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + obj_init
        query_emb = self.query_emb.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + obj_emb

        # positional embeddings for pixel features
        pixel_init = self.pixel_init_proj(pixel)
        pixel_emb = self.pixel_emb_proj(pixel)
        pixel_pe = self.spatial_pe(pixel.flatten(0, 1))
        pixel_emb = pixel_emb.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        pixel_pe = pixel_pe.flatten(1, 2) + pixel_emb

        pixel = pixel_init

        # run the transformer
        aux_features = {'logits': []}

        # first aux output
        aux_logits = self.mask_pred[0](pixel).squeeze(2)
        attn_mask = self._get_aux_mask(aux_logits, selector)
        aux_features['logits'].append(aux_logits)
        for i in range(self.num_blocks):
            query, pixel, q_weights, p_weights = self.blocks[i](query,
                                                                pixel,
                                                                query_emb,
                                                                pixel_pe,
                                                                attn_mask,
                                                                need_weights=need_weights)

            if self.training or i <= self.num_blocks - 1 or need_weights:
                aux_logits = self.mask_pred[i + 1](pixel).squeeze(2)
                attn_mask = self._get_aux_mask(aux_logits, selector)
                aux_features['logits'].append(aux_logits)

        aux_features['q_weights'] = q_weights  # last layer only
        aux_features['p_weights'] = p_weights  # last layer only

        if self.training:
            # no need to save all heads
            aux_features['attn_mask'] = attn_mask.view(bs, num_objects, self.num_heads,
                                                       self.num_queries, H, W)[:, :, 0]

        return pixel, aux_features

    def _get_aux_mask(self, logits: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        # logits: batch_size*num_objects*H*W
        # selector: batch_size*num_objects*1*1
        # returns a mask of shape (batch_size*num_objects*num_heads)*num_queries*(H*W)
        # where True means the attention is blocked

        if selector is None:
            prob = logits.sigmoid()
        else:
            prob = logits.sigmoid() * selector
        logits = aggregate(prob, dim=1)

        is_foreground = (logits[:, 1:] >= logits.max(dim=1, keepdim=True)[0])
        foreground_mask = is_foreground.bool().flatten(start_dim=2)
        inv_foreground_mask = ~foreground_mask
        inv_background_mask = foreground_mask

        aux_foreground_mask = inv_foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_background_mask = inv_background_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)

        aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)

        aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False

        return aux_mask





if __name__ == '__main__':
    pe = PositionalEncoding(8).cuda()
    input = torch.ones((1, 8, 8, 8)).cuda()
    output = pe(input)
    # print(output)
    print(output[0, :, 0, 0])
    print(output[0, :, 0, 5])
    print(output[0, 0, :, 0])
    print(output[0, 0, 0, :])



