import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from isegm.model.ops import DistMaps, BatchImageNormalize
from isegm.model.efficientmodeling.models_vit import EfficientViT,PatchEmbed
from isegm.model.modeling.cross_attention import CrossBlock
from isegm.utils.serialization import serialize




# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
# Build the EfficientViT Model
# Written by: Xinyu Liu
# --------------------------------------------------------
class EfficientVitModel(nn.Module):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        head_params={},
        fusion_params={},
        norm_radius=5, 
        use_disks=False, 
        cpu_dist_maps=False, 
        norm_mean_std=([.485, .456, .406], [.229, .224, .225])        
    ) -> None:
        super().__init__()
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.dist_maps = DistMaps(
            norm_radius=norm_radius, 
            spatial_scale=1.0,
            cpu_mode=cpu_dist_maps, 
            use_disks=use_disks,
        )

        self.backbone = EfficientViT(**backbone_params)


        self.visual_prompts_encoder = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3, # prev mask + positive & negative clicks
            embed_dim=backbone_params['embed_dim'],
            flatten=True
        )

        self.target_length = self.backbone.patch_embed.img_size[0]

        self.fusion_type = fusion_params['type']
        if self.fusion_type == 'naive':
            pass
        elif self.fusion_type == 'self_attention':
            depth = int(fusion_params['depth'])
            self.fusion_blocks = nn.Sequential(
                *[Block(**fusion_params['params']) for _ in range(depth)]
            )
        elif self.fusion_type == 'cross_attention': # warning: may have bugs !!!
            depth = int(fusion_params['depth'])
            self.fusion_blocks = nn.Sequential(
                *[CrossAttentionBlock(**fusion_params['params']) for _ in range(depth)]
            )
        else:
            raise ValueError(f'Unsupported fusion type: {self.fusion_type}.')


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tenosr and resize it to target lenght

        Arguments:
            x: a tensor of shape [B, C, H, W]

        Return:
            x: a tensor of shape [B, C, target_length, target_length]
        """
        self.orig_size = x.shape[-2:]

        # normalize and resize image
        x = self.normalization(x)
        x = F.interpolate(x, self.target_length, mode="bilinear", align_corners=False)

        return x

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize the input tensor to the original size

        Arguments:
            x: tensor of shape [B, C, H, W]
        """
        # resize
        x = F.interpolate(x, self.orig_size, mode='bilinear', align_corners=False)
        return x

    def get_image_feats(self, image, keep_shape=True):
        image = self.preprocess(image)
        image_feats = self.backbone(image, keep_shape=keep_shape)
        return image_feats

    def get_prompt_feats(self,image_size, inputs,keep_shape=True):
        """Get feature representation for prompts
        
        Arguments:
            image_shape: original image shape
            prompts: a dictionary containing all possible prompts

        Returns:
            prompt features
        """
        # resize the previous mask to the target length
        
        prev_mask = inputs['prev_mask']
        prev_mask = F.interpolate(
            prev_mask, self.target_length, mode='bilinear', align_corners=False)

        # convert points from original coordinates to resized coordinates
        points = inputs['points']
        for batch_id in range(len(points)):
            for point_id in range(len(points[batch_id])):
                if points[batch_id, point_id, 2] > -1: # pos. or neg. points
                    w, h = points[batch_id, point_id, 0], points[batch_id, point_id, 1]
                    w = int(w * (self.target_length / self.orig_size[0]) + 0.5)
                    h = int(h * (self.target_length / self.orig_size[1]) + 0.5)
                    points[batch_id, point_id, 0], points[batch_id, point_id, 1] = w, h 

        point_mask = self.dist_maps(prev_mask.shape, points)
        prompt_mask = torch.cat((prev_mask, point_mask), dim=1)
        prompt_feats = self.visual_prompts_encoder(prompt_mask)

        if keep_shape: # BNC -> BCHW
            B, N, C = prompt_feats.shape
            H = self.target_length // self.visual_prompts_encoder.patch_size[0]
            W = self.target_length // self.visual_prompts_encoder.patch_size[1]
            assert N == H*W
            prompt_feats = prompt_feats.transpose(1,2).contiguous()
            prompt_feats = prompt_feats.reshape(B, C, H, W)

        return prompt_feats

    def fusion(self, image_feats, prompt_feats):
        if self.fusion_type == 'naive':
            return image_feats + prompt_feats
        
        elif self.fusion_type == 'cross_attention':
            num_blocks = len(self.fusion_blocks)
            for i in range(num_blocks):
                image_feats, prompt_feats = self.fusion_blocks[i](
                    image_feats, prompt_feats, keep_shape=True)
            return image_feats
        
        elif self.fusion_type == 'self_attention':
            image_feats = image_feats + prompt_feats
            B, C, H, W = image_feats.shape
            image_feats = image_feats.permute(0,2,3,1).contiguous().reshape(B,H*W,C)

            num_blocks = len(self.fusion_blocks)
            for i in range(num_blocks):
                image_feats = self.fusion_blocks[i](image_feats)            
            image_feats = image_feats.transpose(1,2).contiguous().reshape(B,C,H,W)
            return image_feats

        else:
            raise ValueError(f'Unsupported fusion type: {self.fusion_type}.')

    def forward(self, image_shape, image_feats, prompt_feats):
        """Segment an image object given prompts
        """
        #if True:
        # image_feats = self.backbone(image_feats, keep_shape=True)

        fused_features = self.fusion(image_feats, prompt_feats)
        pyramid_features = self.neck(fused_features)
        seg_prob = self.head(pyramid_features)
        seg_prob = F.interpolate(
            seg_prob, 
            size=self.target_length, 
            mode='bilinear', 
            align_corners=True
        )

        seg_prob = self.postprocess(seg_prob)
        return {'instances': seg_prob}
    