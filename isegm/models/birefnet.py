import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import laplacian
from huggingface_hub import PyTorchModelHubMixin

from config import Config
from isegm.models.backbones.build_backbone import build_backbone
from isegm.models.modules.decoder_blocks import BasicDecBlk, ResBlk
from isegm.models.modules.lateral_blocks import BasicLatBlk
from isegm.models.modules.aspp import ASPP, ASPPDeformable
from isegm.models.refinement.refiner import Refiner, RefinerPVTInChannels4, RefUNet
from isegm.models.refinement.stem_layer import StemLayer
from utils.ops import DistMaps, BatchImageNormalize

class_labels_TR_sorted = ["a","b","c"]
class BiRefNet(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="birefnet",
    repo_url="https://github.com/ZhengPeng7/BiRefNet",
    tags=['Image Segmentation', 'Background Removal', 'Mask Generation', 'Dichotomous Image Segmentation', 'Camouflaged Object Detection', 'Salient Object Detection']
):
    def __init__(self, bb_pretrained=True,norm_radius=5,
        use_disks=False,
        cpu_dist_maps=False,
        norm_mean_std=([.485, .456, .406], [.229, .224, .225]) ):
        super(BiRefNet, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.bb = build_backbone(self.config.bb, pretrained=bb_pretrained)

        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.dist_maps = DistMaps(
            norm_radius=norm_radius,
            spatial_scale=1.0,
            cpu_mode=cpu_dist_maps,
            use_disks=use_disks,
        )
        fusion_params = {}
        fusion_params['type'] = "naive"
        self.fusion_type = fusion_params['type']
        if self.fusion_type == 'naive':
            pass
        # elif self.fusion_type == 'self_attention':
        #     depth = int(fusion_params['depth'])
        #     self.fusion_blocks = nn.Sequential(
        #         *[Block(**fusion_params['params']) for _ in range(depth)]
        #     )
        # elif self.fusion_type == 'cross_attention': # warning: may have bugs !!!
        #     depth = int(fusion_params['depth'])
        #     self.fusion_blocks = nn.Sequential(
        #         *[CrossAttentionBlock(**fusion_params['params']) for _ in range(depth)]
        #     )
        else:
            raise ValueError(f'Unsupported fusion type: {self.fusion_type}.')



        channels = self.config.lateral_channels_in_collection

        if self.config.auxiliary_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_head = nn.Sequential(
                nn.Linear(channels[0], len(class_labels_TR_sorted))
            )

        if self.config.squeeze_block:
            self.squeeze_module = nn.Sequential(*[
                eval(self.config.squeeze_block.split('_x')[0])(channels[0]+sum(self.config.cxt), channels[0])
                for _ in range(eval(self.config.squeeze_block.split('_x')[1]))
            ])

        self.decoder = Decoder(channels)

        if self.config.ender:
            self.dec_end = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),
                nn.Conv2d(16, 1, 3, 1, 1),
                nn.ReLU(inplace=True),
            )

        # refine patch-level segmentation
        if self.config.refine:
            if self.config.refine == 'itself':
                self.stem_layer = StemLayer(in_channels=3+1, inter_channels=48, out_channels=3, norm_layer='BN' if self.config.batch_size > 1 else 'LN')
            else:
                self.refiner = eval('{}({})'.format(self.config.refine, 'in_channels=3+1'))

        if self.config.freeze_bb:
            # Freeze the backbone...
            print(self.named_parameters())
            for key, value in self.named_parameters():
                if 'bb.' in key and 'refiner.' not in key:
                    value.requires_grad = False

    def forward_enc(self, x, coord_features =None):
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.bb.conv1(x); x2 = self.bb.conv2(x1); x3 = self.bb.conv3(x2); x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x, coord_features)
            if self.config.mul_scl_ipt == 'cat':
                B, C, H, W = x.shape
                x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))
                x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            elif self.config.mul_scl_ipt == 'add':
                B, C, H, W = x.shape
                x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))
                x1 = x1 + F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)
                x2 = x2 + F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)
                x3 = x3 + F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)
                x4 = x4 + F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)
        class_preds = self.cls_head(self.avgpool(x4).view(x4.shape[0], -1)) if self.training and self.config.auxiliary_classification else None
        if self.config.cxt:
            x4 = torch.cat(
                (
                    *[
                        F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),
                        F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                        F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                    ][-len(self.config.cxt):],
                    x4
                ),
                dim=1
            )
        return (x1, x2, x3, x4), class_preds

    def forward_ori(self, x, coord_features = None):
        ########## Encoder ##########
        (x1, x2, x3, x4), class_preds = self.forward_enc(x,coord_features)
        features = [x, x1, x2, x3, x4]

        if self.config.squeeze_block:
            x4 = self.squeeze_module(x4)
        ########## Decoder ##########
        features = [x, x1, x2, x3, x4]

        if self.training and self.config.out_ref:
            features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))
        scaled_preds = self.decoder(features)
        return scaled_preds, class_preds

    # def fusion(self, image_feats, prompt_feats):
    #     if self.fusion_type == 'naive':
    #         return image_feats + prompt_feats

    #     elif self.fusion_type == 'cross_attention':
    #         num_blocks = len(self.fusion_blocks)
    #         for i in range(num_blocks):
    #             image_feats, prompt_feats = self.fusion_blocks[i](
    #                 image_feats, prompt_feats, keep_shape=True)
    #         return image_feats

    #     elif self.fusion_type == 'self_attention':
    #         image_feats = image_feats + prompt_feats
    #         B, C, H, W = image_feats.shape
    #         image_feats = image_feats.permute(0,2,3,1).contiguous().reshape(B,H*W,C)

    #         num_blocks = len(self.fusion_blocks)
    #         for i in range(num_blocks):
    #             image_feats = self.fusion_blocks[i](image_feats)
    #         image_feats = image_feats.transpose(1,2).contiguous().reshape(B,C,H,W)
    #         return image_feats

    #     else:
    #         raise ValueError(f'Unsupported fusion type: {self.fusion_type}.')

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
        # prev_mask = F.interpolate(
        #     prev_mask, self.target_length, mode='bilinear', align_corners=False)

        # convert points from original coordinates to resized coordinates
        points = inputs['points']
        # for batch_id in range(len(points)):
        #     for point_id in range(len(points[batch_id])):
        #         if points[batch_id, point_id, 2] > -1: # pos. or neg. points
        #             w, h = points[batch_id, point_id, 0], points[batch_id, point_id, 1]
        #             w = int(w * (self.target_length / self.orig_size[0]) + 0.5)
        #             h = int(h * (self.target_length / self.orig_size[1]) + 0.5)
        #             points[batch_id, point_id, 0], points[batch_id, point_id, 1] = w, h

        point_mask = self.dist_maps(prev_mask.shape, points)
        prompt_mask = torch.cat((prev_mask, point_mask), dim=1)
        #prompt_feats = self.visual_prompts_encoder(prompt_mask)

        # if keep_shape: # BNC -> BCHW
        #     B, N, C = prompt_feats.shape
        #     H = self.target_length // self.visual_prompts_encoder.patch_size[0]
        #     W = self.target_length // self.visual_prompts_encoder.patch_size[1]
        #     assert N == H*W
        #     prompt_feats = prompt_feats.transpose(1,2).contiguous()
        #     prompt_feats = prompt_feats.reshape(B, C, H, W)
        #visualize_tensor(prompt_mask)
        return prompt_mask

    def forward(self, image_feats, coord_feats):
        #x = self.fusion(image_feats, prompt_feats)
        scaled_preds, class_preds = self.forward_ori(image_feats, coord_feats)
        class_preds_lst = [class_preds]
        return [scaled_preds, class_preds_lst] if self.training else scaled_preds


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        DecoderBlock = eval(self.config.dec_blk)
        LateralBlock = eval(self.config.lat_blk)
        if self.config.dec_ipt:
            self.split = self.config.dec_ipt_split
            N_dec_ipt = 64
            DBlock = SimpleConvs
            ic = 64
            ipt_cha_opt = 1
            self.ipt_blk5 = DBlock(2**10*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk4 = DBlock(2**8*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk3 = DBlock(2**6*3 if self.split else 3, [N_dec_ipt, channels[1]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk2 = DBlock(2**4*3 if self.split else 3, [N_dec_ipt, channels[2]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk1 = DBlock(2**0*3 if self.split else 3, [N_dec_ipt, channels[3]//8][ipt_cha_opt], inter_channels=ic)
        else:
            self.split = None

        self.decoder_block4 = DecoderBlock(channels[0]+([N_dec_ipt, channels[0]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[1])
        self.decoder_block3 = DecoderBlock(channels[1]+([N_dec_ipt, channels[0]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[2])
        self.decoder_block2 = DecoderBlock(channels[2]+([N_dec_ipt, channels[1]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3])
        self.decoder_block1 = DecoderBlock(channels[3]+([N_dec_ipt, channels[2]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3]//2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3]//2+([N_dec_ipt, channels[3]//8][ipt_cha_opt] if self.config.dec_ipt else 0), 1, 1, 1, 0))

        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)

            if self.config.out_ref:
                _N = 16
                self.gdt_convs_4 = nn.Sequential(nn.Conv2d(channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))
                self.gdt_convs_3 = nn.Sequential(nn.Conv2d(channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))
                self.gdt_convs_2 = nn.Sequential(nn.Conv2d(channels[3], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))

                self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

                self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    def get_patches_batch(self, x, p):
        _size_h, _size_w = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = torch.split(x[idx], split_size_or_sections=_size_w, dim=-1)
            patches_x = []
            for column_x in columns_x:
                patches_x += [p.unsqueeze(0) for p in torch.split(column_x, split_size_or_sections=_size_h, dim=-2)]
            patch_sample = torch.cat(patches_x, dim=1)
            patches_batch.append(patch_sample)
        return torch.cat(patches_batch, dim=0)

    def forward(self, features):
        if self.training and self.config.out_ref:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features
        outs = []
        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, x4) if self.split else x
            x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(patches_batch, size=x4.shape[2:], mode='bilinear', align_corners=True))), 1)
        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p4_gdt = self.gdt_convs_4(p4)
            if self.training:
                # >> GT:
                m4_dia = m4
                gdt_label_main_4 = gdt_gt * F.interpolate(m4_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_4)
                # >> Pred:
                gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
                outs_gdt_pred.append(gdt_pred_4)
            gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
            # >> Finally:
            p4 = p4 * gdt_attn_4
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p3) if self.split else x
            _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=True))), 1)
        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                # >> GT:
                # m3 --dilation--> m3_dia
                # G_3^gt * m3_dia --> G_3^m, which is the label of gradient
                m3_dia = m3
                gdt_label_main_3 = gdt_gt * F.interpolate(m3_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_3)
                # >> Pred:
                # p3 --conv--BN--> F_3^G, where F_3^G predicts the \hat{G_3} with xx
                # F_3^G --sigmoid--> A_3^G
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            # >> Finally:
            # p3 = p3 * A_3^G
            p3 = p3 * gdt_attn_3
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p2) if self.split else x
            _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=True))), 1)
        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.config.ms_supervision and self.training else None
        if self.config.out_ref:
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                # >> GT:
                m2_dia = m2
                gdt_label_main_2 = gdt_gt * F.interpolate(m2_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_2)
                # >> Pred:
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            # >> Finally:
            p2 = p2 * gdt_attn_2
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p1) if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=True))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)

        if self.config.dec_ipt:
            patches_batch = self.get_patches_batch(x, _p1) if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches_batch, size=x.shape[2:], mode='bilinear', align_corners=True))), 1)
        p1_out = self.conv_out1(_p1)

        if self.config.ms_supervision and self.training:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)
        return outs if not (self.config.out_ref and self.training) else ([outs_gdt_pred, outs_gdt_label], outs)


class SimpleConvs(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels=64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))



def visualize_tensor(tensor, title="Image", denormalize=True):
    """
    可视化 PyTorch 张量图像
    参数：
        tensor : torch.Tensor (形状可为 [C, H, W] 或 [B, C, H, W])
        title : 图像标题
        denormalize : 是否反归一化（如果输入做过归一化预处理）
    """
    import matplotlib.pyplot as plt
    # 将张量转换为 numpy 数组
    image = tensor.detach().cpu().numpy()

    # 处理 4D 批次张量 (取第一个样本)
    if image.ndim == 4:
        image = image[0]

    # 调整通道顺序：CHW -> HWC
    image = image.transpose(1, 2, 0)

    # 反归一化（假设使用标准归一化 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
    # if denormalize:
    #     image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    #     image = image.clip(0, 1)  # 确保像素值在 [0,1] 之间

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
