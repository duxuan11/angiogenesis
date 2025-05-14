import torch
import numpy
import argparse
from isegm.utils import exp
from isegm.inference import utils
import torch.nn.functional as F
import torch.nn as nn
from isegm.inference import clicker
import onnxruntime as rt

class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from isegm.utils.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i], rows, cols,
                                                  norm_delimeter))
                                # coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                #                   norm_delimeter))
            #coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
            coords = np.stack(coords, axis=0)
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2)
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, img_size, coords):
        return self.get_coord_features(coords, img_size[0], img_size[2], img_size[3])


class ImageEncoder(nn.Module):
    def __init__(self, plainVitModel) -> None:
        super().__init__()
        self.model = plainVitModel
        self.image_encoder = plainVitModel.backbone

    def forward(self, x, keep_shape=True):
        image_feats, addition_feats = self.image_encoder(x, keep_shape=keep_shape)
        if addition_feats is not None:
            image_feats = image_feats + addition_feats[0]
        return image_feats

class ImageDecoder(nn.Module):
    def __init__(self, plainVitModel) -> None:
        super().__init__()
        self.model = plainVitModel
        self.target_length = self.model.backbone.patch_embed.img_size[0]
        self.target_size = (1024, 1024)
        self.neck = self.model.neck
        self.head = self.model.head
        self.dist_maps = self.model.dist_maps
        self.visual_prompts_encoder = self.model.visual_prompts_encoder

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize the input tensor to the original size

        Arguments:
            x: tensor of shape [B, C, H, W]
        """
        # resize
        x = F.interpolate(x, self.target_size, mode='bilinear', align_corners=False)
        return x

    def get_prompt_feats(self, prompt_mask, keep_shape=True):
        """Get feature representation for prompts

        Arguments:
            image_shape: original image shape
            prompts: a dictionary containing all possible prompts

        Returns:
            prompt features
        """

        prompt_feats = self.visual_prompts_encoder(prompt_mask)

        if keep_shape: # BNC -> BCHW
            B, N, C = prompt_feats.shape
            H = self.target_length // self.visual_prompts_encoder.patch_size[0]
            W = self.target_length // self.visual_prompts_encoder.patch_size[1]
            assert N == H*W
            prompt_feats = prompt_feats.transpose(1,2).contiguous()
            prompt_feats = prompt_feats.reshape(B, C, H, W)

        return prompt_feats

    def forward(self, image_feats, prompt_feat):
        prompt_feats = self.get_prompt_feats(prompt_feat)
        fused_features = self.model.fusion(image_feats, prompt_feats)
        pyramid_features = self.neck(fused_features)
        seg_prob = self.head(pyramid_features)
        seg_prob = F.interpolate(
            seg_prob,
            size=self.target_size,
            mode='bilinear',
            align_corners=True
        )

        seg_prob = self.postprocess(seg_prob)
        return seg_prob

def get_prompt_feats(points_nd, prev_mask):
            # resize the previous mask to the target length
        points = points_nd

        prev_mask = prev_mask
        orig_size = prev_mask.shape[-2:]
        target_length = 1024
        # prev_mask = F.interpolate(
        #     prev_mask, target_length, mode='bilinear', align_corners=False)

        # convert points from original coordinates to resized coordinates

        for batch_id in range(len(points)):
            for point_id in range(len(points[batch_id])):
                if points[batch_id, point_id, 2] > -1: # pos. or neg. points
                    w, h = points[batch_id, point_id, 0], points[batch_id, point_id, 1]
                    w = int(w * (target_length / orig_size[0]) + 0.5)
                    h = int(h * (target_length / orig_size[1]) + 0.5)
                    points[batch_id, point_id, 0], points[batch_id, point_id, 1] = w, h

        return prev_mask, points

def get_prompt_feats_2(points_nd, prev_mask):
            # resize the previous mask to the target length
        points = points_nd

        prev_mask = prev_mask
        orig_size = prev_mask.shape[-2:]
        target_length = 1024
        prev_mask = F.interpolate(
            prev_mask, target_length, mode='bilinear', align_corners=False)

        # convert points from original coordinates to resized coordinates

        for batch_id in range(len(points)):
            for point_id in range(len(points[batch_id])):
                if points[batch_id, point_id, 2] > -1: # pos. or neg. points
                    w, h = points[batch_id, point_id, 0], points[batch_id, point_id, 1]
                    w = int(w * (target_length / orig_size[0]) + 0.5)
                    h = int(h * (target_length / orig_size[1]) + 0.5)
                    points[batch_id, point_id, 0], points[batch_id, point_id, 1] = w, h

        return prev_mask, points

import numpy as np
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', default='coco_lvis_h18_itermask', type=str, required=False,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')
    parser.add_argument('--image', type=str, default='datasets/1111.png',
                        help='path to image')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')
    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')
    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    args = parser.parse_args()
    # parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


def main():
    args, cfg = parse_args()
    torch.backends.cudnn.deterministic = True
    # checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    # model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)

    checkpoint_path = "weights\\vitb_sa1_cocolvis_epoch_90.pth"
    model = utils.load_is_model(checkpoint_path, "cuda", cpu_dist_maps=True)
    point_nd = torch.tensor([[(200, 200, 1),
                            [-1, -1, -1]],
                            [(250, 250, 0),
                            [-1, -1, -1]]], device='cuda')  # tensor(2,2,3)

    print(point_nd.shape)
    image = torch.randn(1, 3, 1024, 1024, device='cuda')
    image_encoder = ImageEncoder(model)
    image_decoder = ImageDecoder(model)
    prev_mask = torch.zeros_like(image[:, :1, :, :])
    clicks_list = [(100,100,1),(200,200,0)]
    points_nd = get_points(clicks_list)
    clickers = clicker.Clicker()
    click = clicker.Click(is_positive=True, coords=(100, 200))
    clickers.add_click(click)

    # torch.onnx.export(image_encoder,
    #                 image,
    #                 f"vit_encoder.onnx",
    #                 export_params=True,
    #                 opset_version=17,
    #                 do_constant_folding=True,
    #                 input_names=['image'],
    #                 output_names=['image_feats']
    #                 )
    clicks_list = clickers.get_clicks()
    points_nd = get_points_nd([clicks_list])
    image_feats = image_encoder(image)
    dist_map = model.dist_maps
    prev_mask,points = get_prompt_feats(points_nd, prev_mask)
    point_mask = dist_map(prev_mask.shape, points)
    prompt_feat = torch.cat((prev_mask, point_mask), dim=1)

    #prompts = {'points': points_nd, 'prev_mask': prev_mask}
    torch.onnx.export(
        image_decoder,
        {"image_feats": image_feats, 'prompt_feat':prompt_feat},  # 字典形式传递
        "vit_decoder.onnx",
        export_params=True,
        verbose=False,
        opset_version=17,
        do_constant_folding=False,
        input_names=["image_feats", "prompt_feat"],
        output_names=["output"],
    )

def apply_transforms(transforms, image_nd, clicks_lists):
    is_image_changed = False
    for t in transforms:
        image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
        is_image_changed |= t.image_changed
    return image_nd, clicks_lists, is_image_changed

def get_points(clicks_lists):
    total_clicks = []
    net_clicks_limit = None
    num_pos_clicks = sum(1 for t in clicks_lists if t[2] == 1)
    num_neg_clicks = sum(1 for t in clicks_lists if t[2] != 1 and t[2] != -1)
    num_max_points = max(num_pos_clicks + num_neg_clicks, 1)
    if net_clicks_limit is not None:
        num_max_points = min(net_clicks_limit, num_max_points)
    num_max_points = max(1, num_max_points)
    pos_clicks = [clicks_and_indx for clicks_and_indx in clicks_lists if clicks_and_indx[2] == 1]
    pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]
    neg_clicks = [clicks_and_indx for clicks_and_indx in clicks_lists if clicks_and_indx[2] != 1 and clicks_and_indx[2] != -1]
    neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
    total_clicks.append(pos_clicks + neg_clicks)

    return total_clicks


def get_points_nd(clicks_lists):
    total_clicks = []
    net_clicks_limit = None

    num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
    num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
    num_max_points = max(num_pos_clicks + num_neg_clicks)
    if net_clicks_limit is not None:
        num_max_points = min(net_clicks_limit, num_max_points)
    num_max_points = max(1, num_max_points)

    for clicks_list in clicks_lists:
        clicks_list = clicks_list[:net_clicks_limit]
        pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

        neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)

    #return torch.tensor(total_clicks,dtype=torch.float32, device="cuda")
    return np.array(total_clicks, dtype=np.float32)


def preprocess_input(image):
    image /= 255.0
    return image


def postprocess_output(pred, thres=0.45):
    pred = np.transpose(pred, (1, 2, 0))
    pred = np.squeeze(pred, axis=2)
    pred[pred > thres] = 1
    return pred


def draw_and_blend_image(pred, image, alpha=0.5):
    img_to_draw = np.copy(image).astype(np.uint8)
    img_to_draw[pred == 1] = (242, 0, 0)
    image = np.uint8(image)
    image = image * (1 - alpha) + img_to_draw * alpha
    return image



def test_onnx_model():
    import numpy as np
    from PIL import Image
    import time

    '''定义处理函数'''
    '''输入图像'''
    img = np.array(Image.open("F:\\figure_three\\1.tif").resize((1024,1024)))  # image_501_501.jpg
    image = np.transpose(img, (2, 0, 1))
    image = preprocess_input(image.astype(np.float32))
    image = image[np.newaxis, :]
    mask = np.zeros_like(image[:, :1, :, :])

    '''输入点击'''
    clickers = clicker.Clicker()
    click = clicker.Click(is_positive=True, coords=(100, 200))
    clickers.add_click(click)
    clicks_list = clickers.get_clicks()
    points_nd = get_points_nd([clicks_list])


    '''点击交互式模型部署与推理 部署ONNX Runtime 这个推理引擎上'''
    encoder = rt.InferenceSession('vit_encoder.onnx',providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    output_name = encoder.get_outputs()[0].name
    start = time.time()
    image_feats = \
    encoder.run([output_name], {"image": image.astype(np.float32)})[0]
    end = time.time()
    print(end - start)
    input = np.concatenate([image, mask], axis=1)
    # sess_options = rt.SessionOptions()
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # 启用图优化

    # 指定 GPU 执行提供者
    # providers = [('CUDAExecutionProvider', {
    #     'device_id': 0,          # GPU 设备编号
    #     'arena_extend_strategy': 'kNextPowerOfTwo',  # 内存分配策略
    #     'gpu_mem_limit': 2048    # GPU 内存限制（MB）
    # })]
    decoder = rt.InferenceSession('vit_decoder.onnx',providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    output_name = decoder.get_outputs()[0].name
    prompts = {'points': points_nd, 'prev_mask': mask}
    dist_map = DistMaps(cpu_mode=True,use_disks=True,norm_radius=5)

    prev_mask,points = get_prompt_feats(points_nd, mask)
    start2 = time.time()
    point_mask = dist_map(prev_mask.shape, points)
    prompt_feat = np.concatenate((prev_mask, point_mask), axis=1)

    prediction = \
    decoder.run([output_name], {"image_feats":image_feats.astype(np.float32),'prompt_feat':prompt_feat})[0]
    end1 = time.time()
    print(end1 - start2)
    start3 = time.time()
    prediction = \
    decoder.run([output_name], {"image_feats":image_feats.astype(np.float32),'prompt_feat':prompt_feat})[0]
    end3 = time.time()
    print(end3 - start3)
    '''后处理'''
    pred = postprocess_output(prediction[0], thres=0.45)
    '''绘制和显示图像'''
    Image.fromarray(np.uint8(draw_and_blend_image(pred, img))).show()
    # print("outputs:")
    # print(prediction[0][0])

    input_name = decoder.get_inputs()[0].name
    input_name = decoder.get_inputs()[1].name
    print(input_name)

'''点击交互式模型部署部分测试'''
if __name__ == '__main__':
    #main()
    test_onnx_model()
