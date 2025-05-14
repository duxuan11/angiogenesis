
from isegm.utils.exp_imports.default import *

MODEL_NAME = 'plainvit_base1024_finue_angiogenesis'
from isegm.data.datasets.angiogenesis import AngiogenesisDatasets

def main(cfg):
    model = build_model(img_size=512)
    train(model, cfg)


def build_model(img_size) -> PlainVitModel:
    backbone_params = dict(
        img_size=(img_size, img_size),
        patch_size=(16,16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        global_atten_freq=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(in_dim = 768, out_dims = [128, 256, 512, 1024],)

    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_select_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        out_channels=256,
    )

    fusion_params = dict(
        type='self_attention',
        depth=2,
        params=dict(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,)
    )

    model = PlainVitModel(
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        fusion_params=fusion_params,
        use_disks=True,
        norm_radius=5,
    )

    return model


def train(model: PlainVitModel, cfg) -> None:
    cfg.img_size = model.backbone.patch_embed.img_size[0]
    cfg.val_batch_size = cfg.batch_size
    cfg.num_max_points = 24
    cfg.num_max_next_points = 3

    # initialize the model
    model.backbone.init_weights_from_pretrained(cfg.MAE_WEIGHTS.VIT_BASE)
    model.to(cfg.device)

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    cfg.loss_cfg = loss_cfg

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(p=0.3),
        VerticalFlip(p=0.3),
        RandomResizedCrop(
        size=(cfg.img_size, cfg.img_size),
        scale=(0.2, 1.0),
        ratio=(0.9, 1.1),
        p=0.3,),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0,
            rotate_limit=(-3, 3),
            border_mode=0,
            p=0.75
        ),
        GaussianBlur(blur_limit=(3, 7), sigma_limit=0.5, p=0.5),
        CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(-0.15, 0.4),
            p=0.75
        ),
        ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ToGray(
        num_output_channels=3,  # 输出三通道
        p=0.2  # 20%概率应用灰度化
        ),
        ResizeLongestSide(target_length=cfg.img_size),
        PadIfNeeded(
            min_height=cfg.img_size,
            min_width=cfg.img_size,
            border_mode=0,
            position='top_left',
        ),
        Resize(height=1024, width=1024, p=1.0),
    ], p=1.0)

    val_augmentator = Compose([
        ResizeLongestSide(target_length=cfg.img_size),
        PadIfNeeded(
            min_height=cfg.img_size,
            min_width=cfg.img_size,
            border_mode=0,
            position='top_left',
        ),
    ], p=1.0)

    points_sampler = MultiPointSampler(
        cfg.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2
    )

    trainset = AngiogenesisDatasets(
        cfg.Angiogenesis_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        keep_background_prob=0.05,
        epoch_len=-1,
    )

    valset = AngiogenesisDatasets(
        cfg.Angiogenesis_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=-1
    )

    optimizer_params = {'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8}
    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 90], gamma=0.1
    )
    trainer = ISTrainer(
        model,
        cfg,
        trainset,
        valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 10), (90, 1)],
        image_dump_interval=500,
        metrics=[AdaptiveIoU()],
        max_interactive_points=cfg.num_max_points,
        max_num_next_clicks=cfg.num_max_next_points
    )
    trainer.run(num_epochs=100, validation=True)
