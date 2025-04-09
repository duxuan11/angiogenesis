from albumentations import *
from dataset.transforms import *
from datasets import Datasets
from dataset.points_sampler import MultiPointSampler
from config import Config
from torch.utils.data import DataLoader
from utils.distributed import get_sampler
config = Config()



def init_data_loader():
    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0,
            rotate_limit=(-3, 3),
            border_mode=0,
            p=0.75
        ),
        RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(-0.15, 0.4),
            p=0.75
        ),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ResizeLongestSide(target_length=config.img_size),
        PadIfNeeded(
            min_height=config.img_size[0],
            min_width=config.img_size[1],
            border_mode=0,
            position='top_left',
    ),
    ], p=1.0)

    val_augmentator = Compose([
        ResizeLongestSide(target_length=config.img_size),
        PadIfNeeded(
            min_height=config.img_size[0],
            min_width=config.img_size[1],
            border_mode=0,
            position='top_left',
        ),
    ], p=1.0)

    points_sampler = MultiPointSampler(
        config.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2
    )

    trainset=Datasets(
        config.data_root_dir,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=-1,
    )

    valset=Datasets(
        config.data_root_dir,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=-1
    )
    return trainset, valset

def init_dataloader(distributed):

    trainset, valset = init_data_loader()
    train_data = DataLoader(
        trainset, config.batch_size,
        sampler=get_sampler(trainset, shuffle=True, distributed=distributed),
        drop_last=True, pin_memory=True,
        num_workers=config.workers
    )

    val_data = DataLoader(
        valset, config.batch_size,
        sampler=get_sampler(valset, shuffle=False, distributed=distributed),
        drop_last=True, pin_memory=True,
        num_workers=config.workers
    )
    return train_data, val_data


if __name__ == "__main__":
    tran, val = init_data_loader()
    print(len(tran))
