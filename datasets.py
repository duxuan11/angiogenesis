
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import cv2
from config import Config
from utils.utils import path_to_image
from dataset.base import ISDataset
from dataset.sample import DSample
Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()
import numpy as np


class Datasets(ISDataset):
    def __init__(self, dataset_path, is_train=True, split='train'
                 , **kwargs) -> None:
        super(Datasets, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.keep_size = not config.img_size
        self.data_size = config.img_size
        self.is_train = is_train
        self.load_all = config.load_all
        self.device = config.device
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']

                # train set: 44320 images
        dis_train = {
            "name": "DIS5K-TR",
            "im_dir": "DIS5K/train/im",
            "gt_dir": "DIS5K/train/gt"}

        dis_val = {
            "name": "DIS5K-VD",
            "im_dir": "DIS5K/val/im",
            "gt_dir": "DIS5K/val/gt"}

        if split == 'train':
            self.datasets = [dis_train]
        elif split == 'val':
            self.datasets = [dis_val]
        else:
            raise ValueError(f'Undefined split: {split}')

        self.dataset_samples = []
        for idx, dataset in enumerate(self.datasets):
            image_path = self.dataset_path / dataset['im_dir']
            samples = [(x.stem, idx) for x in sorted(image_path.glob('*.jpg'))]
            self.dataset_samples.extend(samples)
        assert len(self.dataset_samples) > 0

        assert len(self.dataset_samples) > 0

    def get_sample(self, index) -> DSample:
        image_name, idx = self.dataset_samples[index]
        image_path = str(self.dataset_path / self.datasets[idx]['im_dir'] / f'{image_name}.jpg')
        mask_path = str(self.dataset_path / self.datasets[idx]['gt_dir'] / f'{image_name}.png')
        _image = path_to_image(image_path, size=config.img_size, color_type='rgb')
        instances_mask = path_to_image(mask_path, size=config.img_size, color_type='gray')
        #instances_mask = np.max(_label.astype(np.int32), axis=1) # 由于我们单层掩码，因此不需要
        instances_mask[instances_mask > 0] = 1

        return DSample(_image, instances_mask, objects_ids=[1], sample_id=index)


def unit_test():
    dataset_train = Datasets(config.data_root_dir, split='train')
    num_samples_train = dataset_train.get_samples_number()

    dataset_val = Datasets(config.data_root_dir, split='val')
    num_samples_val = dataset_val.get_samples_number()

    assert num_samples_train == 28 and num_samples_val == 28


if __name__ == '__main__':
    unit_test()
