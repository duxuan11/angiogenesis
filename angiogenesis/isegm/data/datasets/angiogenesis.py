
from pickle import FALSE
from albumentations.augmentations.utils import F
from tqdm import tqdm

from pathlib import Path
import cv2
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import numpy as np
from skimage.measure import label, regionprops
import random
import matplotlib.pyplot as plt

class AngiogenesisDatasets(ISDataset):
    def __init__(self, dataset_path, is_train=True, split='train'
                 , **kwargs) -> None:
        super(AngiogenesisDatasets, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.data_size = (512,512)
        self.is_train = is_train
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']

        # train set: 44320 images
        bri_other_train = {
            "name": "brightfield",
            "im_dir": "brightfield/train/im",
            "gt_dir": "brightfield/train/gt"}

        bri_train = {
            "name": "brightfield",
            "im_dir": "brightfield/train/im",
            "gt_dir": "brightfield/train/gt"}

        flu_train = {
            "name": "Fluorescence",
            "im_dir": "Fluorescence/train/im",
            "gt_dir": "Fluorescence/train/gt"}

        pascalvoc_train = {
            "name": "PascalVOC",
            "im_dir": "PascalVOC/train/JPEGImages",
            "gt_dir": "PascalVOC/train/SegmentationObject"}

        dis_train = {
            "name": "DIS5K",
            "im_dir": "DIS5K/train/im",
            "gt_dir": "DIS5K/train/gt"}

        thin_train = {
            "name": "THIN",
            "im_dir": "THIN/train/im",
            "gt_dir": "THIN/train/gt"}

        cascade_psp_train = {
            "name": "cascade_psp",
            "im_dir": "cascade_psp/train/im",
            "gt_dir": "cascade_psp/train/gt"}

        cascade_psp_train = {
            "name": "cascade_psp",
            "im_dir": "cascade_psp/train/im",
            "gt_dir": "cascade_psp/train/gt"}

        thin_val = {
            "name": "THIN-VD",
            "im_dir": "THIN/val/im",
            "gt_dir": "THIN/val/gt"}
        DIS5K_val = {
            "name": "DIS5K",
            "im_dir": "DIS5K/val/im",
            "gt_dir": "DIS5K/val/gt"}

        if split == 'train':
            self.datasets = [bri_other_train, bri_train, flu_train, pascalvoc_train, dis_train, thin_train, cascade_psp_train]
        elif split == 'val':
            self.datasets = [thin_val, DIS5K_val]
        else:
            raise ValueError(f'Undefined split: {split}')

        self.dataset_samples = []
        for idx, dataset in enumerate(self.datasets):
            image_path = self.dataset_path / dataset['im_dir']
            samples = [(x.stem, idx) for x in sorted(image_path.glob('*.jpg'))]
            self.dataset_samples.extend(samples)
        assert len(self.dataset_samples) > 0

    def get_sample(self, index) -> DSample:
        image_name, idx = self.dataset_samples[index]
        image_path = str(self.dataset_path / self.datasets[idx]['im_dir'] / f'{image_name}.jpg')
        if self.datasets[idx]["name"] == "Fluorescence":
            mask_path = str(self.dataset_path / self.datasets[idx]['gt_dir'] / f'{image_name}_mask.png')
        elif self.datasets[idx]["name"] == "birghtfield":
            mask_path = str(self.dataset_path / self.datasets[idx]['gt_dir'] / f'{image_name}_mask.png')
        else:
            mask_path = str(self.dataset_path / self.datasets[idx]['gt_dir'] / f'{image_name}.png')
        _image = path_to_image(image_path, size=self.data_size, color_type='rgb')
        instances_mask = path_to_image(mask_path, size=self.data_size, color_type='gray')
        #instances_mask = np.max(_label.astype(np.int32), axis=1) # 由于我们单层掩码，因此不需要


        if random.random() < 0.3 and self.datasets[idx]["name"] == "Fluorescence" or "birghtfield":
            instances_mask[instances_mask > 0] = 1
            background_mask = select_background_region(instances_mask)
            objects_ids = np.unique(background_mask)
            objects_ids = [x for x in objects_ids if x != 0 and x != 220]
            if len(objects_ids) == 0:
                objects_ids = [1]
                background_mask = instances_mask
            return DSample(_image, background_mask, objects_ids=objects_ids, sample_id=index)
        else:
            if np.unique(instances_mask) > 2:
                objects_ids = np.unique(instances_mask)
                objects_ids = [x for x in objects_ids if x != 0 and x != 220]

                return DSample(_image, instances_mask, objects_ids=objects_ids, ignore_ids=[220], sample_id=index)
            else:
                instances_mask[instances_mask > 0] = 1
            # plt.figure(figsize=(6,6))
            # plt.imshow(background_mask, cmap='gray')
            # plt.title("Binary Mask")
            # plt.axis("off")
            # plt.show()
                return DSample(_image, instances_mask, objects_ids=[1], sample_id=index)


def path_to_image(path, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if size and False:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    # if color_type.lower() == 'rgb':
    #     image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    # else:
    #     image = Image.fromarray(image).convert('L')
    return image


def select_background_region(binary_mask):
    inverted_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    inverted_mask[binary_mask == 1] = 0
    inverted_mask[binary_mask == 0] = 1

    # 2. 连通域标记与统计

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    inverted_mask, connectivity=8, ltype=cv2.CV_32S
    )
    # 3. 创建结果掩码并筛选大区域
    current_label = 1
    result_mask = np.zeros_like(inverted_mask, dtype=np.uint8)
    for label in range(1, num_labels):
        if label >= 255:
            break
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 2000:
            result_mask[labels == label] = current_label
            current_label += 1
    return result_mask
    # plt.figure(figsize=(6,6))
    # plt.imshow(result_mask, cmap='gray')
    # plt.title("Binary Mask")
    # plt.axis("off")
    # plt.show()
