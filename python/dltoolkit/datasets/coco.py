import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import dltoolkit.utils.logging as logging
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class COCOCaptionDataset(Dataset):
    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.DATA.PATH_TO_DATA_DIR
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for ImageNet".format(mode)
        logger.info("Constructing ImageNet {}...".format(mode))

        self._load_annotations()

        # 创建图像ID到文件名的映射
        self.id_to_filename = {img['id']: img['file_name'] for img in self.annotations['images']}

        # 创建图像ID到描述的映射
        self.id_to_captions = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            if img_id not in self.id_to_captions:
                self.id_to_captions[img_id] = []
            self.id_to_captions[img_id].append(caption)

        # 创建数据集索引：(图像ID, 描述索引)
        self.index_pairs = []
        for img_id in self.id_to_captions:
            num_captions = len(self.id_to_captions[img_id])
            for cap_idx in range(num_captions):
                self.index_pairs.append((img_id, cap_idx))
    def _load_annotations(self):
        if self.mode == 'train':
            annotations_file = os.path.join(self.data_path, self.cfg.TRAIN_FILE)
        elif self.mode == 'val':
            annotations_file = os.path.join(self.data_path, self.cfg.VAL_FILE)
        else:
            annotations_file = os.path.join(self.data_path, self.cfg.TEST_FILE)

        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        # 获取图像ID和描述索引
        img_id, cap_idx = self.index_pairs[idx]

        # 加载图像
        img_name = os.path.join(self.root_dir, self.id_to_filename[img_id])
        image = Image.open(img_name).convert('RGB')


        # 获取描述
        caption = self.id_to_captions[img_id][cap_idx]

        return image, caption


def get_coco_dataloader(root_dir, annotation_file, batch_size=32, shuffle=True, num_workers=4):
    """
    创建COCO Caption数据集的数据加载器

    参数:
    root_dir (string): 图像所在目录
    annotation_file (string): 标注文件路径
    batch_size (int, optional): 批次大小
    shuffle (bool, optional): 是否打乱数据
    num_workers (int, optional): 加载数据的进程数

    返回:
    DataLoader: COCO Caption数据集的数据加载器
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集
    dataset = COCOCaptionDataset(root_dir, annotation_file, transform=transform)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return dataloader


def collate_fn(data):
    """
    自定义整理函数，用于处理不同长度的描述

    参数:
    data: 元组列表 (图像, 描述)

    返回:
    图像列表和描述列表
    """
    images, captions = zip(*data)
    return torch.stack(images, 0), captions


# 使用示例
if __name__ == "__main__":
    # 设置数据集路径
    root_dir = '/path/to/coco/images/train2017'  # 替换为实际路径
    annotation_file = '/path/to/coco/annotations/captions_train2017.json'  # 替换为实际路径

    # 创建数据加载器
    dataloader = get_coco_dataloader(root_dir, annotation_file, batch_size=8)

    # 测试数据加载器
    for images, captions in dataloader:
        print(f"图像批次形状: {images.shape}")
        print(f"第一批描述: {captions[0]}")
        break