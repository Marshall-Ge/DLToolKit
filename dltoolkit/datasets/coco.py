import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import dltoolkit.utils.logging as logging
import random
from .build import DATASET_REGISTRY
from .transform import *


logger = logging.get_logger(__name__)
@DATASET_REGISTRY.register()
class CocoCaption(Dataset):
    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.DATA.PATH_TO_DATA_DIR
        if self.mode == "train":
            self.meta_data_path = os.path.join(cfg.DATA.PATH_TO_METADATA_DIR, "train2017")
        else:
            self.meta_data_path = os.path.join(cfg.DATA.PATH_TO_METADATA_DIR, "val2017")
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for ImageNet".format(mode)
        logger.info("Constructing ImageNet {}...".format(mode))

        self.transform = build_transforms(cfg, mode)
        self._load_annotations()

    def _load_annotations(self):
        if self.mode == 'train':
            annotations_file = os.path.join(self.data_path, self.cfg.TRAIN_FILE)
        elif self.mode == 'val':
            annotations_file = os.path.join(self.data_path, self.cfg.VAL_FILE)
        else:
            annotations_file = os.path.join(self.data_path, self.cfg.TEST_FILE)

        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        self.id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

        self.id_to_captions = {}
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            if img_id not in self.id_to_captions:
                self.id_to_captions[img_id] = []
            self.id_to_captions[img_id].append(caption)

        self.index_pairs = []
        for img_id in self.id_to_captions:
            num_captions = len(self.id_to_captions[img_id])
            for cap_idx in range(num_captions):
                self.index_pairs.append((img_id, cap_idx))

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        img_id, cap_idx = self.index_pairs[idx]
        caption = self.id_to_captions[img_id][cap_idx]

        if self.cfg.TASK == 'itm':
            if random.random() < self.cfg.DATA.NEGATIVE_SAMPLING_RATE:
                negative_img_id = random.choice(list(self.id_to_filename.keys()))
                while negative_img_id == img_id:
                    negative_img_id = random.choice(list(self.id_to_filename.keys()))
                negative_img_name = os.path.join(self.meta_data_path, self.id_to_filename[negative_img_id])
                negative_image = Image.open(negative_img_name).convert('RGB')
                negative_image = self.transform(negative_image)
                return (negative_image, caption), 0
            else:
                img_name = os.path.join(self.meta_data_path, self.id_to_filename[img_id])
                image = Image.open(img_name).convert('RGB')
                image = self.transform(image)
                return (image, caption), 1
        else:
            img_name = os.path.join(self.meta_data_path, self.id_to_filename[img_id])
            image = Image.open(img_name).convert('RGB')
            image = self.transform(image)
        return image, caption


