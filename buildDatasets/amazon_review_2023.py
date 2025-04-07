import time
import urllib.request
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import os
import base64
import cv2
from PIL import Image
from models.clip import create_pretrained_clip

class Preprocess:
    def __init__(self, dataset_path, sample_number, sampled_dataset_path):
        self.dataset_path = dataset_path
        self.sample_number = sample_number
        self.sampled_dataset_path = os.path.abspath(sampled_dataset_path)
        self.dataset = self.sample_dataset()
        self.num_items = None
        self.transform_dataset()

    def sample_dataset(self):
        path = "./data/amazon-reviews-data-2023/data_origin"
        path = os.path.abspath(path)
        ds = datasets.load_from_disk(path)
        ds = ds.select(range(self.sample_number)) if self.sample_number is not None else ds
        return ds

    def download_image(self, url, save_path):
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(f"Error downloading image: {e}")
            return

    def transform_dataset(self):

        def get_unique_item_id():
            unique_ids = list(set(self.dataset['parent_asin']))
            id_mapping = {asin: idx for idx, asin in enumerate(unique_ids)}
            with open(
                    '/Users/fege/Desktop/project/MultimodalToolBox/data/amazon-reviews-data-2023/data/parent_asin_mapping.txt',
                    'w') as f:
                for asin, idx in id_mapping.items():
                    f.write(f"{asin} {idx}\n")
            return id_mapping

        item_id_mapping = get_unique_item_id()
        self.num_items = len(item_id_mapping)
        def transform_item(item):
            item_id = item_id_mapping[item['parent_asin']]
            r_id = []
            t = []

            if item['title']:
                r_id.append('title')
                t.append(item['title'])

            if item['description'] and item['description'] not in ['None', '']:
                r_id.extend(['description'] * len(item['description']))
                t.extend(item['description'])

            if item['images']['large'] and item['images']['large'] not in ['None', '']:
                r_id.extend(['has_image'] * len(item['images']['large']))
                t.extend([e.split('/')[-1] for e in item['images']['large']])

            return {
                'item_id': [item_id] * len(r_id),
                'r_id': r_id,
                't': t
            }

        new_ds = {'item_id': [], 'r_id': [], 't': []}
        for idx in range(self.dataset.num_rows):
            t = transform_item(self.dataset[idx])
            new_ds['item_id'].extend(t['item_id'])
            new_ds['r_id'].extend(t['r_id'])
            new_ds['t'].extend(t['t'])
        new_ds = datasets.Dataset.from_dict(new_ds)
        new_ds.save_to_disk(self.sampled_dataset_path)


#TODO: Check the speed
class AmazonItemDataset(Dataset):
    def __init__(self, path, device = 'cpu', clip_arch = 'ViT-B/32'):
        self.dataset_path = os.path.abspath(path)
        self.img_path = os.path.join(self.dataset_path, "images")
        self.dataset = datasets.load_from_disk(self.dataset_path)
        self.r_id_map = {'title': 0, 'description': 1, 'has_image':2}
        self.clip_modal, self.image_preprocess, self.tokenizer = create_pretrained_clip(clip_arch, device=device)
        self.device = device

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, idx):
        item_id = torch.tensor(self.dataset['item_id'][idx]).to(self.device)
        r_id = torch.tensor(self.r_id_map[self.dataset['r_id'][idx]])
        if self.dataset['r_id'][idx] == 'has_image':
            image_path = os.path.join(self.img_path, str(self.dataset['t'][idx]))
            preprocessed_img = self.image_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                t = self.clip_modal.encode_image(preprocessed_img).float()
                t /= t.norm(dim=-1, keepdim=True)
        else:
            text = self.dataset['t'][idx]
            text = self.tokenizer(text).to(self.device)
            with torch.no_grad():
                t = self.clip_modal.encode_text(text).float()
                t /= t.norm(dim=-1, keepdim=True)
        t = t.squeeze(0)
        return {'item_id': item_id, 'r_id': r_id, 't': t}

def data_preprocess(path, sample_number):
    origin_path = path + '_origin'
    preprocessor = Preprocess(origin_path, sample_number, path)
    return

def build_amazon_dataloader(path, batch_size, device, need_preprocess=False, sample_number = None):
    if need_preprocess:
        data_preprocess(path, sample_number)
    # TODO: enhance for dataset split
    item_dataset = AmazonItemDataset(path, device)
    item_dataloader = DataLoader(item_dataset, batch_size=batch_size)
    return item_dataloader
