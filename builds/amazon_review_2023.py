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
    def __init__(self):
        pass

    def download_image(self, url, save_path):
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(f"Error downloading image: {e}")
            return


    def transform_dataset(self, review):
        pass

    def get_unique_item_id(self):
        pass

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
        return {'item_id': item_id, 'r_id': r_id, 't': t}

def data_preprocess():
    preprocessor = Preprocess()
    return

def build_amazon_dataloader(path, batch_size, device):
    # data_preprocess()
    # TODO: enhance for dataset split
    item_dataset = AmazonItemDataset(path, device)
    item_dataloader = DataLoader(item_dataset, batch_size=batch_size)
    return item_dataloader
