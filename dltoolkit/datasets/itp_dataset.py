from PIL.Image import Image
from torch.utils.data import Dataset
from typing import Callable,  Optional
import torch


class ImgTxtPairDataset(Dataset):
    """
    Dataset for image-text pairs.

    Args:
        dataset: dataset for image-text pairs
        tokenizer: tokenizer for text
        transform: transform for images
        max_length: max length of text input
        strategy: training strategy
        input_template: template for text input
    """

    def __init__(
        self,
        dataset,
        strategy,
        max_length: Optional[int] = None,
        input_template: Optional[str] = None,
        tokenizer: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.strategy = strategy
        self.input_template = input_template
        self.images, self.texts = self.prepare_data(dataset)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'text_or_label': self.texts[idx],
        }

    def prepare_data(self, dataset):
        # Process data
        images, texts = [], []
        for item in dataset:
            image = item[self.strategy.config.data.image_key]
            assert isinstance(image, Image), f"Expected PIL Image, but got {type(image)}"
            if self.transform:
                image = self.transform(image)
            else:
                import torchvision.transforms as T
                image = T.ToTensor()(image)
            images.append(image)
            text = item[self.strategy.config.data.text_key]
            if self.input_template:
                text = self.input_template.format(text)
            if self.tokenizer:
                text = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors='pt'
                )
            texts.append(text)
        return images, texts

    def collate_fn(self, batch):
        images, texts = [], []
        for item in batch:
            images.append(item['image'].unsqueeze(0))
            texts.append(item['text_or_label'])
        images = torch.cat(images, dim=0)
        return {
            'image': images,
            'text_or_label': torch.tensor(texts),
        }

if __name__ == '__main__':
    from datasets import load_dataset

    ds = load_dataset("ylecun/mnist", split='test')
    print(ds[0])