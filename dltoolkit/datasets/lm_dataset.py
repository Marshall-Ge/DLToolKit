from PIL.Image import Image
from torch.utils.data import Dataset
from typing import Callable,  Optional
import torch


class LMDataset(Dataset):
    """
    Dataset for language model training.

    Args:
        dataset: dataset for image-text pairs
        tokenizer: tokenizer for text
        max_length: max length of text input
        strategy: training strategy
    """

    def __init__(
        self,
        dataset,
        strategy,
        max_length: Optional[int] = None,
        tokenizer: Optional[Callable] = None,
        num_processors=8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.strategy = strategy

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        self.input_ids = processed_dataset["input_ids"]


    def process_data(self, data):
        token = self.tokenizer(
            data[self.strategy.config.data.text_key],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        # to avoid EOS_token truncation
        token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        return {'input_ids': token["input_ids"][0]}

    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
        }