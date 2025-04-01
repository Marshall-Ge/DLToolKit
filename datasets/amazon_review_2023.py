import kagglehub
import os
from datasets import load_dataset

DATA_PATH = './data/amazon-reviews-data-2023'

def download_dataset():
  path = kagglehub.dataset_download("wajahat1064/amazon-reviews-data-2023")
  print("Path to dataset files:", path)

def load_user_review():

  dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
  print(dataset["full"][0])

if __name__ == "__main__":
  if not os.path.exists(DATA_PATH):
    print("Dataset not found, downloading...")
    download_dataset()
  load_user_review()