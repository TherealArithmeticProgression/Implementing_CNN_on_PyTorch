#DATASET

import os
from huggingface_hub import login
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
def create_dataset():
  hf_token=os.getenv("HF_TOKEN")
  if hf_token:
      login(token=hf_token)
  train_image_transforms = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor()
  ])

  test_image_transforms = transforms.Compose([
      transforms.ToTensor()
  ])

  train_subset = load_dataset(
      'uoft-cs/cifar10',
      split='train[:50000]'
  )

  test_subset = load_dataset(
      'uoft-cs/cifar10',
      split='test[:10000]'
  )

  train_subset = train_subset.map(lambda example: {'img': train_image_transforms(example['img']), 'label': example['label']})

  test_subset = test_subset.map(lambda example: {'img': test_image_transforms(example['img']), 'label': example['label']})

  train_subset = train_subset.with_format('torch')
  test_subset = test_subset.with_format('torch')

  print("Loaded the data!")

  return train_subset, test_subset

def get_the_dataloader(datasets:list):
  k=[]
  for dataset in datasets:
    k.append(DataLoader(dataset, num_workers=4, batch_size=32, shuffle=True))
  return k

