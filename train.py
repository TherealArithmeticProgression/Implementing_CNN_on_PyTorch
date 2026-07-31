from model import Modl
from config import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_the_model(modl:Modl, epochs:int, train_dataload, test_dataload, dev=DEVICE):
  torch.manual_seed(24)
  modl.to(dev)
  loss_fn=nn.CrossEntropyLoss()
  optimizer=optim.Adam(modl.parameters(), lr=LEARNING_RATE)
  for epoch in range(epochs):
    modl.train()
    train_loss=0
    for batch in tqdm(train_dataload, desc=f"Epoch {epoch}"):
      images = batch['img'].to(dev)
      actual_labels=batch['label'].to(dev)
      preds=modl(images)
      loss=loss_fn(preds, actual_labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss+=loss.item()
    train_loss=train_loss/(len(train_dataload))
    modl.eval()
    with torch.inference_mode():
      for batch in tqdm(test_dataload, desc=f"Epoch {epoch}"):
        images = batch['img'].to(dev)
        actual_labels=batch['label'].to(dev)
        test_preds=modl(images)
        test_loss=loss_fn(test_preds, actual_labels)
        test_loss/=len(test_dataload)
