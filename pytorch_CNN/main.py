import torch
from timeit import default_timer as timer
from model import Modl
from utils import how_long_did_it_take, evaluate_the_model
from data import create_dataset, get_the_dataloader
from config import *
from train import train_the_model


def launch_model():
  #1.
  training_dataset, testing_dataset = create_dataset()
  print(training_dataset)
  print(testing_dataset)
  #2.
  [training_dataloader, testing_dataloader]=get_the_dataloader([training_dataset, testing_dataset])
  #3.
  Model=Modl(Input_features, Hidden_units, Output_features)
  #4.
  start=timer()
  #5.
  train_the_model(Model, EPOCHS, training_dataloader, testing_dataloader)
  #6.
  end=timer()
  how_long_did_it_take(start, end)
  #7.
  Model.eval()
  all_preds=[]
  all_targets=[]
  with torch.inference_mode():
    for batch in testing_dataloader:
      images = batch['img'].to(DEVICE)
      preds = torch.argmax(Model(images), dim=1)

      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(batch['label'].numpy())

    evaluate_the_model(torch.tensor(all_preds), torch.tensor(all_targets))

if '__main__'==__name__:
  launch_model()