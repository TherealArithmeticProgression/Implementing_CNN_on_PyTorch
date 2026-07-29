#Hyperparameters
import torch

EPOCHS=50
LEARNING_RATE=0.01
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
Input_features=2048
Hidden_units=128
Output_features=10