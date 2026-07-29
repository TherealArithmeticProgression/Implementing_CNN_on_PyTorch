import torch.nn as nn

class Modl(nn.Module):
  def __init__(self, input_features, hidden_units, output_features):
    super().__init__()
    self.conv=nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    self.fc=nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_units),
        nn.Linear(hidden_units, output_features),
        nn.BatchNorm1d(output_features),
        nn.ReLU(),
        nn.Softmax(dim=1)
    )
  def forward(self, x):
    return self.fc(self.conv(x))