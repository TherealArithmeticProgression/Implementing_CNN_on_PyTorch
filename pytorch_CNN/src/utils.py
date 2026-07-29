from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .config import DEVICE

def how_long_did_it_take(start:float, end:float, device=DEVICE):
  total_time=end-start
  print(f"| Device: {device} | Time taken for model training: {total_time} |")

def evaluate_the_model(preds, test_labels, dev=DEVICE):
  y_true = test_labels.cpu().numpy() if hasattr(test_labels, 'cpu') else test_labels
  y_pred = preds.cpu().numpy() if hasattr(preds, 'cpu') else preds
  print("\n Classification Report: \n")
  print(classification_report(y_true, y_pred))
  print("\n Confusion Matrix: \n")
  cm = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(cmap=plt.cm.Blues)
  plt.title("CIFAR-10 Confusion Matrix")
  plt.show()