
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch 
from torch import nn
import torchaudio
from torch.nn import functional as Functional

# device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 
MODEL_NAME = "cats-dogs-sound-cnn.pt"
n_input = n_output = 1
model_path = os.path.join(os.getcwd(), f"models/static/{MODEL_NAME}")

# Model Module
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super(M5, self).__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = Functional.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = Functional.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = Functional.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = Functional.relu(self.bn4(x))
        x = self.pool4(x)
        x = Functional.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x


# Model instances


print(" *   LOADING AUDIO CLASSIFICATION MODEL")
model = M5(n_input=n_input,
           n_output=n_output
           ).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print("\n *   DONE LOADING CLASSIFICATION MODELS THE MODEL")



# Transforms
sample_rate = 16000
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                           new_freq=new_sample_rate)

classes = ["cat", "dog"]

def pad_sequence(batch):
    batch = torch.nn.utils.rnn.pad_sequence([batch], batch_first=True, padding_value=0.)
    return batch

def preprocess(waveform):
  waveform = pad_sequence(waveform)
  return transform(waveform)


def predict_label(model, waveform):
  processed = preprocess(waveform).to(device)
  model.eval()
  with torch.no_grad():
    prediction = torch.sigmoid(model(processed).squeeze())
    probability = float(prediction.item()) if prediction.item() > .5 else 1 - prediction.item()
    label = 1 if prediction.item() >= 0.5 else 0
    pred =  {
        'label': label,
        'class': classes[label],
        'probability':round(probability, 2),
    }
    return pred
