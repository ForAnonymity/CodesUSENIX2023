import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
  # Initialization
  def __init__(self):
    super().__init__()

    # Adapter
    self.avgpool = nn.AvgPool2d((2, 3))

    self.conv1 = nn.Conv2d(1, 3, 9)
    self.bn1 = nn.BatchNorm2d(3)
    self.conv2 = nn.Conv2d(3, 3, 9)
    self.bn2 = nn.BatchNorm2d(3)
    self.conv3 = nn.Conv2d(3, 3, 9)
    self.bn3 = nn.BatchNorm2d(3)
    self.conv4 = nn.Conv2d(3, 3, 9)
    self.bn4 = nn.BatchNorm2d(3)

    # Linear Layer
    self.linear1 = nn.Linear(3*224*224, 1000)

    # CV Encoder
    self.conv11 = nn.Conv2d(3, 12, 4)
    self.bn11 = nn.BatchNorm2d(12)
    self.conv12 = nn.Conv2d(12, 24, 4)
    self.bn12 = nn.BatchNorm2d(24)
    self.conv13 = nn.Conv2d(24, 48, 4)
    self.bn13 = nn.BatchNorm2d(48)
    self.conv14 = nn.Conv2d(48, 96, 4)
    self.bn14 = nn.BatchNorm2d(96)

    # NLP Encoder
    self.conv21 = nn.Conv2d(3, 12, 4)
    self.bn21 = nn.BatchNorm2d(12)
    self.conv22 = nn.Conv2d(12, 24, 4)
    self.bn22 = nn.BatchNorm2d(24)
    self.conv23 = nn.Conv2d(24, 48, 4)
    self.bn23 = nn.BatchNorm2d(48)
    self.conv24 = nn.Conv2d(48, 96, 4)
    self.bn24 = nn.BatchNorm2d(96)

    # Decoder
    self.convt1 = nn.ConvTranspose2d(192, 96, 4)
    self.bnt1 = nn.BatchNorm2d(96)
    self.convt2 = nn.ConvTranspose2d(96, 48, 4)
    self.bnt2 = nn.BatchNorm2d(48)
    self.convt3 = nn.ConvTranspose2d(48, 24, 4)
    self.bnt3 = nn.BatchNorm2d(24)
    self.convt4 = nn.ConvTranspose2d(24, 3, 4)
  
  def forward(self, x1, x2):
  
    # CV
    x1 = self.conv11(x1)
    x1 = F.relu(self.bn11(x1))
    x1 = self.conv12(x1)
    x1 = F.relu(self.bn12(x1))
    x1 = self.conv13(x1)
    x1 = F.relu(self.bn13(x1))
    x1 = self.conv14(x1)
    x1 = F.relu(self.bn14(x1))

    # Adapter
    x2 = self.avgpool(x2)
    
    x2 = self.conv1(x2)
    x2 = F.relu(self.bn1(x2))
    x2 = self.conv2(x2)
    x2 = F.relu(self.bn2(x2))
    x2 = self.conv3(x2)
    x2 = F.relu(self.bn3(x2))
    x2 = self.conv4(x2)
    x2 = F.relu(self.bn4(x2))

    # Linear Layer
    temp_x2 = x2.view(x2.size(0), -1)
    temp_x2 = F.relu(self.linear1(temp_x2))
    
    # NLP Encoder
    x2 = self.conv21(x2)
    x2 = F.relu(self.bn21(x2))
    x2 = self.conv22(x2)
    x2 = F.relu(self.bn22(x2))
    x2 = self.conv23(x2)
    x2 = F.relu(self.bn23(x2))
    x2 = self.conv24(x2)
    x2 = F.relu(self.bn24(x2))

    # Concatenate
    x = torch.cat((x1, x2), 1)

    # Decoder
    x = self.convt1(x)
    x = F.relu(self.bnt1(x))
    x = self.convt2(x)
    x = F.relu(self.bnt2(x))
    x = self.convt3(x)
    x = F.relu(self.bnt3(x))
    x = F.tanh(self.convt4(x))

    return x, temp_x2