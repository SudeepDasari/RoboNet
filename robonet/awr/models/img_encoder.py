import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules=list(resnet.children())[:-1]

        self._resnet=nn.Sequential(*modules)
        self._fc1 = nn.Linear(2053, 512)
        self._fc2 = nn.Linear(512, 256)
        self._fc3 = nn.Linear(256, 128)

    def forward(self, images, state):
        image_enc = self._resnet(images).reshape((images.shape[0], -1))
        image_state = torch.cat((image_enc, state), -1)

        fc1 = F.relu(self._fc1(image_state))
        fc2 = F.relu(self._fc2(fc1))
        return F.relu(self._fc3(fc2))
