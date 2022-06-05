import torch
from torch.utils import model_zoo
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import ResNet

from retinanet import model
from retinanet.dataloader import CSVDataset, Normalizer, Augmenter, Resizer
from retinanet.utils import Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

writer = SummaryWriter('retinanet')

retinanet = torch.load('resnet101-5d3b4d8f.pth')

writer.add_graph(retinanet)
writer.close()
