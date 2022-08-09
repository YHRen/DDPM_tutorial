from ddpm_tutorial.unet import Unet
import numpy as np
import torch

model = Unet(dim=64)

# x = torch.rand((2,3,256,256))
# t = torch.randint(0, 1000, (2,))
# print(sum((p.numel() for p in model.parameters())))
# print(x.shape)
# y = model(x, t)
# print(y.shape)

from torchvision.datasets import CIFAR10
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(32),
    T.RandomCrop(32),
    T.ToTensor(),
    T.Lambda(lambda t: (t * 2) - 1),
    ])

reverse_transform = T.Compose([
    T.Lambda(lambda t: (t + 1) / 2),
    T.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    T.Lambda(lambda t: t * 255.),
    T.Lambda(lambda t: t.numpy().astype(np.uint8)),
    T.ToPILImage(),
    ])

dataset = CIFAR10('./', train=True, transform=transform, download=True)

