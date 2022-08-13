from torchvision.datasets import CIFAR10, FashionMNIST
import torchvision.transforms as T
import numpy as np

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
    T.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    T.ToPILImage(),
    ])
