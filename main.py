import sys
from pathlib import Path

from tqdm import tqdm
import torch
import fire

from ddpm_tutorial.unet import Unet
from ddpm_tutorial.data import FashionMNIST, CIFAR10, transform, reverse_transform
from ddpm_tutorial.ddpm import DDPM

DATA_DIR = Path("./data/")
CKPT_DIR = Path("./checkpoints/")
IMG_DIR = Path("./images/")

def get_config(dataset):
    if dataset == "fashion":
        data = FashionMNIST(DATA_DIR, train=True, transform=transform, download=True)
        return {'data': data, 'channels': 1, "img_sz": (1, 32, 32)}

    if dataset == "cifar10":
        data = CIFAR10(DATA_DIR, train=True, transform=transform, download=True)
        return  {"data": data, "channels": 3, "img_sz": (3, 32, 32)}

    raise ValueError(f"{dataset} not supported")


class DDPM_RUN():
    def __init__(self, dataset="cifar10", timesteps=1000):

        assert (dataset in ("fashion", "cifar10")) 
        self.dataset = dataset
        self.timesteps = timesteps

    def train(self, batch_size=256, epochs=1000, ckpt_interval=10):
        cfg = get_config(self.dataset)
        dataloader = torch.utils.data.DataLoader(
            cfg["data"],
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        model = Unet(dim=32, dim_mults=(1, 2, 4,), channels=cfg["channels"])
        ddpm = DDPM(model, timesteps=self.timesteps, img_sz=cfg["img_sz"])
        ddpm.cuda()
        optim = torch.optim.Adam(ddpm.parameters(), lr=0.0001)
        
        CKPT_DIR.mkdir(exist_ok=True)
        ema_loss = 5.0

        for e in tqdm(range(epochs)):
            for x, _ in dataloader:
                optim.zero_grad()
                x = x.cuda()
                loss = ddpm(x)
                loss.backward()
                optim.step()
                
                ema_loss *= 0.001
                ema_loss += 0.999 * loss.item()

            print(f"epoch {e:05} loss {ema_loss}")
            if (e+1) % ckpt_interval == 0:
                torch.save(ddpm.state_dict(), CKPT_DIR/f"{self.dataset}_epc_{e}.pt")

        torch.save(ddpm.state_dict(), CKPT_DIR/f"{self.dataset}_epc_{epochs-1}.pt")

    def infer(self, epoch, sample_n=16):
        """
            use saved model at epoch 
        """
        cfg = get_config(self.dataset)
        model = Unet(dim=32, dim_mults=(1, 2, 4,), channels=cfg["channels"])
        ddpm = DDPM(model, timesteps=self.timesteps, img_sz=cfg["img_sz"])
        states = torch.load(CKPT_DIR/f"{self.dataset}_epc_{epoch}.pt")
        ddpm.load_state_dict(states)
        ddpm.cuda()
        imgs = ddpm.deblur_loop(batch_size=sample_n)
        IMG_DIR.mkdir(exist_ok=True)
        for idx, img in enumerate(imgs):
            for j, im in enumerate(img):
                with open(IMG_DIR/f"t_{idx:03}_sample_{j:03}.png", "wb") as fp:
                    im = reverse_transform(im)
                    im.save(fp)


if __name__ == "__main__":
    fire.Fire(DDPM_RUN)
