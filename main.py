import sys
from ddpm_tutorial.unet import Unet
from ddpm_tutorial.data import cifar10, fashionmnist, reverse_transform
from ddpm_tutorial.ddpm import DDPM
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path


ckpt_dir = Path("./checkpoints/")
ckpt_dir.mkdir(exist_ok=True)

fmnist_cfg = {'data': fashionmnist,
        'channels': 1,
        "img_sz": (1, 32, 32)}
cifar10_cfg = {"data": cifar10,
        "channels": 3,
        "img_sz": (3, 32, 32)}

cfg = fashionmnist
bsz = 256
epc = 1000
timesteps = 1000


if __name__ == "__main__":
    if sys.argv[1] == "train":
        dataloader = torch.utils.data.DataLoader(
                cfg["data"],
                batch_size=bsz,
                num_workers=8,
                shuffle=True,
                pin_memory=True,
                drop_last=True)
        model = Unet(dim=32, dim_mults=(1, 2, 4,), channels=cfg["channels"])
        ddpm = DDPM(model, timesteps=timesteps, img_sz=cfg["img_sz"])
        ddpm.cuda()
        optim = torch.optim.Adam(ddpm.parameters(), lr=0.0001)

        for e in tqdm(range(epc)):
            for x, y in dataloader:
                optim.zero_grad()
                x = x.cuda()
                loss = ddpm(x)
                loss.backward()
                optim.step()
                print(loss.item())
            torch.save(ddpm.state_dict(), ckpt_dir/f"e_{e}.pt")

    elif sys.argv[1] == "infer":
        model = Unet(dim=32, dim_mults=(1, 2, 4,), channels=cfg["channels"])
        ddpm = DDPM(model, timesteps=timesteps, img_sz=cfg["img_sz"])
        states = torch.load(ckpt_dir/f"e_{sys.argv[2]}.pt")
        ddpm.load_state_dict(states)
        ddpm.cuda()
        imgs = ddpm.deblur_loop()
        img_folder = Path("./images")
        img_folder.mkdir(exist_ok=True)
        for idx, img in enumerate(imgs):
            for j, im in enumerate(img):
                if idx == 4:
                    print(idx, j, im)
                with open(img_folder/f"t_{idx:03}_sample_{j:03}.png", "wb") as fp:
                    im = reverse_transform(im)
                    im.save(fp)
