from ddpm_tutorial.unet import Unet
from ddpm_tutorial.data import cifar10
import numpy as np
import torch
import torch.nn.functional as F

model = Unet(dim=64)

# x = torch.rand((2,3,256,256))
# t = torch.randint(0, 1000, (2,))
# print(sum((p.numel() for p in model.parameters())))
# print(x.shape)
# y = model(x, t)
# print(y.shape)



bsz = 128
dataloader = torch.utils.data.DataLoader(
        cifar10,
        batch_size=bsz,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

# for x, y in dataloader:
#     print(x.shape, y.shape)
#     break

class DDPM(torch.nn.Module):

    def __init__(self, model, img_sz, timeline=1000):
        super().__init__()
        self.model = model
        self.img_sz = img_sz
        self.timeline = timeline
        beta = torch.linspace(0.0001, 0.2, timeline)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.register_buffer("beta", beta) 
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)

    def blurring(self, x0, t, noise):
        tmp = torch.gather(self.alpha_bar, dim=0, index=t).view(-1, 1, 1, 1)
        xt = torch.sqrt(tmp) * x0 + (1-tmp)*noise
        return xt

    def denoise(self, xt, t):
        estimated_error = self.model(xt, t)
        return estimated_error

    def forward(self, x0):
        bsz, *_ = x0.shape
        device = x0.device
        t = torch.randint(0, self.timeline, (bsz,), dtype=torch.long, device=device)
        noise = torch.randn_like(x0).to(device)
        xt = self.blurring(x0, t, noise)
        pred = self.model(xt, t)
        loss = F.mse_loss(pred, noise)
        return loss





ddpm = DDPM(model, 256,10)
print(ddpm.beta, ddpm.beta.shape)
print(ddpm.alpha, ddpm.alpha.shape)
print(ddpm.alpha_bar, ddpm.alpha.shape)

optim = torch.optim.Adam(ddpm.parameters(), lr=0.0001)

ddpm.cuda()
print(ddpm.alpha_bar.device)
for x, y in dataloader:
    optim.zero_grad()
    x = x.cuda()
    loss = ddpm(x)
    loss.backward()
    optim.step()
    print(loss.item())

