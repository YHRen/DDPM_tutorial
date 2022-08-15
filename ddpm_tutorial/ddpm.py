import torch
import torch.nn.functional as F
from tqdm import tqdm

class DDPM(torch.nn.Module):

    def __init__(self, model, img_sz=(3,32,32), timesteps=1000):
        super().__init__()
        self.model = model
        self.img_sz = img_sz
        self.timesteps = timesteps
        beta = torch.linspace(
                1000/timesteps*0.0001,
                1000/timesteps*0.02,
                timesteps,
                dtype=torch.float64)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta.float())
        self.register_buffer("alpha_bar", alpha_bar.float())

    def add_noise(self, x0, t, noise):
        tmp = torch.gather(self.alpha_bar, dim=0, index=t).view(-1, *[1]*3)
        xt = torch.sqrt(tmp) * x0 + torch.sqrt(1-tmp)*noise
        return xt

    def forward(self, x0):
        bsz, *_ = x0.shape
        device = x0.device
        t = torch.randint(0, self.timesteps, (bsz,), dtype=torch.long, device=device)
        noise = torch.randn_like(x0).to(device)
        xt = self.add_noise(x0, t, noise)
        pred = self.model(xt, t)
        loss = F.mse_loss(pred, noise)
        return loss

    @torch.no_grad()
    def denoise(self, x, t):
        beta = torch.gather(self.beta, dim=0, index=t).view(-1, *[1]*3)
        alpha_bar = torch.gather(self.alpha_bar, dim=0, index=t).view(-1, *[1]*3)
        coef = beta / torch.sqrt(1-alpha_bar)
        pred = self.model(x, t)
        pred *= coef

        noise = torch.randn_like(x, device=x.device)
        noise *= torch.sqrt(beta)

        ans = (x-pred)/torch.sqrt(1-beta) + noise
        return ans

    @torch.no_grad()
    def denoise_last_step(self, x, t):
        beta = torch.gather(self.beta, dim=0, index=t).view(-1, *[1]*3)
        alpha_bar = torch.gather(self.alpha_bar, dim=0, index=t).view(-1, *[1]*3)
        coef = beta / torch.sqrt(1-alpha_bar)
        pred = self.model(x, t)
        pred *= coef
        ans = (x-pred)/torch.sqrt(1-beta)
        return ans

    @torch.no_grad()
    def denoise_loop(self, batch_size=16, record_step=50):
        device = self.beta.device
        imgs = []
        img = torch.randn(batch_size, *self.img_sz, device=device)
        imgs.append(img)
        for t in tqdm(range(self.timesteps-1, 0, -1)):
            bt = torch.full((batch_size,), t, device=device)
            img = self.denoise(img, bt)
            if t%record_step==0:
                imgs.append(img)

        t = torch.full((batch_size,), 0, device=device)
        img = self.denoise_last_step(img, t)
        imgs.append(img)
        return imgs

