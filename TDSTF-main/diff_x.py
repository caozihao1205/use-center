import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from attn import ResNet
# from attn_ns import ResNet
# from attn_new import  ResNet
# from tranenc import  ResNet
# from lstm_attn import  ResNet
from attn_timediffde import ResNet


class TDSTF(nn.Module):

    def __init__(self, config, args, device):
        super().__init__()
        self.device = device
        self.config_diff = config['diffusion']
        var, _ = pickle.load(open('preprocess/data/var.pkl', 'rb'))
        self.lv = len(var)
        self.res_model = ResNet(self.config_diff, args, self.device)
        # parameters for diffusion model
        self.num_steps = self.config_diff['num_steps']
        self.beta = np.linspace(self.config_diff['beta_start'] ** 0.5, self.config_diff['beta_end'] ** 0.5,
                                self.num_steps) ** 2  # size(50)一维
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)  # size(50,1)
        self.alpha_cumprod_prev = np.pad(self.alpha[:-1], (1, 0), mode='constant', constant_values=1.)
        self.posterior_mean_coef1 = self.beta * np.sqrt(self.alpha_cumprod_prev) / (1. - self.alpha)
        self.posterior_mean_coef2 = (1. - self.alpha_cumprod_prev) * np.sqrt(self.alpha) / (1. - self.alpha)
        self.sigma = ((1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha) * self.beta) ** 0.5
        self.posterior_mean_coef1_torch = torch.tensor(self.posterior_mean_coef1)
        self.posterior_mean_coef2_torch = torch.tensor(self.posterior_mean_coef2)
        self.sigma_torch = torch.tensor(self.sigma)
    def process(self, batch):
        samples_x = batch['samples_x'].to(self.device).float()
        samples_y = batch['samples_y'].to(self.device).float()
        info = batch['info'].to(self.device)

        return samples_x, samples_y, info

    def forward(self, batch, size_x, size_y):
        samples_x, samples_y, info = self.process(batch)
        t = torch.randint(0, self.num_steps, [len(samples_x)]).to(self.device)  # size(a)一维（32，）
        current_alpha = self.alpha_torch[t]  # （32,1）
        noise = torch.randn((len(samples_x), size_y)).to(samples_y.device)  # 32,30
        mask_x = samples_x[:, 3]
        mask_y = samples_y[:, 3]
        samples_x[:, 0] = torch.where(mask_x == 1, samples_x[:, 0],
                                      torch.tensor(self.lv, dtype=torch.float32).to(self.device))
        samples_x[:, 1] = torch.where(mask_x == 1, samples_x[:, 1],
                                      torch.tensor(-1, dtype=torch.float32).to(self.device))
        samples_y[:, 0] = torch.where(mask_y == 1, samples_y[:, 0],
                                      torch.tensor(self.lv, dtype=torch.float32).to(self.device))
        samples_y[:, 1] = torch.where(mask_y == 1, samples_y[:, 1],
                                      torch.tensor(-1, dtype=torch.float32).to(self.device))
        samples_y0      = samples_y
        samples_yt      = samples_y
        samples_yt[:,2] = ((current_alpha ** 0.5) * samples_y[:, 2] + (
                    (1.0 - current_alpha) ** 0.5) * noise) * mask_y  # 正向扩散
        predicted = self.res_model(samples_x, samples_y0,samples_yt, info, t)
        residual = torch.where(mask_y == 1, samples_y0[:,2] - predicted, torch.tensor(0, dtype=torch.float32).to(self.device))
        loss = (residual ** 2).sum() / info[:, 2].sum()

        return loss

    def forecast(self, samples_x, samples_y, info, n_samples):
        generation = torch.zeros(n_samples, samples_y.shape[0], samples_y.shape[-1]).to(self.device)
        for i in range(n_samples):
            samples_y0 = samples_y
            samples_yt = samples_y
            samples_yt[:, 2] = torch.randn_like(samples_y[:, 2]) * samples_y[:, 3]
            for t in range(self.num_steps - 1, -1, -1):
                mask_x = samples_x[:, 3]
                mask_y = samples_y[:, 3]
                samples_x[:, 0] = torch.where(mask_x == 1, samples_x[:, 0],
                                              torch.tensor(self.lv, dtype=torch.float32).to(self.device))
                samples_x[:, 1] = torch.where(mask_x == 1, samples_x[:, 1],
                                              torch.tensor(-1, dtype=torch.float32).to(self.device))
                samples_y0[:, 0] = torch.where(mask_y == 1, samples_y[:, 0],
                                              torch.tensor(self.lv, dtype=torch.float32).to(self.device))
                samples_y0[:, 1] = torch.where(mask_y == 1, samples_y[:, 1],
                                              torch.tensor(-1, dtype=torch.float32).to(self.device))
                samples_yt[:, 0] = torch.where(mask_y == 1, samples_y[:, 0],
                                              torch.tensor(self.lv, dtype=torch.float32).to(self.device))
                samples_yt[:, 1] = torch.where(mask_y == 1, samples_y[:, 1],
                                              torch.tensor(-1, dtype=torch.float32).to(self.device))
                predicted = self.res_model(samples_x,samples_y0, samples_yt, info, torch.tensor([t]).to(self.device))

                samples_yt[:, 2] = (self.posterior_mean_coef1_torch[t] * predicted + self.posterior_mean_coef2_torch[t] * samples_yt[:,2]) * samples_yt[:, 3]
                if t > 0:
                    noise = torch.randn_like(samples_y[:, 2]) * samples_y[:, 3]
                    samples_yt[:, 2] += self.sigma_torch[t] * noise

            generation[i] = samples_yt[:, 2].detach()

        return generation.permute(1, 2, 0)  # 32,30,5

    def evaluate(self, batch, n_samples):
        samples_x, samples_y, info = self.process(batch)
        with torch.no_grad():
            generation = self.forecast(samples_x, samples_y, info, n_samples)

        return generation, samples_y, samples_x
