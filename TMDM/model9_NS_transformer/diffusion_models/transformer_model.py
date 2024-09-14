import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F




# from models import FEDformer, Autoformer, Informer, Transformer
def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)

    return layer


class TimeEmbedding(nn.Module):
    def __init__(self, tp, d_model):
        super().__init__()
        self.register_buffer('time_embedding', self._build_embedding(tp + 1, d_model), persistent=False)

    def forward(self, m):
        return self.time_embedding[m]

    def _build_embedding(self, t, d_model):
        pe = torch.zeros(t, d_model)
        position = torch.arange(t).unsqueeze(1)
        div_term = (1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim):
        super().__init__()
        self.register_buffer('diffusion_embedding', self._build_embedding(num_steps, embedding_dim / 2),
                             persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, diffusion_step):
        x = self.diffusion_embedding[diffusion_step]  # 32,128
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)

        return x

    def _build_embedding(self, num_steps, dim):  # 50,128
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = (10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0))
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)

        return table


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        var, target_var = pickle.load(open('data/var.pkl', 'rb'))
        lv = len(var)
        self.size_x = 60
        # self.size_y = 10 * len(target_var)
        self.channels = 128
        self.emb_f = nn.Embedding(lv + 1, self.channels) # 130,128
        self.emb_t = TimeEmbedding(40, 128)  # 41,128
        self.emb_vx = nn.Linear(1, self.channels)
        self.emb_vy = nn.Linear(2, self.channels)
        self.dec1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.dec2 = Conv1d_with_init(self.channels, 1, 1)
        self.diffusion_embedding = DiffusionEmbedding(1000, 128)
        self.diffusion_projection = nn.Linear(128,128)
        self.residual_layers = nn.ModuleList([
            Triplet_cor()
            for _ in range(3)])

    def forward(self, x,x_mark,y_mark,y_t,y_0_hat,diffusion_step):
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb)  # 32,128
        diffusion_emb = diffusion_emb.unsqueeze(1).expand(diffusion_emb.shape[0], self.size_x,
                                                          diffusion_emb.shape[1])  # 32,60,128

        triplets_x = (self.emb_f(x_mark[:, 0].to(torch.int64))  # 32,60,128
                      + self.emb_t(x_mark[:, 1].to(torch.int64))
                      + self.emb_vx(x.unsqueeze(-1))
                      + diffusion_emb) * x_mark[:, 2].unsqueeze(-1)

        triplets_y = (self.emb_f(y_mark[:, 0].to(torch.int64))  # 32,30,128
                      + self.emb_t(y_mark[:, 1].to(torch.int64))
                      + self.emb_vy(torch.cat((y_t.unsqueeze(-1), y_0_hat.unsqueeze(-1)), dim=-1))
                      ) * y_mark[:, 2].unsqueeze(-1)
        # diffussion_emb_y = diffusion_emb[:, : self.size_y] * y_mark[:,2].unsqueeze(-1)
        diffussion_emb_y = diffusion_emb * y_mark[:,2].unsqueeze(-1)
        skip = []
        for layer in self.residual_layers:
            triplets_y = triplets_y + diffussion_emb_y
            triplets_y, skip_connection = layer(triplets_x, triplets_y)

            skip.append(skip_connection)  # 3个skip_connection:32,30,128

        output = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))  # 32,30,128
        output = self.dec1(output.permute(0, 2, 1))  # 32,256,30
        output = F.relu(output)
        output = self.dec2(output)  # 32,1,30

        return output.squeeze()  # 32,30


class Triplet_cor(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 128
        # self.attn = Transformer(d_model=self.channels, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1, activation='gelu', batch_first=True, device=device)
        self.attn = torch.nn.Transformer(d_model=self.channels, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
                                         dim_feedforward=256, dropout=0.1, activation='gelu', batch_first=True
                                         )
        self.expand = Conv1d_with_init(self.channels, 2 * self.channels, 1)

    def forward(self, triplets_x, triplets_y):
        output = self.attn(triplets_x, triplets_y)
        output = self.expand(output.transpose(1, 2)).transpose(1, 2)
        residual, skip = torch.chunk(output, 2, dim=-1)

        return residual, skip