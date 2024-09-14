import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import  Transformer,st_encoder,social_transformer,MLP,ConcatSquashLinear

# from models import FEDformer, Autoformer, Informer, Transformer
def Conv1d_with_init(in_channels, out_channels, kernel_size, device):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size).to(device)
    nn.init.kaiming_normal_(layer.weight)

    return layer




class TimeEmbedding(nn.Module):
    def __init__(self, tp, d_model, device):
        super().__init__()
        self.device = device
        self.register_buffer('time_embedding', self._build_embedding(tp + 1, d_model), persistent=False)

    def forward(self, m):
        return self.time_embedding[m]

    def _build_embedding(self, t, d_model):
        pe = torch.zeros(t, d_model).to(self.device)
        position = torch.arange(t).unsqueeze(1).to(self.device)
        div_term = (1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)).to(self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim, device):
        super().__init__()
        self.device = device
        self.register_buffer('diffusion_embedding', self._build_embedding(num_steps, embedding_dim / 2),
                             persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim).to(device)

    def forward(self, diffusion_step):
        x = self.diffusion_embedding[diffusion_step]  # 32,128
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)

        return x

    def _build_embedding(self, num_steps, dim):  # 50,128
        steps = torch.arange(num_steps).unsqueeze(1).to(self.device)
        frequencies = (10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)).to(self.device)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)

        return table


class ResNet(nn.Module):
    def __init__(self, config, args, device):
        super().__init__()
        var, target_var = pickle.load(open('preprocess/data/var.pkl', 'rb'))
        lv = len(var)
        self.size_x = config['size']
        self.size_y = 10 * len(target_var)
        self.channels = config['channels']
        self.emb_f = nn.Embedding(lv + 1, self.channels).to(device)  # 130,128
        self.emb_t = TimeEmbedding(config['time_points'], config['time_embedding_dim'], device)  # 41,128
        self.emb_v = nn.Linear(1, self.channels).to(device)
        self.dec1 = Conv1d_with_init(self.channels, self.channels, 1, device)
        self.dec2 = Conv1d_with_init(self.channels, 1, 1, device)
        self.dec3 = Conv1d_with_init(32, 128, 1, device)
        self.dec4 = Conv1d_with_init(60, 30, 1, device)
        self.diffusion_embedding = DiffusionEmbedding(config['num_steps'], config['diffusion_embedding_dim'], device)
        self.diffusion_projection = nn.Linear(config['diffusion_embedding_dim'], self.channels).to(device)
        self.residual_layers = nn.ModuleList([
            Triplet_cor(config, lv, device)
            for _ in range(config['layers'])])

        self.tran =torch.nn.Transformer(d_model=self.channels, nhead=8, num_encoder_layers=config['enl'], num_decoder_layers=config['del'], dim_feedforward=256, dropout=0.1, activation='gelu', batch_first=True, device=device)

        self.st_encoder=st_encoder(128)
        self.social_encoder=social_transformer(128)
        self.x_decoder = MLP(128*2, 128, hid_feat=(512,256), activation=nn.ReLU())


        self.st_encoder_y=st_encoder(128)
        self.social_encoder_y=social_transformer(128)
        self.y_decoder = MLP(128*2, 128, hid_feat=(512,256), activation=nn.ReLU())

        self.st_encoder1=st_encoder(3)
        self.social_encoder1=social_transformer(3)
        self.x_decoder1 = MLP(128*2, 128, hid_feat=(192,), activation=nn.ReLU())


        self.st_encoder_y1=st_encoder(3)
        self.social_encoder_y1=social_transformer(3)
        self.y_decoder1 = MLP(128*2, 128, hid_feat=(192,), activation=nn.ReLU())

        self.concat1 = ConcatSquashLinear(128, 2 * 128, 128)
        self.concat2 = ConcatSquashLinear(2*128, 128, 128)
        self.concat3 = ConcatSquashLinear(128,  128//2, 128)
        self.concat4 = ConcatSquashLinear(128//2, 128//4, 128)
    def forward(self, samples_x, samples_y, info, diffusion_step):
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb)  # 32,128
        diffusion_emb = diffusion_emb.unsqueeze(1).expand(diffusion_emb.shape[0], self.size_x,
                                                          diffusion_emb.shape[1])  # 32,60,128
        triplets_x = (self.emb_f(samples_x[:, 0].to(torch.int64))  # 32,60,128
                      + self.emb_t(samples_x[:, 1].to(torch.int64))
                      + self.emb_v(samples_x[:, 2].unsqueeze(-1))
                      # + diffusion_emb
                      ) * samples_x[:, 3].unsqueeze(-1)
        triplets_y = (self.emb_f(samples_y[:, 0].to(torch.int64))  # 32,30,128
                      + self.emb_t(samples_y[:, 1].to(torch.int64))
                      + self.emb_v(samples_y[:, 2].unsqueeze(-1))
                      ) * samples_y[:, 3].unsqueeze(-1)
        diffussion_emb_y = diffusion_emb[:, : self.size_y] * samples_y[:, 3].unsqueeze(-1)
        skip = []

        # triplets_x = triplets_x + diffusion_emb * samples_x[:, 3].unsqueeze(-1)
        # triplets_y = triplets_y + diffussion_emb_y

        # x_ori = samples_x[:,0:3].permute(0,2,1)
        # x_mask = samples_x[:, 3].unsqueeze(-1)
        # y_ori = samples_y[:,0:3].permute(0,2,1)
        # y_mask = samples_y[:, 3].unsqueeze(-1)
        #
        # social_embed_x = self.social_encoder1(x_ori,None ) * x_mask
        # st_embed_x = self.st_encoder1(x_ori).repeat(1,60,1) * x_mask
        # x_total = torch.cat((st_embed_x, social_embed_x), dim=-1)
        # x = self.x_decoder1(x_total)
        # x = x + diffusion_emb * samples_x[:, 3].unsqueeze(-1)
        # social_embed_y = self.social_encoder_y1(y_ori,None ) * y_mask
        # st_embed_y = self.st_encoder_y1(y_ori).repeat(1,30,1) * y_mask
        # y_total = torch.cat((st_embed_y, social_embed_y), dim=-1)
        # y = self.y_decoder1(y_total)
        # y = y +diffussion_emb_y

        #
        # social_embed_x = self.social_encoder(triplets_x,None )
        # st_embed_x = self.st_encoder(triplets_x).repeat(1,60,1)
        # x_total = torch.cat((st_embed_x, social_embed_x), dim=-1)
        # x = self.x_decoder(x_total)
        # x = x + diffusion_emb * samples_x[:, 3].unsqueeze(-1)
        #
        # social_embed_y = self.social_encoder_y(triplets_y,None )
        # st_embed_y = self.st_encoder_y(triplets_y).repeat(1,30,1)
        # y_total = torch.cat((st_embed_y, social_embed_y), dim=-1)
        # y = self.y_decoder(y_total)
        # y = y + diffussion_emb_y

        #
        x = triplets_x + diffusion_emb
        y= triplets_y + diffussion_emb_y
        #
        output = self.tran(x,y)

        # x=self.dec4(x)
        # y = self.concat1(x,y)
        # y = self.concat2(x, y)
        # y = self.concat3(x, y)
        # output = self.concat4(x, y)
        # output = self.dec3(output.permute(0, 2, 1))  # 32,256,30
        # output = F.relu(output)
        # output = self.dec2(output)  # 32,1,30

        # # 五层tran
        # triplets_x = x + diffusion_emb * samples_x[:, 3].unsqueeze(-1)
        # triplets_y0 = y + diffussion_emb_y
        # triplets_y1, skip_connection1 = self.residual_layers[0](triplets_x, triplets_y0)
        #
        # triplets_y1 = triplets_y1 + diffussion_emb_y
        # triplets_y2, skip_connection2 = self.residual_layers[1](triplets_x, triplets_y1)
        #
        # triplets_y2 = triplets_y2 + diffussion_emb_y
        # triplets_y3, skip_connection3 = self.residual_layers[2](triplets_x, triplets_y2)
        #
        # triplets_y3 = triplets_y3 + diffussion_emb_y + skip_connection2
        # triplets_y4, skip_connection4 = self.residual_layers[3](triplets_x, triplets_y3)
        #
        # triplets_y4 = triplets_y4 + diffussion_emb_y + skip_connection1
        # triplets_y5, skip_connection5 = self.residual_layers[4](triplets_x, triplets_y4)
        # # 1
        # skip.extend([
        #                 skip_connection1 + skip_connection2 + skip_connection3 + skip_connection4 + skip_connection5])  # 3个skip_connection:32,30,128
        # output = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))  # 32,30,128
        #
        # output = self.dec1(output.permute(0, 2, 1))#32,128,30
        # output = F.relu(output)
        # output = self.dec2(output)#32,1,30


        # output = self.dec3(output.permute(0, 2, 1))

        output = self.dec1(output.permute(0, 2, 1))  # 32,256,30
        output = F.relu(output)
        output = self.dec2(output)  # 32,1,30

        return output.squeeze()  # 32,30


class Triplet_cor(nn.Module):
    def __init__(self, config, lv, device):
        super().__init__()
        self.channels = config['channels']
        # self.attn = Transformer(d_model=self.channels, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1, activation='gelu', batch_first=True, device=device)
        self.attn = torch.nn.Transformer(d_model=self.channels, nhead=8, num_encoder_layers=config['enl'],
                                         num_decoder_layers=config['del'], dim_feedforward=256, dropout=0.1,
                                         activation='gelu', batch_first=True, device=device)
        self.expand = Conv1d_with_init(self.channels, 2 * self.channels, 1, device)

    def forward(self, triplets_x, triplets_y):
        output = self.attn(triplets_x, triplets_y)
        output = self.expand(output.transpose(1, 2)).transpose(1, 2)
        residual, skip = torch.chunk(output, 2, dim=-1)

        return residual, skip
