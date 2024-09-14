import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import  Transformer,st_encoder,social_transformer,MLP,ConcatSquashLinear
from typing import Union
# from models import FEDformer, Autoformer, Informer, Transformer
import tran
def Conv1d_with_init(in_channels, out_channels, kernel_size, device):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size).to(device)
    nn.init.kaiming_normal_(layer.weight)

    return layer

class Bottleneck1D(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = out.permute(0, 2, 1)

        out += residual
        out = self.relu(out)

        return out
class TemporalBlock(nn.Module):
    """
    combination of (convolutional layer, chomp, relu, dropout) repeated `layers` times
    adds additional convolutional layer if needed to downsample number of channels
    inspired by https://github.com/locuslab/TCN
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                  stride: int, dilation: int, padding: int,
                  dropout: int = 0.2, layers: int = 2):
        super().__init__()

        conv_layer = nn.Conv1d
        self.padding = padding
        self.dropout = dropout

        net = []
        for i in range(layers):
            net.append(torch.nn.utils.weight_norm(conv_layer(
                (n_inputs if i == 0 else n_outputs), n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
            net.append(nn.ReLU())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
        self.net = nn.ModuleList(net)
        self.con = nn.Conv1d(60,30,1)
        self.downsample = conv_layer(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        sets normal weight distribution for convolutional layers
        """
        for i in range(0, len(self.net), 2 + (self.dropout > 0) ):
            self.net[i].weight.data.normal_(0, 0.5)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        `x` in format [batch_size, channels, *other_dims]
        """
        out = x
        c = out.permute(0,2,1)
        for i in range(len(self.net)):
            out = self.net[i](c)

        res = x if self.downsample is None else self.downsample(x)
        a=self.relu(out.permute(0,2,1) + res)

        a=self.con(a)
        return a

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
        self.dec1 = Conv1d_with_init(256, 128, 1, device)
        self.dec2 = Conv1d_with_init(128, 1, 1, device)
        self.dec3 = Conv1d_with_init(32, 128, 1, device)
        self.dec4 = Conv1d_with_init(60, 30, 1, device)
        self.CNN = Bottleneck1D(128, 64, 1).to(device)
        self.dec5 = Conv1d_with_init(60,30,1,device)
        self.dec6 = Conv1d_with_init(128,256, 1, device)
        self.dec7 = Conv1d_with_init(60, 30, 1, device)
        self.dec8 = Conv1d_with_init(30, 60, 1, device)
        self.diffusion_embedding = DiffusionEmbedding(config['num_steps'], config['diffusion_embedding_dim'], device)
        self.diffusion_projection = nn.Linear(config['diffusion_embedding_dim'], self.channels).to(device)
        # self.residual_layers = nn.ModuleList([
        #     Triplet_cor(config, lv, device)
        #     for _ in range(config['layers'])])

        self.tran = torch.nn.Transformer(d_model=self.channels, nhead=8, num_encoder_layers=config['enl'], num_decoder_layers=config['del'], dim_feedforward=256, dropout=0.1, activation='gelu', batch_first=True, device=device)
        self.armodel = TemporalBlock(128,128,3,1,1,1)
        layer1 = tran.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256, dropout=0.1,batch_first=True)
        self.encoder1 = tran.TransformerEncoder(layer1, num_layers=2)
        layer2 = tran.TransformerDecoderLayer(
            d_model=256, nhead=8, dim_feedforward=128, dropout=0.1,batch_first=True)
        self.encoder2 = tran.TransformerDecoder(layer2, num_layers=2)

    def forward(self, samples_x, samples_y0,samples_yt, info, diffusion_step):
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb)  # 32,128
        diffusion_emb = diffusion_emb.unsqueeze(1).expand(diffusion_emb.shape[0], self.size_x,
                                                          diffusion_emb.shape[1])  # 32,60,128
        triplets_x = (self.emb_f(samples_x[:, 0].to(torch.int64))  # 32,60,128
                      + self.emb_t(samples_x[:, 1].to(torch.int64))
                      + self.emb_v(samples_x[:, 2].unsqueeze(-1))
                      # + diffusion_emb
                      ) * samples_x[:, 3].unsqueeze(-1)
        triplets_y0 = (self.emb_f(samples_y0[:, 0].to(torch.int64))  # 32,30,128
                      + self.emb_t(samples_y0[:, 1].to(torch.int64))
                      + self.emb_v(samples_y0[:, 2].unsqueeze(-1))
                      ) * samples_y0[:, 3].unsqueeze(-1)
        triplets_yt= (self.emb_f(samples_yt[:, 0].to(torch.int64))  # 32,30,128
                      + self.emb_t(samples_yt[:, 1].to(torch.int64))
                      + self.emb_v(samples_yt[:, 2].unsqueeze(-1))
                      ) * samples_yt[:, 3].unsqueeze(-1)
        diffussion_emb_y = diffusion_emb[:, : self.size_y] * samples_y0[:, 3].unsqueeze(-1)

        # x=triplets_x
        # yt=triplets_yt
        # a=torch.cat((x,yt),dim=1)
        # mask = torch.randn_like(a)
        # mask_x = mask[:,0:60,:]
        # mask_y = mask[:,-30:,:]
        # z_mix = self.CNN(x)
        # if self.training == True:
        #     z_mix = (mask_x * z_mix) + self.dec8((1 - mask_y) * yt)
        # else:
        #     z_mix = mask_x * z_mix

        x = triplets_x
        y0= triplets_y0
        yt=triplets_yt
        mask = torch.randn((x.shape[0],1,1)).to(x.device)

        z_mix = self.CNN(x)
        if self.training == True:
            z_mix = mask *self.dec7( x ) + (1 - mask) * y0
        else:
            z_mix = mask *self.dec7( x)
        #
        # if self.training == True:
        #     z_mix = mask * self.dec7( x)
        # else:
        #     z_mix = mask * self.dec7( x )

        z_ar = self.armodel(x)
        c = torch.cat((z_mix,z_ar),dim=2)
        y = yt+diffussion_emb_y
        z_k = self.encoder1(y)

        z_k = self.dec6(z_k.permute(0,2,1)).permute(0,2,1)

        output = self.encoder2(z_k,c)



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
