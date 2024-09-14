import torch
import torch.nn as nn
import torch.nn.functional as F



class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.embed_dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 128)

    def encode(self, x, mask):
        x = self.encoder(x, src_key_padding_mask=mask)
        return x

    def forward(self, x,y):
        # (batch_size, max_seq_len, embed_dim)
        x = torch.cat((x,y),dim=1)
        mask =  None
        x = self.embed_dropout(x)
        x = self.linear1(x)
        x = self.encode(x, mask)
        x = self.linear2(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=n_layer,
                              dropout=rnn_drop if n_layer > 1 else 0, batch_first=True, bidirectional=True)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x, _ = self.bilstm(x)
        return x

    def predict(self, x):
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.t())
        return probs

    def forward(self, x, *args):
        x = self.encode(x)
        return self.predict(x)


class BiLSTMAttn(BiLSTM):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float, n_head: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        x = x.transpose(0, 1)
        x = self.attn(x, x, x, key_padding_mask=mask)[0].transpose(0, 1)
        return self.predict(x)


class BiLSTMCNN(BiLSTM):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, x, *args):
        x = self.encode(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        return self.predict(x)


class BiLSTMConvAttRes(BiLSTM):
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float, n_head: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head)
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        res = x
        x = self.conv(x.transpose(1, 2)).relu()
        x = x.permute(2, 0, 1)
        x = self.attn(x, x, x, key_padding_mask=mask)[0].transpose(0, 1)
        x = self.norm(res + x)
        return self.predict(x)


class CNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, embed_drop: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, *args):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.t())
        return probs

class social_transformer(nn.Module):
    def __init__(self, past_len):
        super(social_transformer, self).__init__()
        self.past_len = past_len
        self.encode_past = nn.Linear(past_len,128, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
        self.conv1d = Conv1d_with_init(128,128,1,'cuda:0')
    def forward(self, h, mask):
        '''
        h: batch_size, t, 2
        '''
        if self.past_len == 128:
            h_feat = self.conv1d(h.transpose(2,1)).transpose(2,1)
        else:
            h_feat = self.encode_past(h)
        # print(h_feat.shape)
        # n_samples, 1, 64
        h_feat_ = self.transformer_encoder(h_feat, mask)
        h_feat = h_feat + h_feat_

        return h_feat

class st_encoder(nn.Module):
    def __init__(self,channel_in):
        super().__init__()
        channel_in = channel_in
        channel_out = 64
        dim_kernel = 3
        self.dim_embedding_key = 128
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X):
        '''
        X: b, T, 2

        return: b, F`
        '''
        X_t = torch.transpose(X, 1, 2)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, 1, 2)

        output_x, state_x = self.temporal_encoder(X_embed)
        state_x = state_x.transpose(1,0)

        return state_x

def Conv1d_with_init(in_channels, out_channels, kernel_size, device):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size).to(device)
    nn.init.kaiming_normal_(layer.weight)

    return layer

class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat=(512, 256), activation=None, dropout=-1):
        super(MLP, self).__init__()
        dims = (in_feat, ) + hid_feat + (out_feat, )

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(Conv1d_with_init(dims[i], dims[i + 1],1,'cuda:0'))

        self.activation = activation if activation is not None else lambda x: x
        self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(x)
            x = self.dropout(x)
            x = self.layers[i](x.transpose(2,1)).transpose(2,1)
        return x


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        # ctx: (B, 1, F+3)
        # x: (B, T, 2)

        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret