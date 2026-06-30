import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from scipy.signal.windows import dpss
import numpy as np


class ChannelMLP(nn.Module):
    def __init__(self, total_K, future_K, hidden_dim=128, e_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = total_K
        for _ in range(e_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, future_K))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ChannelMixMLP(nn.Module):
    def __init__(self, total_K, future_K, num_channels, hidden_dim=256, e_layers=2, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        in_dim = total_K * num_channels

        layers = []
        for _ in range(e_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ]
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, future_K * num_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        B, K, C = x.shape
        x = x.reshape(B, K * C)
        out = self.mlp(x)
        return out.view(B, -1, C)


class Model(nn.Module):
    def __init__(self, configs, future_K=None, dropout=0.1,
                 direction='bottom_up'):
        super().__init__()
        self.use_channel_mixing=configs.use_channel_mixing
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.NWs = configs.NWs
        self.Kmax = configs.Kmax

        self.norm_type = getattr(configs, 'norm_type', 'global')

        # ------------------ DPSS ------------------
        self.dpss_windows_list = []
        sorted_NWs = sorted(self.NWs)

        for i, NW in enumerate(sorted_NWs):
            v = dpss(M=self.seq_len, NW=NW, Kmax=self.Kmax)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            buf = torch.tensor(v, dtype=torch.float32)
            self.register_buffer(f"dpss_{i}", buf)
            self.dpss_windows_list.append(buf)

        self.total_K = len(sorted_NWs) * self.Kmax

        self.scale_weights = nn.Parameter(torch.ones(len(self.dpss_windows_list)))

        # ------------------ future basis ------------------
        self.future_K = future_K if future_K else self.pred_len
        v_future = dpss(M=self.pred_len, NW=sorted_NWs[0], Kmax=self.future_K)
        v_future = v_future / np.linalg.norm(v_future, axis=1, keepdims=True)
        self.register_buffer("future_basis", torch.tensor(v_future, dtype=torch.float32))

        # ------------------ MLP ------------------
        if self.use_channel_mixing:
            self.mlp = ChannelMixMLP(self.total_K, self.future_K, self.enc_in,
                                     hidden_dim=self.d_model, e_layers=self.e_layers)
        else:
            self.linears = nn.ModuleList([
                ChannelMLP(self.total_K, self.future_K,
                           hidden_dim=self.d_model, e_layers=self.e_layers)
                for _ in range(self.enc_in)
            ])

        self.revin = RevIN(self.enc_in)

    def forward(self, x_enc, *args, **kwargs):

        B, L, C = x_enc.shape
        norm_type = self.norm_type

        # ================== normalization ==================
        if norm_type == 'none':
            x = x_enc

        elif norm_type == 'global':
            mean = x_enc.mean(dim=(0, 1), keepdim=True)
            std = x_enc.std(dim=(0, 1), keepdim=True) + 1e-3
            x = (x_enc - mean) / std
            self.mean = mean
            self.std = std

        elif norm_type == 'revin':
            x = self.revin(x_enc.permute(0, 2, 1), 'norm').permute(0, 2, 1)

        elif norm_type == 'channel':
            mean = x_enc.mean(dim=1, keepdim=True)
            std = x_enc.std(dim=1, keepdim=True) + 1e-5
            x = (x_enc - mean) / std
            self.mean = mean
            self.std = std

        elif norm_type == 'temporal':
            mean = x_enc.mean(dim=2, keepdim=True)
            std = x_enc.std(dim=2, keepdim=True) + 1e-5
            x = (x_enc - mean) / std
            self.mean = mean
            self.std = std

        elif norm_type == 'detrend':
            trend = x_enc.mean(dim=1, keepdim=True)
            x = x_enc - trend
            self.trend = trend

        elif norm_type == 'revin_detrend':
            trend = x_enc.mean(dim=1, keepdim=True)
            x = x_enc - trend
            x = self.revin(x.permute(0, 2, 1), 'norm').permute(0, 2, 1)
            self.trend = trend

        else:
            raise ValueError

        # ================== DPSS projection ==================
        weights = torch.softmax(self.scale_weights, dim=0)
        coeffs_all = []

        for c in range(C):
            x_c = x[:, :, c]
            coeff_c = []

            for i, v in enumerate(self.dpss_windows_list):
                coeff = x_c @ v.to(x.device).T
                coeff = coeff * weights[i]
                coeff_c.append(coeff)

            coeff_c = torch.cat(coeff_c, dim=-1)
            coeffs_all.append(coeff_c.unsqueeze(-1))

        coeffs = torch.cat(coeffs_all, dim=-1)

        # ================== MLP ==================
        if self.use_channel_mixing:
            pred_coeffs = self.mlp(coeffs)
        else:
            pred_list = []
            for c in range(C):
                pred = self.linears[c](coeffs[:, :, c])
                pred_list.append(pred.unsqueeze(-1))
            pred_coeffs = torch.cat(pred_list, dim=-1)

        # ================== reconstruction ==================
        future_basis = self.future_basis.to(x.device)
        outputs = []

        for c in range(C):
            seq = pred_coeffs[:, :, c] @ future_basis
            outputs.append(seq.unsqueeze(-1))

        recons = torch.cat(outputs, dim=-1)

        # ================== denorm ==================
        if norm_type == 'global' or norm_type == 'channel' or norm_type == 'temporal':
            recons = recons * self.std + self.mean

        elif norm_type == 'revin':
            recons = self.revin(recons.permute(0, 2, 1), 'denorm').permute(0, 2, 1)

        elif norm_type == 'detrend':
            recons = recons + self.trend

        elif norm_type == 'revin_detrend':
            recons = self.revin(recons.permute(0, 2, 1), 'denorm').permute(0, 2, 1)
            recons = recons + self.trend

        return recons