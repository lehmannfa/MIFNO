"""
original authors: Zongyi Li and Daniel Zhengyu Huang https://github.com/alasdairtran/fourierflow/tree/main
modified by: Fanny Lehmann
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
import copy


class WNLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, wnorm=False):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        if wnorm:
            weight_norm(self)

        self._fix_weight_norm_deepcopy()

    def _fix_weight_norm_deepcopy(self):
        # Fix bug where deepcopy doesn't work with weightnorm.
        # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
        orig_deepcopy = getattr(self, '__deepcopy__', None)

        def __deepcopy__(self, memo):
            # save and delete all weightnorm weights on self
            weights = {}
            for hook in self._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    weights[hook.name] = getattr(self, hook.name)
                    delattr(self, hook.name)
            # remove this deepcopy method, restoring the object's original one if necessary
            __deepcopy__ = self.__deepcopy__
            if orig_deepcopy:
                self.__deepcopy__ = orig_deepcopy
            else:
                del self.__deepcopy__
            # actually do the copy
            result = copy.deepcopy(self)
            # restore weights and method on self
            for name, value in weights.items():
                setattr(self, name, value)
            self.__deepcopy__ = __deepcopy__
            return result
        # bind __deepcopy__ to the weightnorm'd layer
        self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)


class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                WNLinear(in_dim, out_dim, wnorm=ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FactorizedSpectralConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, D1, D2, D3, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [self.modes_x, self.modes_y, self.modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        x = self.forward_fourier(x)

        x = x.permute(0, 2, 3, 4, 1)
        b = self.backcast_ff(x)
        b = b.permute(0, 4, 1, 2, 3)
        if self.use_fork:
            f = self.forecast_ff(x)
            f = f.permute(0, 4, 1, 2, 3)
        else:
            f = None
        return b, f

    def forward_fourier(self, x):
        B, I, S1, S2, S3 = x.shape

        # Dimension Z
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_ftz.new_zeros(B, self.out_dim, self.D1, self.D2, self.D3 // 2 + 1)
        
        out_ft[:, :, :min(S1,self.D1), :min(S2,self.D2), :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))[:, :, :min(S1,self.D1), :min(S2,self.D2), :]

        xz = torch.fft.irfft(out_ft, n=self.D3, dim=-1, norm='ortho')

        # Dimension Y
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_fty.new_zeros(B, self.out_dim, self.D1, self.D2 // 2 + 1, self.D3)
        
        out_ft[:, :, :min(S1,self.D1), :self.modes_y, :min(S3,self.D3)] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]))[:, :, :min(S1,self.D1), :, :min(S3,self.D3)]

        xy = torch.fft.irfft(out_ft, n=self.D2, dim=-2, norm='ortho')
        
        # Dimension X
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        out_ft = x_ftx.new_zeros(B, self.out_dim, self.D1 // 2 + 1, self.D2, self.D3)
        
        out_ft[:, :, :self.modes_x, :min(S2,self.D2), :min(S3,self.D3)] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))[:, :, :, :min(S2,self.D2), :min(S3,self.D3)]

        xx = torch.fft.irfft(out_ft, n=self.D1, dim=-3, norm='ortho')
        
        # Combining Dimensions
        x = xx + xy + xz

        return x


class ModifyDimensions3d(nn.Module):
    ''' Modify the shape of input from batch, in_width, S1, S2, S3, in_width to batch, out_width, dim1, dim2, dim3 '''
    def __init__(self, dim1, dim2, dim3, in_width, out_width):
        super(ModifyDimensions3d,self).__init__()
        self.conv = nn.Conv3d(in_width, out_width, 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self, x):
        x = self.conv(x) # modify the number of channels
        
        ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm='forward')
        d1 = min(ft.shape[2]//2, self.dim1//2)
        d2 = min(ft.shape[3]//2, self.dim2//2)
        d3 = max(min(ft.shape[4]//2, self.dim3//2), 1)
        ft_u = torch.zeros((ft.shape[0], ft.shape[1], self.dim1, self.dim2, self.dim3//2+1), dtype=torch.cfloat, device=ft.device)
        ft_u[:, :, :d1, :d2, :d3] = ft[:, :, :d1, :d2, :d3]
        ft_u[:, :, -d1:, :d2, :d3] = ft[:, :, -d1:, :d2, :d3]
        ft_u[:, :, :d1, -d2:, :d3] = ft[:, :, :d1, -d2:, :d3]
        ft_u[:, :, -d1:, -d2:, :d3] = ft[:, :, -d1:, -d2:, :d3]
        x_out = torch.fft.irfftn(ft_u, s=(self.dim1, self.dim2, self.dim3), norm='forward')

        return x_out
