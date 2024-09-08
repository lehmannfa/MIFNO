import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_ffno import WNLinear, FactorizedSpectralConv3d, ModifyDimensions3d


class FFNO_3D(nn.Module):
    def __init__(self, list_D1, list_D2, list_D3, list_M1, list_M2, list_M3, 
                 width, input_dim, output_dim, list_width=None,
                 n_layers=4, padding=0):
        super().__init__()
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.list_D1 = np.array(list_D1) + padding
        self.list_D2 = np.array(list_D2) + padding
        self.list_D3 = np.array(list_D3) # do not pad the vertical dimension
        if list_width is None:
            self.list_width = [width]*(self.n_layers+1)
        else:
            assert len(list_width) == self.n_layers+1, "You must include the first uplift dv and the final projection dv in list_width"
            self.list_width = list_width
        self.padding = padding # pad the domain if input is non-periodic
        
        self.P = WNLinear(input_dim, self.list_width[0], wnorm=True)
        
        self.spectral_layers = nn.ModuleList([])
        self.modif_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.spectral_layers.append(FactorizedSpectralConv3d(in_dim=self.list_width[i], out_dim=self.list_width[i+1], 
                                                                 D1=self.list_D1[i], D2=self.list_D2[i], D3=self.list_D3[i],
                                           modes_x=list_M1[i], modes_y=list_M2[i], modes_z=list_M3[i],
                                           forecast_ff=None, backcast_ff=None, fourier_weight=None, factor=4, 
                                           ff_weight_norm=True,
                                           n_ff_layers=2, layer_norm=False, use_fork=False, dropout=0.0))
            self.modif_layers.append(ModifyDimensions3d(self.list_D1[i], self.list_D2[i], self.list_D3[i],
                                                        self.list_width[i], self.list_width[i+1]))

        self.QE = nn.Sequential(
            WNLinear(self.list_width[-1], 128, wnorm=True),
            WNLinear(128, output_dim, wnorm=True))
        
        self.QN = nn.Sequential(
            WNLinear(self.list_width[-1], 128, wnorm=True),
            WNLinear(128, output_dim, wnorm=True))
        
        self.QZ = nn.Sequential(
            WNLinear(self.list_width[-1], 128, wnorm=True),
            WNLinear(128, output_dim, wnorm=True))

    def forward(self, x, s):
        ''' x: geology (batch, Dx, Dy, Dz, 1)
        s: source (batch, 1, Ds) where Ds = 6(3 coordinates + 3 angles) or Ds=9 (3 coordinates + 6 moment tensor components) '''
        grid = self.get_grid(x.shape, x.device)
        
        # transform vector of source parameters to 4D variable
        s = torch.unsqueeze(s, 2)
        s = torch.unsqueeze(s, 3)
        s = s.repeat(1,x.shape[1], x.shape[2], x.shape[3], 1)
        
        x = torch.cat((x, grid, s), dim=-1)
        x = self.P(x)

        x = x.permute(0, 4, 1, 2, 3)
        if self.padding != 0:
            x = F.pad(x, [0, 0, 0, self.padding, 0, self.padding]) # pad only x and y
            
        for i in range(self.n_layers):
            layer_s = self.spectral_layers[i]
            layer_m = self.modif_layers[i]
            b, _ = layer_s(x)            
            x = layer_m(x) + b

        if self.padding != 0:
            b = b[..., :, :-self.padding, :-self.padding, :]
        
        xf = b
        xf = xf.permute(0, 2, 3, 4, 1)
        
        uE = self.QE(xf)
        uN = self.QN(xf)
        uZ = self.QZ(xf)

        return uE, uN, uZ

    def get_grid(self, shape, device):
        batchsize, size1, size2, size3 = shape[0], shape[1], shape[2], shape[3]
        grid1 = torch.tensor(np.linspace(0, 1, size1), dtype=torch.float)
        grid1 = grid1.reshape(1, size1, 1, 1, 1).repeat(
            [batchsize, 1, size2, size3, 1])
        grid2 = torch.tensor(np.linspace(0, 1, size2), dtype=torch.float)
        grid2 = grid2.reshape(1, 1, size2, 1, 1).repeat(
            [batchsize, size1, 1, size3, 1])
        grid3 = torch.tensor(np.linspace(0, 1, size3), dtype=torch.float)
        grid3 = grid3.reshape(1, 1, 1, size3, 1).repeat(
            [batchsize, size1, size2, 1, 1])
        return torch.cat((grid1, grid2, grid3), dim=-1).to(device)
