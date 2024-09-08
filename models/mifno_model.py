import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_ffno import WNLinear, FactorizedSpectralConv3d, ModifyDimensions3d


class MIFNO_3D(nn.Module):
    def __init__(self, list_D1, list_D2, list_D3, list_M1, list_M2, list_M3, 
                 width, input_dim, output_dim, source_dim,
                 branching_index=4, n_layers=16, padding=0):
        ''' list_D1: list of ints: Number of dimensions along the 1st axis for each Fourier layer
        list_D2: list of ints: Number of dimensions along the 2nd axis for each Fourier layer
        list_D3: list of ints: Number of dimensions along the 3rd axis for each Fourier layer
        list_M1: list of ints: Number of Fourier modes along the 1st axis for each Fourier layer
        list_M2: list of ints: Number of Fourier modes along the 2nd axis for each Fourier layer
        list_M3: list of ints: Number of Fourier modes along the 3rd axis for each Fourier layer
        width: int: number of channels (created by the uplift layer)
        input_dim: int: number of dimensions of input (typically 4 when using a, x, y, z)
        output_dim: int: number of dimensions of input in each component (typically 1)
        source_dim: int: size of source variable (with angle: 3 coordinates + 3 angles = 6, with moment: 3 coordinates + 6 moment tensor components = 9)
        branching_index: int: index of layer where source and geology branches get concatenated
        n_layers: int: number of Fourier layers in geology branch + common branch
        padding: int: number of zeros to add on the horizontal dimensions to compensate for non-periodic input (does not seem necessary from our experiments)
        '''
        
        super().__init__()
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.source_dim = source_dim
        self.list_D1 = np.array(list_D1) + padding
        self.list_D2 = np.array(list_D2) + padding
        self.list_D3 = np.array(list_D3) # do not pad the vertical dimension
        self.n_layers = n_layers
        self.branching_index = branching_index
        self.padding = padding
        
        self.P = WNLinear(input_dim, self.width, wnorm=True)
        
        self.spectral_layers = nn.ModuleList([])
        self.modif_layers = nn.ModuleList([])
        for i in range(self.branching_index):
            self.spectral_layers.append(FactorizedSpectralConv3d(in_dim=width, out_dim=width, 
                                                                 D1=self.list_D1[i], D2=self.list_D2[i], D3=self.list_D3[i],
                                                                 modes_x=list_M1[i], modes_y=list_M2[i], modes_z=list_M3[i],
                                                                 forecast_ff=None, backcast_ff=None, fourier_weight=None, factor=4, 
                                                                 ff_weight_norm=True,
                                                                 n_ff_layers=2, layer_norm=False,
                                                                 use_fork=False, dropout=0.0))
            self.modif_layers.append(ModifyDimensions3d(self.list_D1[i], self.list_D2[i], self.list_D3[i], width, width))

        for i in range(self.branching_index, self.n_layers):
            self.spectral_layers.append(FactorizedSpectralConv3d(in_dim=3*width, out_dim=3*width, 
                                                                 D1=self.list_D1[i], D2=self.list_D2[i], D3=self.list_D3[i],
                                                                 modes_x=list_M1[i], modes_y=list_M2[i], modes_z=list_M3[i],
                                                                 forecast_ff=None, backcast_ff=None, fourier_weight=None, factor=4, 
                                                                 ff_weight_norm=True,
                                                                 n_ff_layers=2, layer_norm=False,
                                                                 use_fork=False, dropout=0.0))
            self.modif_layers.append(ModifyDimensions3d(self.list_D1[i], self.list_D2[i], self.list_D3[i], 3*width, 3*width))


        ### SOURCE BRANCH
        # size of the variables to concatenate with the part coming from the geology
        self.D1s = self.list_D1[self.branching_index-1]
        self.D2s = self.list_D2[self.branching_index-1]
        self.D3s = self.list_D3[self.branching_index-1] # variable created from the source will be width x D1s x D2s x D3s

        # weights cannot depend on the dimension of the geology, otherwise the FNO is no longer resolution independent
        self.T1 = 2*list_M1[self.branching_index-1]
        self.T2 = 2*list_M2[self.branching_index-1]
        self.T3 = 2*list_M3[self.branching_index-1]
        
        # goes from 3 coordinates to x*y
        self.MLP1 = nn.Sequential(
            nn.Linear(self.source_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.T1*self.T2))
        
        # create the z dimension
        self.Conv2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, self.T3, kernel_size=3, padding='same')
        )
        
        # create the features
        self.Conv3 = nn.Sequential(
            nn.Conv3d(1, self.width//2, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(self.width//2, self.width, kernel_size=3, padding='same')
        )

        self.modif_dim_source = ModifyDimensions3d(self.D1s, self.D2s, self.D3s, self.width, self.width)
        
        ### FINAL PROJECTIONS
        self.QE = nn.Sequential(
            WNLinear(3*self.width, 128, wnorm=True),
            WNLinear(128, output_dim, wnorm=True))
        
        self.QN = nn.Sequential(
            WNLinear(3*self.width, 128, wnorm=True),
            WNLinear(128, output_dim, wnorm=True))
        
        self.QZ = nn.Sequential(
            WNLinear(3*self.width, 128, wnorm=True),
            WNLinear(128, output_dim, wnorm=True))
        

    def forward(self, x, s):
        ''' x: geology, s: source '''
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.P(x)

        x = x.permute(0, 4, 1, 2, 3)
        if self.padding != 0:
            x = F.pad(x, [0, 0, 0, self.padding, 0, self.padding]) # pad only x and y
            
        # Geology branch
        for i in range(self.branching_index):
            layer_s = self.spectral_layers[i]
            layer_m = self.modif_layers[i]
            b, _ = layer_s(x)
            x = layer_m(x) + b

        # Source branch
        s1 = self.MLP1(s) # s1: _ x 1 x 1024
        s1 = s1.reshape(s1.shape[0], 1, self.T1, self.T2) # _ x 1 x 32 x 32
        
        s2 = self.Conv2(s1) # _ x 32 x 32 x 32
        s2 = torch.unsqueeze(s2, 1) # _ x 1 x 32 x 32 x 32
        
        s3 = self.Conv3(s2) # _ x 16 x 32 x 32 x 32
        s3 = self.modif_dim_source(s3)
        
        y = torch.cat((x + s3, x - s3, x*s3), dim=1)
    
        # Common branch
        for i in range(self.branching_index, self.n_layers):
            layer_s = self.spectral_layers[i]
            layer_m = self.modif_layers[i]
            b, _ = layer_s(y)
            y = layer_m(y) + b

        if self.padding != 0:
            b = b[..., :, :-self.padding, :-self.padding, :]
        
        yf = b
        yf = yf.permute(0, 2, 3, 4, 1)
        
        # Projection
        uE = self.QE(yf)
        uN = self.QN(yf)
        uZ = self.QZ(yf)

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
