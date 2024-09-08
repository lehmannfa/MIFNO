import re, os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class Normalization:
    ''' X: array (n, p) is a matrix of samples
    x: array (p,) is one sample to be normalized 
    y: array(p,) is one normalized sample that we want to denormalize
    '''
    
    def __init__(self, X, x_mean=None, x_std=None, norm_type=None, eps=1e-6):
        if x_mean is None:
            if norm_type=='log-normal':
                self.x_mean = np.mean(np.log(X))
            elif norm_type=='point-normal':
                self.x_mean = np.mean(X, axis=0) # mean over all samples, shape is the same as X[0]
            else:
                self.x_mean = X.mean()
        else:
            self.x_mean = x_mean
        
        if x_std is None:
            if norm_type=='log-normal':
                self.x_std = 4*np.std(np.log(X))
            elif norm_type=='point-normal':
                self.x_std = 4*np.std(X, axis=0)
            else:
                self.x_std = 4*X.std()
        else:
            self.x_std = x_std
            
        self.norm_type = norm_type
        self.eps = eps
    
    def forward(self, x):
        if self.norm_type == 'normal' or self.norm_type == 'point-normal':
            return (x - self.x_mean)/(self.x_std + self.eps)
        if self.norm_type == 'log':
            return np.log(x)
        if self.norm_type == 'log-normal':
            return (np.log(x) - self.x_mean)/self.x_std
        if self.norm_type is None:
            return x
    
    def inverse(self, y):
        if self.norm_type == 'normal' or self.norm_type == 'point-normal':
            return self.x_mean + self.x_std * y
        if self.norm_type == 'log':
            return np.exp(y)  
        if self.norm_type == 'log-normal':
            return np.exp(self.x_mean + self.x_std * y)
        if self.norm_type is None:
            return y
        

def norm_constant_distance_Vs(s, a, S_in=32, S_in_z=32, Dx=9600, Dy=9600, Dz=9600):
    ''' compute the normalization constant that depends on the velocity of shear waves at the source location and the source depth
    Input:
    s: array (3,): source coordinates
    a: array (S_in, S_in, S_in_z) or (S_in, S_in, S_in_z, 1): velocity of shear waves
    S_in: int: size of horizontal dimensions
    S_in_z: int: size of vertical dimension
    Dx: float: physical size along axis x (in meters)
    Dy: float: physical size along axis y (in meters)
    Dz: float: physical size along axis z (in meters)
    
    Output: float '''
    
    # reshape tensors with batch size of 1 to arrays without unnecessary dimensions
    s = s.flatten()
    if a.ndim==4 and a.shape[3] == 1:
        a = a[:, :, :, 0]

    hx = Dx/S_in # x size of a mesh element
    hy = Dy/S_in # y size of a mesh element
    hz = Dz/S_in_z # z size of a mesh element
        
    ix = int(s[0]//hx) # index of source on axis x
    iy = int(s[1]//hy) # index of source on axis y
    iz = int(S_in_z-1 + s[2]//hz) # index of source on axis z (reversed depth)
    
    Vs_source = a[ix, iy, iz] # shear wave velocity at the source location
    R = np.sqrt((1e-3*s[2])**2 + (1e-3*Dx/4)**2) # distance to the source
    
    norm_cst = 1e-8*Vs_source**2 * R 
    
    return norm_cst


class GeologyTracesSourceDataset(Dataset):
    def __init__(self, dir_data, T_out=320, S_in=32, S_in_z=32, S_out=32, 
                 transform_a='normal', N=None, orientation=None,
                 transform_angle=None, transform_position=None, transform_traces=None):
        ''' 
        dir_data: list of strings: path to folders where data are stored
        T_out: int: number of time steps in outputs
        S_in: int: number of grid points along x and y, in inputs
        S_in_z: int: number of grid points along z, in inputs
        S_out: int: number of grid points along x and y, in outputs
        N: int: number of elements to load
        transform_a: string ("normal" or "scalar_normal"): normalization method for inputs
        transform_angle: string (None or "unit"): normalization method for angle in inputs
        transform_position: array of three floats: size of x, y, z axes
        transform_traces: string (None or "distance_traces"): normalization method for traces
        orientation: string (None, "angle" or "moment"): indicates the orientation of the source
        '''
        self.dir_data = dir_data
        self.T_out = T_out
        self.S_in = S_in
        self.S_in_z = S_in_z
        self.S_out = S_out
        self.transform_a = transform_a
        self.transform_angle = transform_angle
        self.transform_position = transform_position
        self.transform_traces = transform_traces
        self.orientation = orientation

        a_mean = np.load(dir_data[0] + '/a_mean.npy')
        a_std = np.load(dir_data[0] + '/a_std.npy')

        if self.transform_a == 'scalar_normal':
            self.a_mean = np.mean(a_mean)
            self.a_std = np.mean(a_std)
        else:
            self.a_mean = a_mean[:self.S_in, :self.S_in, :self.S_in_z]
            self.a_std = a_std[:self.S_in, :self.S_in, :self.S_in_z]
        
        if self.transform_a == 'normal' or self.transform_a == 'scalar_normal':
            self.ANorm = Normalization(1, norm_type='normal', x_mean=self.a_mean, x_std=4*self.a_std) # the first argument has no influence since we are imposing the mean and std

        # list of all files
        self.all_files = []
        for indiv_dir_data in dir_data:
            l = os.listdir(indiv_dir_data)
            l = [item for item in l if item[:6] == 'sample']
            if len(l) == 0:
                raise Exception(f"folder {indiv_dir_data} is empty")
            l = sorted(l, key=lambda s: int(re.search(r'\d+', s).group()))
            self.all_files += [indiv_dir_data+'/'+li for li in l]

        # sort by index of samples (default sort groups by dir_data name first, and then sample index -> pb for rotations)
        list_numbers = [item[item.find('sample')+6:-3] for item in self.all_files]
        sorted_numbers = np.argsort(list_numbers)
        self.all_files = list(np.array(self.all_files)[sorted_numbers])

        if N is not None:
            self.all_files = self.all_files[:N]
        
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        f = h5py.File(self.all_files[idx], 'r')
        a = f['a'][:self.S_in, :self.S_in, :self.S_in_z]
        if self.transform_a is not None:
            a = self.ANorm.forward(a)
        a = np.expand_dims(a, axis=3)
        
        if 's' in f.keys():
            s_raw = f['s'][:]
        else:
            s_raw = np.array([4800, 4800, -8400]) # default source position in the middle of the bottom layer
            
        if self.transform_position is not None:
            s = s_raw/np.array(self.transform_position)
        else:
            s = s_raw
        s = s.astype(np.float32)
        
        if self.transform_traces == 'distance_Vs':
            norm_cst = norm_constant_distance_Vs(s_raw, f['a'][:])            
        else:
            norm_cst = 1
            
        if self.orientation == 'angle':
            if 'angle' in f.keys():
                ang = f['angle'][:].astype(np.float32)
            else:
                ang = np.array([48, 45, 88], dtype=np.float32) # default moment tensor

            if self.transform_angle == 'unit':
                ang[0] = (ang[0] - 180)/180
                ang[1] = (ang[1] - 45)/45
                ang[2] = (ang[2] - 180)/180

            s = np.concatenate([s, ang])             
                
        elif self.orientation == 'moment':
            s = np.concatenate([s, f['moment'][:].astype(np.float32)])
        s = np.expand_dims(s, axis=0)        
        
        uE = f['uE'][:self.S_out, :self.S_out, :self.T_out]
        uE = np.expand_dims(uE, axis=3)
        uE *= norm_cst
        
        uN = f['uN'][:self.S_out, :self.S_out, :self.T_out]
        uN = np.expand_dims(uN, axis=3)
        uN *= norm_cst
        
        uZ = f['uZ'][:self.S_out, :self.S_out, :self.T_out]
        uZ = np.expand_dims(uZ, axis=3)
        uZ *= norm_cst        

        f.close()

        return a, uE, uN, uZ, s, norm_cst
