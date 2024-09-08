import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import h5py
import argparse

parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('@Nval', type=int, default=3000, help="Number of validation samples")
parser.add_argument('@Ntest', type=int, default=1000, help="Number of test samples")
parser.add_argument('@S_in', type=int, default=32, help="Size of geology (in dimensions x and y)")
parser.add_argument('@S_in_z', type=int, default=32, help="Depth of geology")
parser.add_argument('@Nt', type=int, default=320, help='Number of time steps in the output (only used for the folder name in this function)')
parser.add_argument('@fmax', type=int, default=5, help='Maximum frequency for filtering (only used for the folder name in this function)')
parser.add_argument('@interpolate', action='store_true', help="Whether to interpolate or keep the original dimensions")
parser.add_argument('@path_raw', type=str, default='./raw/', help="Path to raw geologies *.npy")
parser.add_argument('@path_formatted', type=str, default='./formatted/', help="Path to formatted samples *.h5")
parser.add_argument('@rotation_angle', type=int, default=0, help="Rotation angle in ° to apply for data augmentation: 0, 90, 180, 270")
options = parser.parse_args().__dict__

interpolate=options['interpolate']
interpolate_method = 'nearest'
S_in = options['S_in']
S_in_z = options['S_in_z']
Nt = options['Nt']
Ntrain = options['Ntrain']
Nval = options['Nval']
Ntest = options['Ntest']
fmax = options['fmax']
rotation_angle = options['rotation_angle']

path_raw = options['path_raw']
path_formatted = options['path_formatted']

folder_name = f"HEMEWS3D_S{S_in}_Z{S_in_z}_T{Nt}_fmax{fmax}_rot{rotation_angle}"
if not folder_name+'_train' in os.listdir('./formatted/'):
    os.mkdir('./formatted/'+folder_name+'_train')
if not folder_name+'_val' in os.listdir('./formatted/'):
    os.mkdir('./formatted/'+folder_name+'_val')
if not folder_name+'_test' in os.listdir('./formatted/'):
    os.mkdir('./formatted/'+folder_name+'_test')


for I in range(100000, 100000+Ntrain+Nval+Ntest, 2000):
    # Load geology matrices
    data_a = np.load(path_raw + f'material{I}-{I+1999}.npy')

    if rotation_angle == 90:
        data_a = np.rot90(data_a, 1, (1,2)) # 1 rotation in the plane defined by axes (1,2)
    
    elif rotation_angle == 180:
        data_a = np.rot90(data_a, 2, (1,2)) # 2 rotations in the plane defined by axes (1,2)
    
    elif rotation_angle == 270:
        data_a = np.rot90(data_a, 3, (1,2)) # 3 rotations in the plane defined by axes (1,2)

    if interpolate:    
        x = np.linspace(0, 9600, 32)
        x2 = np.linspace(0, 9600, S_in)
        y = np.linspace(0, 9600, 32)
        y2 = np.linspace(0, 9600, S_in)
        z = np.linspace(0, 9600, 32)
        z2 = np.linspace(0, 9600, S_in_z)

        data_a_interp = np.zeros((2000, S_in, S_in, S_in_z))
        
        for i in range(2000):
            points = pd.MultiIndex.from_product([x2, y2, z2])
            points = points.to_frame(name=['x','y','z']).values
            f_interp = RegularGridInterpolator((x, y, z), data_a[i, :, :, :], method=interpolate_method)
            data_a_interp[i, :, :, :] = f_interp(points).reshape(S_in, S_in, S_in_z)
        data_a = data_a_interp

    for i in range(2000):
        j = I-100000+i # real index of sample
        # Save geologies in individual .h5 files
        if j<Ntrain:
            with h5py.File(f'{path_formatted}{folder_name}_train/sample{j}.h5', 'w') as f:
                f.create_dataset('a', data=data_a[i].astype(np.float32))
                
        elif j<Ntrain+Nval:
            with h5py.File(f'{path_formatted}{folder_name}_val/sample{j}.h5', 'w') as f:
                f.create_dataset('a', data=data_a[i].astype(np.float32))
                
        elif j<Ntrain+Nval+Ntest:
            with h5py.File(f'{path_formatted}{folder_name}_test/sample{j}.h5', 'w') as f:
                f.create_dataset('a', data=data_a[i].astype(np.float32))
        
    if I==100000: 
        # Compute the mean and standard deviation of geologies to normalize the inputs
        # Only the first training data are used for normalization
        m = np.mean(data_a[:Ntrain], axis=0).astype(np.float32)
        np.save(f'{path_formatted}{folder_name}_train/a_mean.npy', m)
        np.save(f'{path_formatted}{folder_name}_val/a_mean.npy', m)
        np.save(f'{path_formatted}{folder_name}_test/a_mean.npy', m)
        
        s = np.std(data_a[:Ntrain], axis=0).astype(np.float32)
        np.save(f'{path_formatted}{folder_name}_train/a_std.npy', s)
        np.save(f'{path_formatted}{folder_name}_val/a_std.npy', s)
        np.save(f'{path_formatted}{folder_name}_test/a_std.npy', s)

    print(f'Geologies {I+2000}/{100000+Ntrain+Nval+Ntest} done')
