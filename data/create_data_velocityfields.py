import shutil
import numpy as np
import h5py
from zipfile import ZipFile 
import argparse

from utils_data import butter_lowpass_filter


def rotate_velocity_90(uE, uN, uZ):
    ''' apply one rotation of 90° to the surface velocity fields, corresponding to 90° in the horizontal plane 

    uE: array (Nx, Ny, Nt)
    uN: array (Nx, Ny, Nt)
    uZ: array (Nx, Ny, Nt)
    '''

    uE_rot = np.rot90(-uN, -1, (0,1))
    uN_rot = np.rot90(uE, -1, (0,1))
    uZ_rot = np.rot90(uZ, -1, (0,1))

    return uE_rot, uN_rot, uZ_rot


parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('@Nval', type=int, default=3000, help="Number of validation samples")
parser.add_argument('@Ntest', type=int, default=1000, help="Number of test samples")
parser.add_argument('@S_in', type=int, default=32, help="Size of geology (in dimensions x and y)")
parser.add_argument('@S_in_z', type=int, default=32, help="Depth of geology")
parser.add_argument('@S_out', type=int, default=32, help="Number of sensors in dimensions x and y")
parser.add_argument('@Nt', type=int, default=320, help='Number of time steps in the output (only used for the folder name in this function)')
parser.add_argument('@f', type=int, default=50, help="Sampling frequency (in Hz)")
parser.add_argument('@fmax', type=int, default=5, help='Maximum frequency for filtering (only used for the folder name in this function)')
parser.add_argument('@path_raw', type=str, default='./raw/', help="Path to raw geologies *.npy")
parser.add_argument('@path_formatted', type=str, default='./formatted/', help="Path to formatted samples *.h5")
parser.add_argument('@rotation_angle', type=int, default=0, help="Rotation angle in ° to apply for data agumentation")
options = parser.parse_args().__dict__


S_in = options['S_in']
S_in_z = options['S_in_z']
S_out = options['S_out']
assert S_out == S_in, f"Are you sure that the resolution of velocity fields ({S_out}) should be different from the resolution of geologies ({S_in})? If yes, then you can comment this line"
Nt = options['Nt']
f = options['f'] # sampling frequency
assert 100 % f ==0, f'You asked a sampling frequency of {f}Hz that is not a divider of the recording frequency of 100Hz'
Ntrain = options['Ntrain']
Nval = options['Nval']
Ntest = options['Ntest']
fmax = options['fmax']
assert fmax <= 5, f'The mesh is designed up to a 5Hz frequency, you cannot request {fmax}Hz'
rotation_angle = options['rotation_angle']

path_raw = options['path_raw']
path_formatted = options['path_formatted']

folder_name = f"HEMEWS3D_S{S_in}_Z{S_in_z}_T{Nt}_fmax{fmax}_rot{rotation_angle}"

for I in range(0, Ntrain+Nval+Ntest, 100): # data are provided in batches of 100 samples
    # extract individual velocity fields
    with ZipFile(f"./raw/velocity{100000+I}-{100000+I+99}.zip", 'r') as zObject: 
        zObject.extractall("./raw/")
        
    for i in range(I, I+100):
        if i < Ntrain:
            folder_type = 'train'
        elif i < Ntrain+Nval:
            folder_type = 'val'
        else:
            folder_type = 'test'
        
        g = h5py.File(f"./raw/velocity{100000+I}-{100000+I+99}/sample{100000+i}.h5", 'r')
        for comp in ['uE', 'uN', 'uZ']:
            u = g[comp][:].reshape(S_out*S_out, -1) # reshape to S_out*S_out x 800
            u = butter_lowpass_filter(u, fmax, dt=0.01, order=4) # filter to remove high frequencies
            u = u.reshape(S_out, S_out, -1)
            u = u[:, :, ::int(100/f)] # temporal downscaling
            u = u[:, :, :Nt] # restrict the time window

            if comp == 'uE':
                uE = u.copy()
            elif comp == 'uN':
                uN = u.copy()
            else:
                uZ = u.copy()

        if rotation_angle >= 90:
            uE, uN, uZ = rotate_velocity_90(uE, uN, uZ)
        if rotation_angle >= 180:
            uE, uN, uZ = rotate_velocity_90(uE, uN, uZ)
        if rotation_angle == 270:
            uE, uN, uZ = rotate_velocity_90(uE, uN, uZ)
        
        # save data
        with h5py.File(f"{path_formatted}{folder_name}_{folder_type}/sample{i}.h5", 'a') as h:
            h.create_dataset('uE', data = uE, dtype=np.float32)
            h.create_dataset('uN', data = uN, dtype=np.float32)
            h.create_dataset('uZ', data = uZ, dtype=np.float32)
    
    shutil.rmtree(f"./raw/velocity{100000+I}-{100000+I+99}")
    if I % 1000 == 0:
        print(f'step {I}/{Ntrain+Nval+Ntest}')
