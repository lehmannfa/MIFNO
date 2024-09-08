import pandas as pd
import numpy as np
import h5py
import argparse

from utils_data import compute_moment_tensor


def rotate_source_90(position, angle, moment):
    ''' apply one rotation of 90° in the horizontal plane to the source parameters

    position: array (3,1), coordinates x, y, z
    angle: array(3, 1), strike, dip, rake
    moment: array(6, 1), moment tensor coordinates Mxx, Myy, Mzz, Mxy, Mxz, Myz '''

    rot_position = np.array([9600-position[1],
                             position[0],
                             position[2]])
    rot_angle = np.array([(270+angle[0])%360,
                          angle[1],
                          angle[2]])
    rot_moment = np.array([moment[1],
                           moment[0],
                           moment[2],
                           -moment[3],
                           -moment[5],
                           moment[4]])
    return rot_position, rot_angle, rot_moment

parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('@Nval', type=int, default=3000, help="Number of validation samples")
parser.add_argument('@Ntest', type=int, default=1000, help="Number of test samples")
parser.add_argument('@S_in', type=int, default=32, help="Size of geology (in dimensions x and y)")
parser.add_argument('@S_in_z', type=int, default=32, help="Depth of geology")
parser.add_argument('@Nt', type=int, default=320, help='Number of time steps in the output (only used for the folder name in this function)')
parser.add_argument('@fmax', type=int, default=5, help='Maximum frequency for filtering (only used for the folder name in this function)')
parser.add_argument('@path_raw', type=str, default='./raw/', help="Path to source properties *.csv")
parser.add_argument('@path_formatted', type=str, default='./formatted/', help="Path to formatted samples *.h5")
parser.add_argument('@rotation_angle', type=int, default=0, help="Rotation angle in ° to apply for data augmentation")
options = parser.parse_args().__dict__

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

data = pd.read_csv(f'{path_raw}source_properties.csv', index_col=[0])

for i in range(Ntrain+Nval+Ntest):
    source_position = data.iloc[i,0:3].values.astype(np.float32)
    source_angle = data.iloc[i,3:6].values.astype(np.float32)
    source_moment = compute_moment_tensor(source_angle[0], source_angle[1], source_angle[2], 1)
    source_moment = np.array([*source_moment])

    if rotation_angle >= 90:
        source_position, source_angle, source_moment = rotate_source_90(source_position, source_angle, source_moment)
    if rotation_angle >= 180:
        source_position, source_angle, source_moment = rotate_source_90(source_position, source_angle, source_moment)
    if rotation_angle == 270:
        source_position, source_angle, source_moment = rotate_source_90(source_position, source_angle, source_moment)
    
    
    if i < Ntrain:
        folder_type = 'train'
    elif i < Ntrain+Nval:
        folder_type = 'val'
    else:
        folder_type = 'test'
    with h5py.File(f"{path_formatted}{folder_name}_{folder_type}/sample{i}.h5", 'a') as f:
        f.create_dataset('s', data=source_position)
        f.create_dataset('angle', data=source_angle)
        f.create_dataset('moment', data=source_moment)
