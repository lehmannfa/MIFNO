import h5py
import numpy as np
import scipy
import json
import torch
import timeit
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from utils_models import get_device, loss_criterion, RunningAverage
from ffno_model import FFNO_3D
from mifno_model import MIFNO_3D
from dataloaders import GeologyTracesSourceDataset
from utils_metrics import fourier_spectra

parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@dir_logs', type=str, default='../logs/', help="Path to folder to store loss and models")
parser.add_argument('@name_config', type=str, default="MIFNO3D-moment-dv16-16layers-branching4-S32-T320-learningrate0p0004-Ntrain27000-batchsize16-normedsource-normedtraces", help="Name of model to load")
parser.add_argument('@epochs', type=int, default=250, help="Number of epochs used to train the model")
parser.add_argument('@dir_data_test', type=str, nargs='+', default=['../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_test'], help="Name of folders with test data")
parser.add_argument('@Ntest', type=int, default=1000, help="Number of test samples")
parser.add_argument('@S', type=int, default=32, help="Horizontal size of inputs")
parser.add_argument('@T_out', type=int, default=320, help="Temporal length of outputs")
parser.add_argument('@list_D1', type=int, nargs='+', default=None)
parser.add_argument('@list_D2', type=int, nargs='+', default=None)
parser.add_argument('@list_D3', type=int, nargs='+', default=None)
parser.add_argument('@batch_size', type=int, default=16, help="Number of samples used in the predictions")
options = parser.parse_args().__dict__

dir_logs = options['dir_logs']
name_config = options['name_config']
epochs = options['epochs']
dir_data_test = options['dir_data_test']
Ntest = options['Ntest']
S_in = options['S']
S_in_z = options['S']
S_out = options['S']
T_out = options['T_out']

def myL1(x, axis=0):
    return np.mean(np.abs(x), axis=axis)

def myL2(x, axis=0):
    return np.sqrt(np.mean(x**2, axis=axis))
    

if __name__ == '__main__':
    with open(f"{dir_logs}models/architecture-{name_config}-epochs{epochs}.json", 'r') as param_file:
        params = json.load(param_file)

    if options['list_D1'] is not None:
        list_D1 = options['list_D1']
    else:
        list_D1 = np.array(params['architecture']['list D1']).astype(int)
        
    if options['list_D2'] is not None:
        list_D2 = options['list_D2']
    else:
        list_D2 = np.array(params['architecture']['list D2']).astype(int)
        
    if options['list_D3'] is not None:
        list_D3 = options['list_D3']
    else:
        list_D3 = np.array(params['architecture']['list D3']).astype(int)
    
        
    device = get_device()
    print(f"Using device {device}")
        
    ### DATA
    test_data = GeologyTracesSourceDataset(dir_data_test, 
                                           S_in=S_in,
                                           S_in_z=S_in_z,
                                           S_out=S_out,
                                           T_out=T_out,
                                           transform_a='normal', 
                                           N=Ntest, 
                                           orientation=params['data']['source orientation'],
                                           transform_position=params['data']['transform position'], 
                                           transform_angle='unit',
                                           transform_traces=params['data']['normalize traces'])
    
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=options['batch_size'],
                                              shuffle=False,
                                              num_workers=2)
    
    
    ### MODEL
    model_type = params['architecture']['model type']
    if model_type == 'MIFNO':
        model = MIFNO_3D(list_D1, 
                         list_D2,
                         list_D3,
                         np.array(params['architecture']['list M1']).astype(int), 
                         np.array(params['architecture']['list M2']).astype(int), 
                         np.array(params['architecture']['list M3']).astype(int), 
                         params['architecture']['dv'],
                         input_dim=params['architecture']['input dim'], 
                         output_dim=params['architecture']['output dim'], 
                         source_dim=params['architecture']['source dim'],
                         n_layers=params['architecture']['nlayers'], 
                         branching_index=params['architecture']['branching index'], 
                         padding=params['architecture']['padding']
                         )
        
    elif model_type == 'FFNO':
        model = FFNO_3D(list_D1, 
                        list_D2,
                        list_D3,
                        np.array(params['architecture']['list M1']).astype(int), 
                        np.array(params['architecture']['list M2']).astype(int), 
                        np.array(params['architecture']['list M3']).astype(int), 
                        int(params['architecture']['list dv'][0]),
                        input_dim=params['architecture']['input dim'], 
                        output_dim=params['architecture']['output dim'], 
                        list_width=np.array(params['architecture']['list dv']).astype(int),
                        n_layers=params['architecture']['nlayers'], 
                        padding=params['architecture']['padding']
                        )
        
    model.to(device)
    model.load_state_dict(torch.load(f"{dir_logs}models/bestmodel-{name_config}-epochs{epochs}.pt",
                                     map_location=device))
    model.eval()    
    
    
    ### COMPUTE PREDICTIONS
    all_uE = torch.zeros((Ntest, S_out, S_out, T_out), dtype=torch.float32, device=device)
    all_outE = torch.zeros((Ntest, S_out, S_out, T_out), dtype=torch.float32, device=device)

    all_uN = torch.zeros((Ntest, S_out, S_out, T_out), dtype=torch.float32, device=device)
    all_outN = torch.zeros((Ntest, S_out, S_out, T_out), dtype=torch.float32, device=device)

    all_uZ = torch.zeros((Ntest, S_out, S_out, T_out), dtype=torch.float32, device=device)
    all_outZ = torch.zeros((Ntest, S_out, S_out, T_out), dtype=torch.float32, device=device)

    all_norm_traces = torch.zeros((Ntest, 1, 1, 1))

    i = 0
    t1 = timeit.default_timer()
    with torch.no_grad():        
        for _ in test_loader:
            a = _[0].to(device)
            uE = _[1].to(device)
            uN = _[2].to(device)
            uZ = _[3].to(device)               
            s = _[4].to(device)
            norm_traces = _[5].to(device)
            outE, outN, outZ = model(a, s)

            all_uE[i:i+a.shape[0]] = uE[:, :, :, :, 0]
            all_outE[i:i+a.shape[0]] = outE[:, :, :, :, 0]
            all_uN[i:i+a.shape[0]] = uN[:, :, :, :, 0]
            all_outN[i:i+a.shape[0]] = outN[:, :, :, :, 0]
            all_uZ[i:i+a.shape[0]] = uZ[:, :, :, :, 0]
            all_outZ[i:i+a.shape[0]] = outZ[:, :, :, :, 0]
            all_norm_traces[i:i+a.shape[0]] = norm_traces.reshape(-1, 1, 1, 1)
            i += a.shape[0]
            
    t2 = timeit.default_timer()
    print(f"time to compute outputs: {t2 - t1:.4f}s")

    all_uE = all_uE.cpu().numpy()
    all_uN = all_uN.cpu().numpy()
    all_uZ = all_uZ.cpu().numpy()
    all_outE = all_outE.cpu().numpy()
    all_outN = all_outN.cpu().numpy()
    all_outZ = all_outZ.cpu().numpy()
    all_norm_traces = all_norm_traces.cpu().numpy()
    

    ### METRICS
    if dir_data_test[0][-5:] == 'train':
        data_type = '_train'
    elif dir_data_test[0][-3:] == 'val':
        data_type = '_val'
    elif dir_data_test[0][-4:] == 'test':
        data_type = '_test'
    else:
        data_type = ''
        
    # DataFrame to store results
    df = pd.DataFrame(index=np.arange(Ntest), columns=['rMAE', 'rRMSE', 'rFFTlow', 'rFFTmid', 'rFFThigh', 'EG', 'PG'], 
                      dtype=np.float)

    # Relative MAE and RMSE
    eps = 1e-2
    base_E = np.abs(all_uE) + eps
    base_N = np.abs(all_uN) + eps
    base_Z = np.abs(all_uZ) + eps
    
    rMAE = myL1(all_outE/base_E - all_uE/base_E, axis=-1)
    rMAE += myL1(all_outN/base_N - all_uN/base_N, axis=-1)
    rMAE += myL1(all_outZ/base_Z - all_uZ/base_Z, axis=-1)
    rMAE /= 3
    df.loc[:, 'rMAE'] = np.mean(rMAE, axis=(1,2)) # mean over sensors

    rRMSE = myL2(all_outE/base_E - all_uE/base_E, axis=-1)
    rRMSE += myL2(all_outN/base_N - all_uN/base_N, axis=-1)
    rRMSE += myL2(all_outZ/base_Z - all_uZ/base_Z, axis=-1)
    rRMSE /= 3
    df.loc[:, 'rRMSE'] = np.mean(rRMSE, axis=(1,2)) # mean over sensors


    # Remove the normalization constant
    all_uE = all_uE/all_norm_traces
    all_uN = all_uN/all_norm_traces
    all_uZ = all_uZ/all_norm_traces
    all_outE = all_outE/all_norm_traces
    all_outN = all_outN/all_norm_traces
    all_outZ = all_outZ/all_norm_traces

  
    # Frequency biases
    dt = 0.02
    low_freq_uE, mid_freq_uE, high_freq_uE = fourier_spectra(all_uE, (0, 1), (1, 2), (2, 5), dt=dt)
    low_freq_uN, mid_freq_uN, high_freq_uN = fourier_spectra(all_uN, (0, 1), (1, 2), (2, 5), dt=dt)
    low_freq_uZ, mid_freq_uZ, high_freq_uZ = fourier_spectra(all_uZ, (0, 1), (1, 2), (2, 5), dt=dt)
    low_freq_u = (low_freq_uE + low_freq_uN + low_freq_uZ)/3
    mid_freq_u = (mid_freq_uE + mid_freq_uN + mid_freq_uZ)/3
    high_freq_u = (high_freq_uE + high_freq_uN + high_freq_uZ)/3
    
    low_freq_outE, mid_freq_outE, high_freq_outE = fourier_spectra(all_outE, (0, 1), (1, 2), (2, 5), dt=dt)
    low_freq_outN, mid_freq_outN, high_freq_outN = fourier_spectra(all_outN, (0, 1), (1, 2), (2, 5), dt=dt)
    low_freq_outZ, mid_freq_outZ, high_freq_outZ = fourier_spectra(all_outZ, (0, 1), (1, 2), (2, 5), dt=dt)
    low_freq_out = (low_freq_outE + low_freq_outN + low_freq_outZ)/3
    mid_freq_out = (mid_freq_outE + mid_freq_outN + mid_freq_outZ)/3
    high_freq_out = (high_freq_outE + high_freq_outN + high_freq_outZ)/3

    fourier_bias_low = (low_freq_out - low_freq_u)/low_freq_u
    fourier_bias_mid = (mid_freq_out - mid_freq_u)/mid_freq_u
    fourier_bias_high = (high_freq_out - high_freq_u)/high_freq_u
    df.loc[:, 'rFFTlow'] = np.mean(fourier_bias_low, axis=(1,2))
    df.loc[:, 'rFFTmid'] = np.mean(fourier_bias_mid, axis=(1,2))
    df.loc[:, 'rFFThigh'] = np.mean(fourier_bias_high, axis=(1,2))

  
    # Statistics over samples
    summary = pd.DataFrame(index=['mean', 'std', 'q1', 'q3'], columns=df.columns, dtype=float)
    summary.loc['mean'] = df.mean(axis=0)
    summary.loc['std'] = df.std(axis=0)
    summary.loc['q1'] = df.quantile(0.25, axis=0)
    summary.loc['q3'] = df.quantile(0.75, axis=0)

    df = pd.concat([summary, df], axis=0)
    df.to_csv(f"{dir_logs}/metrics/metrics_{name_config}-epochs{epochs}{data_type}.csv")
    print(summary)


    ### Save outputs for goodness-of-fit calculations
    with h5py.File(f'{dir_logs}outputs/outputs-{name_config}-epochs{epochs}{data_type}.h5', 'w') as f:
        f.create_dataset('uE', data=all_uE)
        f.create_dataset('uN', data=all_uN)
        f.create_dataset('uZ', data=all_uZ)
        f.create_dataset('outE', data=all_outE)
        f.create_dataset('outN', data=all_outN)
        f.create_dataset('outZ', data=all_outZ)
