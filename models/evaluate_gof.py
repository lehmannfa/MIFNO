from mpi4py import MPI
import numpy as np
import h5py
import pandas as pd
from obspy.signal import tf_misfit
import timeit
import argparse

parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@N', type=int, default=10, help="Number of elements to compute")
parser.add_argument('@dir_logs', type=str, default='../logs/', help="Path to log folder")
parser.add_argument('@name_outputs', type=str, help="Name of .h5 file containing reference timeseries and predictions")
parser.add_argument('@comp', type=str, default='all', help="Name of components to compute: all, E, N, or Z")
parser.add_argument('@fmin', type=int, default=0.01, help="Maximal frequency for GOF computation")
parser.add_argument('@fmax', type=int, default=5, help="Maximal frequency for GOF computation")
parser.add_argument('@dt', type=float, default=0.02, help="Time step (s)")
options = parser.parse_args().__dict__

def main():
    N = options['N'] # number of elements to compute
    dir_logs = options['dir_logs']
    name_outputs = options['name_outputs']
    fmin = options['fmin']
    fmax = options['fmax']
    c = options['comp']
    dt = options['dt']
    
    t1 = timeit.default_timer()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    file_output = h5py.File(dir_logs + 'outputs/' + name_outputs, 'r')
    S = file_output['uE'].shape[1] # number of sensors

    ## each processor computes the gof for its elements
    if N % size == 0:
        nb_elts_per_proc = N//size
    else:
        nb_elts_per_proc = N//size + 1 # give more work to the first processors
    startval = rank*nb_elts_per_proc
    endval = min(N, (rank+1)*nb_elts_per_proc)
    
    Egof_proc = np.zeros((endval - startval, S, S, 1))
    Pgof_proc = np.zeros((endval - startval, S, S, 1))
    for i in range(startval, endval):
        EG = np.zeros((S, S))
        PG = np.zeros((S, S))
        for iy_sensor in range(0, S, 1):
            for ix_sensor in range(0, S, 1): # we need to loop on sensors to apply the norm on the three components
                if c == 'all':
                    u = np.array([file_output['uE'][i, iy_sensor, ix_sensor, :],
                                  file_output['uN'][i, iy_sensor, ix_sensor, :],
                                  file_output['uZ'][i, iy_sensor, ix_sensor, :]])
                    out = np.array([file_output['outE'][i, iy_sensor, ix_sensor, :],
                                    file_output['outN'][i, iy_sensor, ix_sensor, :],
                                    file_output['outZ'][i, iy_sensor, ix_sensor, :]])
                else:
                    u = file_output[f'u{c}'][i, iy_sensor, ix_sensor, :]
                    out = file_output[f'out{c}'][i, iy_sensor, ix_sensor, :]
                
                try:
                    EG[iy_sensor, ix_sensor] = tf_misfit.eg(out, u, dt=dt, fmin=fmin, fmax=fmax, norm='global').mean() # global norm (default behaviour) normalises wrt all components, then compute the mean of all components
                    PG[iy_sensor, ix_sensor] = tf_misfit.pg(out, u, dt=dt, fmin=fmin, fmax=fmax, norm='global').mean()
                except:
                    print(f"cannot evaluate misfit for sample {i} and sensor iy={iy_sensor}")
        Egof_proc[i-startval, :, :, 0] = EG
        Pgof_proc[i-startval, :, :, 0] = PG

    if rank != 0: # all processors except root send their information to root
        comm.Send(np.concatenate([Egof_proc, Pgof_proc], axis=3), dest=0, tag=0)
    else:
        all_Egof = np.zeros((N, S, S))
        all_Egof[:nb_elts_per_proc] = Egof_proc[:, :, :, 0] # computed by proc 0
        all_Pgof = np.zeros((N, S, S))
        all_Pgof[:nb_elts_per_proc] = Pgof_proc[:, :, :, 0] # computed by proc 0
        
        i = nb_elts_per_proc
        for iproc in range(1, size):
            startval_proc = iproc*nb_elts_per_proc
            endval_proc = min(N, (iproc+1)*nb_elts_per_proc)
            nb_elts = endval_proc - startval_proc
            temp_gof = np.zeros((nb_elts, S, S, 2))
            comm.Recv(temp_gof, source=iproc, tag=0)
            all_Egof[i:i+nb_elts] = temp_gof[:, :, :, 0]
            all_Pgof[i:i+nb_elts] = temp_gof[:, :, :, 1]
            i += nb_elts

        # load .csv, add values and update summary
        df = pd.read_csv(f"{dir_logs}metrics/metrics_{name_outputs[8:-3]}.csv", index_col=[0])
        df.loc[np.arange(N).astype(str), 'EG'] = np.mean(all_Egof, axis=(1,2))
        df.loc[np.arange(N).astype(str), 'PG'] = np.mean(all_Pgof, axis=(1,2))
        df.loc['mean', ['EG', 'PG']] = df.loc[np.arange(N).astype(str), ['EG', 'PG']].mean(axis=0)
        df.loc['std', ['EG', 'PG']] = df.loc[np.arange(N).astype(str), ['EG', 'PG']].std(axis=0)
        df.loc['q1', ['EG', 'PG']] = df.loc[np.arange(N).astype(str), ['EG', 'PG']].quantile(0.25, axis=0)
        df.loc['q3', ['EG', 'PG']] = df.loc[np.arange(N).astype(str), ['EG', 'PG']].quantile(0.75, axis=0)
        df.to_csv(f"{dir_logs}metrics/metrics_{name_outputs[8:-3]}_GOF.csv")
        

    t2 = timeit.default_timer()
    print(f"[{rank}/{size}] took {t2-t1:.2f}s")
        
  
if __name__ == "__main__":
    main()
