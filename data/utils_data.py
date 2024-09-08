import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd


def compute_moment_tensor(strike, dip, rake, M0):
    ''' From the fault angles and the seismic moment, compute the equivalent moment tensor using of the formula of [Aki and Richards, 1980] '''
    Mxx=np.sin(np.pi*dip/180)*np.cos(np.pi*rake/180)*np.sin(np.pi*2*strike/180)
    Mxx+=np.sin(2*np.pi*dip/180)*np.sin(np.pi*rake/180)*np.sin(np.pi*strike/180)**2
    # to avoid numerical issues due to machine's precision, we check the values before multipyling by usually high M0
    if np.abs(Mxx)<1e-14:
        Mxx=0
    Mxx=-M0*Mxx

    Mxy=np.sin(np.pi*dip/180)*np.cos(np.pi*rake/180)*np.cos(np.pi*2*strike/180)
    Mxy+=0.5*np.sin(2*np.pi*dip/180)*np.sin(np.pi*rake/180)*np.sin(np.pi*2*strike/180)
    if np.abs(Mxy)<1e-14:
        Mxy=0
    Mxy=M0*Mxy
    
    Mxz=np.cos(np.pi*dip/180)*np.cos(np.pi*rake/180)*np.cos(np.pi*strike/180)
    Mxz+=np.cos(2*np.pi*dip/180)*np.sin(np.pi*rake/180)*np.sin(np.pi*strike/180)
    if np.abs(Mxz)<1e-14:
        Mxz=0
    Mxz=-M0*Mxz

    Myy=np.sin(np.pi*dip/180)*np.cos(np.pi*rake/180)*np.sin(np.pi*2*strike/180)
    Myy-=np.sin(2*np.pi*dip/180)*np.sin(np.pi*rake/180)*np.cos(np.pi*strike/180)**2
    if np.abs(Myy)<1e-14:
        Myy=0
    Myy=M0*Myy

    Myz=np.cos(np.pi*dip/180)*np.cos(np.pi*rake/180)*np.sin(np.pi*strike/180)
    Myz-=np.cos(2*np.pi*dip/180)*np.sin(np.pi*rake/180)*np.cos(np.pi*strike/180)
    if np.abs(Myz)<1e-14:
        Myz=0
    Myz=-M0*Myz

    Mzz=M0*np.sin(np.pi*2*dip/180)*np.sin(np.pi*rake/180)
    
    return Mxx, Myy, Mzz, Mxy, Mxz, Myz


def butter_lowpass_filter(data, cutoff, dt=0.01, order=4):
    ''' data: array of pd.Series
    cutoff: cutoff frequency [Hz]
    dt: sampling step
    order: filter order '''
    if isinstance(data, pd.Series):
        idx = data.index
        
    fs = 1/dt # sample rate
    nyq = 0.5*fs # Nyquist frequency
    normal_cutoff = cutoff/nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y=filtfilt(b,a,data)
    
    if isinstance(data, pd.Series):
        return pd.Series(y, index=idx)
    
    return y
