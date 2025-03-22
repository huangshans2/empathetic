from scipy.fft import rfft
import numpy as np

'''
Last revised on 20th Feb 2025
'''

def spectrum(sig,fs):
    '''
    Compute one-sided amplitude spectrum for a real-valued signal. Amplitude units are (data units * time)
    Last dimension should be time. Expect at most 2D data.
    sig: signal for which spectra are to be computed (1D or 2D array)
    fs: sampling frequency
    '''
    ## make sure signal is of even length (for good fft behavior)
    dim = len(sig.Samples.shape)
    N = sig.Samples.shape[-1]
    if N%2 != 0:
        if dim < 2:
            sig = np.pad(sig.Samples,((0,1)),'reflect')
        else:
            sig = np.pad(sig.Samples,((0,0),(0,1)),'reflect')
            
    N = sig.Samples.shape[-1]
    f = np.linspace( 0,fs/2,int(N/2+1) ) # output frequencies
    df = fs/N
    
    Y = np.abs(rfft(sig.Samples,axis=-1,norm='backward')) # make norm explicit lest future versions change default
    
    if dim < 2:
        Y[0 ] /= np.sqrt(2)
        Y[-1] /= np.sqrt(2)
    else:
        Y[:,0] /= np.sqrt(2)
        Y[:,-1] /= np.sqrt(2)
    Y *= 1/fs
    
    return f, Y

def bin_spectrum(f,Y,nbins,a=1.0,log=True):
    '''
    Expect 1D or 2D data with last axis being time. Function approximates input spectrum by binning into linear or octave bins.
    f: frequencies for computed spectra
    Y: computed spectra shaped (nf,) or (nx,nf): shoulf be rms spectrum from spectrum 
    nbins: maximum number of bins in an averaged spectrum. note that if log=True, the actual number of bins may be smaller.
    a: scalar for defining minimum frequency in the output spectrum
    log: False if want linear bins, True for octave bins. 
    Returns:
    fbins: bin edges. compute central frequencies via fnew = fbins[:-1]+np.diff(fbins)/2
    Ynew: binned spectrum 
    '''
    dim = len(Y.shape)
    df = f[1] - f[0] # frequency resolution
    fmin = a*df
    
    # determine bin edges. left-closed bins except last which is closed on both ends
    if log:
        fbins = np.logspace(np.log2(fmin),np.log2(f[-1]),num=int(nbins+1),base=2)
        # for log bins, it may happen that some bins are narrower than spectrum resolution. don't want empty bins => must make adjustments
        if np.any(np.diff(fbins)<=df):
            bin1 = np.where(np.diff(fbins)<=df)[0][-1]
            new_bin = np.array([fmin,fbins[bin1+1]])
            fbins = np.append(new_bin,fbins[(bin1+2):])
            nbins = len(fbins)-1

    else:
        fbins = np.linspace(fmin,f[-1],num=(int(nbins+1)))
    
    k = np.argmin(np.abs(f-fmin))
    if dim == 2:
        Ynew = np.zeros((Y.shape[0],nbins))
        for i in range(nbins):
            counter = 0
            if i+1 < nbins: # all but last bin
                while (f[k] >= fbins[i] and f[k] < fbins[i+1]):
                    Ynew[:,i] += Y[:,k]**2
                    counter += 1
                    k += 1
                if counter == 0: # if the first bin does not start at 0
                    k += 1
                else:
                    Ynew[:,i] = np.sqrt(Ynew[:,i]/counter)

            else: # last bin
                while f[k] >= fbins[i] and k < len(f)-1:
                    Ynew[:,i] += Y[:,k]**2
                    counter += 1
                    k += 1
                Ynew[:,i] = np.sqrt(Ynew[:,i]/counter)

    else:
        Ynew = np.zeros((nbins,))
        k = np.argmin(np.abs(f-fmin))
        for i in range(nbins):
            counter = 0
            if i+1 < nbins: # all but last bin
                while (f[k] >= fbins[i] and f[k] < fbins[i+1]):
                    Ynew[i] += Y[k]**2
                    counter += 1
                    k += 1
                if counter == 0: # if the first bin does not start at 0
                    k += 1
                else:
                    Ynew[i] = np.sqrt(Ynew[i]/counter)
            else: # last bin
                while f[k] >= fbins[i] and k < len(f)-1:
                    Ynew[i] += Y[k]**2
                    counter += 1
                    k += 1
                Ynew[i] = np.sqrt(Ynew[i]/counter)

    return fbins, Ynew

def spectrum2(sig,fs):
    '''
    Compute one-sided amplitude spectrum for a real-valued signal. Amplitude units are (data units * time)
    Last dimension should be time. Expect at most 2D data.
    sig: signal for which spectra are to be computed (1D or 2D array)
    fs: sampling frequency
    '''
    ## make sure signal is of even length (for good fft behavior)
    dim = len(sig.shape)
    N = sig.shape[-1]
    if N%2 != 0:
        if dim < 2:
            sig = np.pad(sig,((0,1)),'reflect')
        else:
            sig = np.pad(sig,((0,0),(0,1)),'reflect')
            
    N = sig.shape[-1]
    f = np.linspace( 0,fs/2,int(N/2+1) ) # output frequencies
    df = fs/N
    Y = np.abs(rfft(sig,axis=-1,norm='backward')) # make norm explicit lest future versions change default
    
    if dim < 2:
        Y[0 ] /= np.sqrt(2)
        Y[-1] /= np.sqrt(2)
    else:
        Y[:,0] /= np.sqrt(2)
        Y[:,-1] /= np.sqrt(2)
    Y *= 1/fs
    return f, Y
    
def ircor(sig,fs,f0,L,G0,fmin=None,fmax=None):
    '''
    last signal dimension should be time
    fs: sampling frequency
    f0: geophone resonant frequency
    L:  damping ratio
    G0: sensitiivity constant
    fmin: minimum frequency for applying the correction
    '''
    if fmin is None:
        fmin = 0.01 * f0
    if fmax is None:
        fmax = 0.5 * fs #Nyquist
    
    dims = sig.shape
    nt = dims[-1]
#     print(f'Computing IR correction for trace with {nt} samples.')
    if nt%2 != 0:
        sig = sig[:nt-1]
    
    # compute transfer function
    w  = 2*np.pi*np.linspace(0,fs/2,int(nt/2+1))
    w0 = 2*np.pi*f0
    
    ntap_lo = int(np.argmin(np.abs(w-2*np.pi*fmin))) # index of low frequency bound
    tap_hi_id = int(np.argmin(np.abs(w-2*np.pi*fmax)))
    ntap_hi = int(nt/2+1)-tap_hi_id # index of low frequency bound   
    
    tap_lo = np.sin(2*np.pi/(2*(2*ntap_lo)) * np.linspace(0,ntap_lo,ntap_lo + 1))**2
    tap_hi = np.sin(2*np.pi/(2*(2*ntap_hi)) * np.linspace(0,ntap_hi,ntap_hi + 1)+np.pi/2)**2
    
    H = G0 * w**2/(-w**2 + 2*1j*L*w0*w + w0**2)
    
    if len(dims) > 1:
        nx = 1
        for i in range(len(dims)-1):
            nx *= dims[i]
            
        sig.reshape((nx,nt),order='F')
        Y = rfft(sig,axis=-1,norm='forward')
        U = np.zeros(Y.shape,dtype=complex)
        U[:,1:] = Y[:,1:]/H[1:]
        U[:,:ntap_lo+1] *= tap_lo
        U[:,tap_hi_id-1:] *= tap_hi
#         for i in range(len(w)):
#             if w[i]/(2*np.pi) > fmin and w[i]/(2*np.pi) < fmax:
#                 U[:,i] = Y[:,i]/H[i]
#             elif w[i]/(2*np.pi) <= fmin:
#                 U[:,i] = Y[:,i]/H[i]
    else:

        Y = rfft(sig,axis=-1,norm='forward')
        U = np.zeros(Y.shape,dtype=complex)
        U[1:] = Y[1:]/H[1:]
        U[:ntap_lo+1] *= tap_lo
        U[tap_hi_id-1:] *= tap_hi
#         for i in range(len(w)):
#             if w[i]/(2*np.pi) > fmin and w[i]/(2*np.pi) < fmax:
#                 U[i] = Y[i]/H[i]
#             else:
#                 U[i] = Y[i]

    return irfft(U,axis=-1,norm='forward')

def window_sin(data,L,tap=None,mode='forward',unity=False):
    '''
    A function to split continuous trace into windows of specified length L and tapered on both edges using the squared cosine function.
    The windows are designed to overlap such that, the original trace can be reassembled (if unity=True).
    If unity=False, beginning and end of the trace also have a cos^2 window applied. 
    
    Parameters:
    data:  either a trace to be windowed or windowed data to reassemble the trace
    L:     length of the window in number of samples
    tap:   length of taper at each edge in number of samples. For now, only tap = L/2 is supported, but target is tap <= L/2
    mode:  'forward' to window data, 'adj' to reassemble the trace
    unity: if True, first and last windows are half square to preserve original trace amplitudes in reconstructions. If False, all windoew are cos^2.
    Last modified by Iga Pawelec on 17/09/2024
    '''
    
    if tap is None:
        tap = int(L/2)
    
    if mode == 'forward':
        nt = len(data)
        nwin = int(nt/(L-tap) - 1)
        window = np.sin(2*np.pi/(2*(2*tap)) * np.linspace(0,2*tap-1,2*tap))**2
        
        windows = np.zeros((L,nwin))
        if unity:
            ## First window
            windows[:,0] = data[0:(L-tap)+tap]
            windows[tap:2*tap,0] = data[tap:2*tap] * window[tap:2*tap]
            ## Last window
            windows[:,nwin-1] = data[(nwin-1)*(L-tap):nwin*(L-tap)+tap]
            windows[:tap,nwin-1] = data[(nwin-1)*(L-tap):nwin*(L-tap)] * window[:tap]
            ## All other windows
            for i in range(1,nwin-1):
                windows[:,i] = data[i*(L-tap):(i+1)*(L-tap)+tap] * window
        else:
            ## All windows are cos^2
            for i in range(nwin):
                windows[:,i] = data[i*(L-tap):(i+1)*(L-tap)+tap] * window
            
        return windows
            
    if mode == 'adj':
        
        nwin = data.shape[1]
        nt = (nwin+1) * (L-tap)
        
        trace = np.zeros((nt,))
        
        for i in range(nwin):
            trace[i*(L-tap):(i+1)*(L-tap)+tap] += data[:,i]
            
        return trace
    
def declip(trace,L,thr,dt):
    '''
    Set windowed parts of the trace to 0 according to a pre-selected RMS threshold.
    '''
    windows = window_sin(trace,L,tap=None,mode='forward')
    mask = np.sqrt(np.sum(windows**2,axis=0)/(dt*L))
    mask[mask==0] = 99999
    idx = np.where(mask > thr)
    mask = mask/mask
    mask[idx] = 0
    trace = window_sin(windows*mask,L,tap=None,mode='adj')
    return trace

def compute_rms(data,L,dt=0.001,f1=0,f2=None):
    '''
    Compute frequency-dependent RMS. 
    data: data samples 
    dt: temporal sampling in sec
    L: window length for RMS computations (in number of samples)
    f1: starting frequency for RMS computations
    f2: ending frequency for RMS computations
    '''
    if f2 is None:
        f2=0.5*1/dt
    nx = data.shape[0]
    nt = data.shape[1]
    nwin = int(nt/(L-int(L/2)) - 1)
    rms = np.zeros((nx,nwin))
    # find indices of min and max freqency
    frq = np.linspace( 0,1/(2*dt),int(L/2+1) )
    fmin = np.argmin(np.abs(frq-f1))
    fmax = np.argmin(np.abs(frq-f2))+1
#     print(f'fmin is {fmin} and fmax is {fmax}')
    #compute rms
    for ix in range(nx):
        # split trace into overlapping windows
        win = window_sin(data[ix,:].flatten(),L=L)
        # compute FFT with the right scaling
        _, Ywin = spectrum2(win.T,1/dt)
        # rms for selected band
        rms[ix,:] = np.sqrt(np.sum(2*Ywin[:,fmin:fmax]**2,axis=1)/dt)/(L*dt)
    return rms
