import numpy as np
import matplotlib.pyplot as plt
import modulationUtil as modUtil
import rfutil as rfutil
import math


#channel Modeling
#transmit through a channel with AWGN noise and fixed path loss of 0 dB
def channelModel(transmittedSignalPassBand, snr_db,channelType="AWGN"): 
    signal_power = rfutil.getSignalPower(transmittedSignalPassBand)[0]  # Power of the input signal in Watts
    snr_linear = 10**(snr_db/10)  # Convert SNR from dB to linear scale
    noise_power = signal_power / snr_linear  # Calculate noise power in Watts
    #print("Power of noise added in dBm: ", rfutil.getLintodBM(noise_power), "dBm")  # Print noise power in dBm
    noise = getAWGNNoise(transmittedSignalPassBand, noise_power)
    noisy_signal = transmittedSignalPassBand + noise  # Add noise to original signal
    #print("Total power of noisy signal in dBm: ", rfutil.getLintodBM(rfutil.getSignalPower(noisy_signal)[0]), "dBm")  # Print total power of noisy signal
    return noisy_signal

def getAWGNNoise(signal, noise_power):
    if(np.iscomplexobj(signal)):
        noise_std_dev = math.sqrt(noise_power*rfutil.RefR/2)  # Standard deviation of noise (assuming Gaussian)
        noise = (np.random.normal(0, noise_std_dev, len(signal)) +1j*np.random.normal(0, noise_std_dev, len(signal)) ) # Generate AWGN noise
    else:
        noise_std_dev = math.sqrt(noise_power*rfutil.RefR)  # Standard deviation of noise (assuming Gaussian)
        noise = np.random.normal(0, noise_std_dev, len(signal))  # Generate AWGN noise for real signals
    return noise

