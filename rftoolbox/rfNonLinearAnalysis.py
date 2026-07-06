
#!/usr/local/bin/python3
# Author: Girish Kakalwar
# Date: April 26, 2026
# ============================================================================
# RF Non-Linear Signal Analysis Tool
# Purpose: Generate multi-tone signals, apply non-linear transformations,
#          and analyze frequency-domain characteristics (harmonics, distortion)
# ============================================================================

import math
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from rfWaveform import rfWaveform as rfw
import rfutil as rfutil

# ============================================================================
# FUNCTION: getNonLinearSignal
# Purpose: Apply polynomial non-linear transformation to input signal
# Parameters:
#   - nlKernals: List of kernel coefficients [a1, a3, a5, ...]
#   - ip_signal: Input time-domain signal (real-valued array)
# Returns: nlSignal - Non-linear output signal
# Equation: output = a1*x + a3*x^3 + a5*x^5 + ...
# Notes: 
#   - a1 = linear gain coefficient
#   - a3 = third-order distortion (creates IMD3 products)
#   - Used to model amplifier compression and harmonic generation
# ============================================================================
def getNonLinearSignal(nlKernals,ip_signal):
    nlSignal=np.array(np.zeros(len(ip_signal)))
    # Sum all polynomial terms (odd-order kernels are typically used)
    for kernal in range(0,len(nlKernals)):
        nlSignal = nlSignal + nlKernals[kernal]*(ip_signal**(kernal+1))
    return nlSignal

# ============================================================================
# SIGNAL GENERATION PARAMETERS
# ============================================================================
# Multi-carrier signal configuration (OFDM-like structure)
numRBs = 1  # Number of resource blocks
numOfSubCarriersPerRB = 10  # Subcarriers per resource block
total_tones = numOfSubCarriersPerRB * numRBs  # Total number of tones

# Initialize signal parameters
amplitude = list(np.ones(total_tones))  # All tones: amplitude = 1 V
phase = list(np.zeros(total_tones))  # All tones: phase = 0 radians
# Frequency sweep: 15 kHz to 15*(numTones+1) kHz with uniform spacing
frequency = list(np.linspace(15000,15000*(total_tones+1),total_tones,endpoint=False))

# Sampling configuration
samplingFreq = 30.72e6  # Sampling frequency (30.72 MHz)
timeduration = 1/min(frequency)  # Signal duration = 1 / minimum frequency (periodic signal)
timewindow = np.linspace(0,timeduration,int(samplingFreq*timeduration),endpoint=False)  # Time vector
numofSamples = int(samplingFreq*timeduration)  # Total samples in duration

# ============================================================================
# STEP 1: Generate multi-tone input signal and plot
# ============================================================================
ipsignal = rfw(amplitude,frequency,phase,samplingFreq,numofSamples)
plt.subplot(2,1,1)
rfutil.plotSignal(ipsignal.timeref,ipsignal.sig,"Input Signal","time","Amplitude",plt)

# Print time-domain power analysis
print("Time domain power of the signal is ",rfutil.getLintodBM(rfutil.getSignalPower(ipsignal.sig)[0]))

# ============================================================================
# STEP 2: Scale input signal to target power level (0 dBm)
# ============================================================================
target_power_dbm = 0  # Target input power in dBm
scaled_signal = rfutil.powerScale(ipsignal.sig,target_power_dbm)  # Scale to target power
print("Time domain power of the scaled signal is ",rfutil.getLintodBM(rfutil.getSignalPower(scaled_signal)[0]),"dBm")
plt.subplot(2,1,2)
rfutil.plotSignal(timewindow,scaled_signal,"Scaled Signal","time","Amplitude",plt)

# ============================================================================
# STEP 3: Apply non-linear transformation to generate distortion products
# ============================================================================
# Non-linear kernel coefficients: [a1=linear gain, a3=third-order, a5=fifth-order, ...]
# a1=1: Linear gain (unity), a3=0: No third-order distortion (currently disabled)
nlKernals = [1,0,0]
nlSignal = getNonLinearSignal(nlKernals,scaled_signal)
print("Time domain power of the non linear signal is ",rfutil.getLintodBM(rfutil.getSignalPower(nlSignal)[0]),"dBm")

# ============================================================================
# STEP 4: Compute FFT of input signal and convert to power spectrum
# ============================================================================
# Get frequency-domain representation (FFT)
[freqbin,scaledSignalFFT,scaledSignalAmplSpect,scaledSignalPhaseSpect] = rfutil.getFFTOfSignal(scaled_signal,ipsignal.Fs,ipsignal.nFFT)

# Convert amplitude spectrum to power spectrum (in dBm)
# Power = (Amplitude^2) / (Impedance * N^2), where N = FFT length
scaledSignalPwrpectrum = (scaledSignalAmplSpect**2)/(rfutil.RefR*(ipsignal.nFFT)**2)
scaledSignalPwrpectrum_dbm = list(map(lambda x: rfutil.getLintodBM(x), scaledSignalPwrpectrum))

# Create mask to filter out noise below -200 dBm (noise floor)
mask = np.array(scaledSignalPwrpectrum_dbm)>-200

# ============================================================================
# STEP 5: Plot frequency-domain analysis (amplitude and power spectra)
# ============================================================================
plt.figure()  # Create new figure for frequency-domain plots
plt.subplot(2,1,1)  # Amplitude spectrum (linear scale)
rfutil.plotFFTAmplitudeSpectrum(np.array(freqbin)[mask],np.array(scaledSignalAmplSpect)[mask],"Output Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
plt.subplot(2,1,2)  # Power spectrum (dBm scale)
rfutil.plotFFTPowerSpectrum(np.array(freqbin)[mask],np.array(scaledSignalPwrpectrum_dbm)[mask],"Output Signal Power Spectrum ","Frequency (Hz)","Power (dBm)",plt)
plt.show()

# Print summary
print(numofSamples)