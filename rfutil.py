
#!/usr/local/bin/python3
# Author: Girish Kakalwar
# Date: April 26, 2026
# ============================================================================
# RF Signal Processing Utility Functions
# Purpose: Utility functions for signal analysis, power calculations, FFT,
#          and visualization for RF signal processing applications
# ============================================================================

import math
import numpy as np
from matplotlib import pyplot as plt

# ============================================================================
# CONSTANTS
# ============================================================================
RefR = 50  # Reference impedance in ohms (standard RF impedance)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

# ============================================================================
# FUNCTION: plotSignal
# Purpose: Plot time-domain signal with time axis in microseconds
# Parameters:
#   - time: Time vector (seconds)
#   - signal: Signal samples (voltage)
#   - title: Plot title
#   - xaxis_label: X-axis label
#   - yaxis_label: Y-axis label
#   - plt: Matplotlib pyplot object
# ============================================================================
def plotSignal(time,signal,title,xaxis_label,yaxis_label,plt):
    plt.title(title)
    plt.xlabel(xaxis_label+" in us")  # X-axis label with units (microseconds)
    plt.ylabel(yaxis_label)
    time_us = time*1e6  # Convert time from seconds to microseconds
    plt.plot(time_us,signal)  # Plot signal vs time

# ============================================================================
# FUNCTION: plotFFTAmplitudeSpectrum
# Purpose: Plot amplitude spectrum from FFT (linear magnitude scale)
# Parameters:
#   - fftFreq: Frequency bins from FFT
#   - amplitudeSpectrum: Magnitude spectrum (linear scale)
#   - title: Plot title
#   - xLabel: X-axis label (frequency)
#   - yLabel: Y-axis label (amplitude)
#   - plt: Matplotlib pyplot object
# Notes: Uses stem plot for discrete frequency components
# ============================================================================
def plotFFTAmplitudeSpectrum(fftFreq,amplitudeSpectrum,title,xLabel,yLabel,plt):
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.stem(fftFreq,amplitudeSpectrum)  # Stem plot for discrete frequencies
        plt.grid(True)
        #plt.show()

# ============================================================================
# FUNCTION: plotFFTPhaseSpectrum
# Purpose: Plot phase spectrum from FFT (radians)
# Parameters:
#   - fftFreq: Frequency bins from FFT
#   - phaseSpectrum: Phase values (radians)
#   - title: Plot title
#   - xLabel: X-axis label (frequency)
#   - yLabel: Y-axis label (phase in radians)
#   - plt: Matplotlib pyplot object
# Notes: Phase values are typically in radians (-π to π)
# ============================================================================
def plotFFTPhaseSpectrum(fftFreq,phaseSpectrum,title,xLabel,yLabel,plt):
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.stem(fftFreq,phaseSpectrum)  # Stem plot for discrete frequencies
        plt.grid(True)
        #plt.show()

# ============================================================================
# FUNCTION: plotFFTPowerSpectrum
# Purpose: Plot power spectrum in dBm with fixed y-axis range
# Parameters:
#   - fftFreq: Frequency bins from FFT
#   - powerSpectrum: Power in dBm
#   - title: Plot title
#   - xLabel: X-axis label (frequency)
#   - yLabel: Y-axis label (power in dBm)
#   - plt: Matplotlib pyplot object
# Notes: Y-axis range is fixed to -200 to 100 dBm (noise floor to saturation)
# ============================================================================
def plotFFTPowerSpectrum(fftFreq,powerSpectrum,title,xLabel,yLabel,plt):
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.stem(fftFreq,powerSpectrum)  # Stem plot for discrete frequencies
        plt.ylim(-200,100)  # Y-axis: -200 dBm (noise floor) to 100 dBm
        plt.grid(True)

# ============================================================================
# POWER CONVERSION FUNCTIONS
# ============================================================================

# ============================================================================
# FUNCTION: getdBmToLin
# Purpose: Convert power from dBm (logarithmic) to linear (Watts)
# Parameter: dbm_power - Power in dBm
# Returns: Power in Watts (linear)
# Equation: P_lin = 10^((P_dBm - 30) / 10)
# Notes: dBm is referenced to 1 mW (0 dBm = 1 mW = 0.001 W)
# ============================================================================
def getdBmToLin(dbm_power):
    return (10**((dbm_power-30)/10))

# ============================================================================
# FUNCTION: getLintodBM
# Purpose: Convert power from linear (Watts) to dBm (logarithmic)
# Parameter: lin_power - Power in Watts (linear scale)
# Returns: Power in dBm
# Equation: P_dBm = 10*log10(P_lin) + 30
# Notes: Inverse of getdBmToLin; 1 mW = 0 dBm
# ============================================================================
def getLintodBM(lin_power):
    return (10*math.log10(lin_power) + 30)

# ============================================================================
# SIGNAL POWER CALCULATION FUNCTIONS
# ============================================================================

# ============================================================================
# FUNCTION: getSignalPower
# Purpose: Calculate total, DC, and AC power of a signal
# Parameter: ipsignal - Input signal (time-domain samples)
# Returns: [total_power, dc_power, ac_power] in Watts
# Equations:
#   - DC Power: P_DC = (mean(signal))^2 / RefR
#   - AC Power: P_AC = var(signal) / RefR (where var = variance after removing DC)
#   - Total Power: P_total = P_DC + P_AC
# Notes:
#   - DC power is from the mean (offset) component
#   - AC power is from signal variations around the mean
#   - Variance = mean((x - mean(x))^2), which equals AC power*RefR
#   - RefR = 50 Ohm (standard RF impedance)
# ============================================================================
def getSignalPower(ipsignal):
    signal_power = 0
    # DC power: P_DC = (mean)^2 / R
    dc_signal_power = (np.mean(ipsignal)**2)/RefR
    
    # AC power: Remove DC component, compute variance, then normalize by impedance
    # Inner product: (sig - mean) · conj(sig - mean)
    # This gives the sum of squared magnitudes (for real signals, conj has no effect)
    # Divide by N (length) to get average, then divide by RefR for power
    ac_signal_power = ((ipsignal - np.mean(ipsignal)) @ (np.conj(ipsignal).T - np.mean(np.conj(ipsignal))))/(len(ipsignal)*RefR)
    
    # Total power = DC power + AC power
    signal_power = dc_signal_power + ac_signal_power
    return [signal_power,dc_signal_power,ac_signal_power]

# ============================================================================
# FUNCTION: powerScale
# Purpose: Scale input signal to achieve target power level
# Parameters:
#   - ipsignal: Input signal (time-domain samples)
#   - target_power_dbm: Desired output power in dBm
# Returns: scaled_signal - Signal scaled to target power
# Equations:
#   - Current power: P_current = getSignalPower(ipsignal)[0] (in Watts)
#   - Target power: P_target = getdBmToLin(target_power_dbm) (in Watts)
#   - Scaling factor: k = sqrt(P_target / P_current)
#   - Output: scaled_signal = k * input_signal
# Notes: Scaling preserves waveform shape, only changes amplitude
# ============================================================================
def powerScale(ipsignal,target_power_dbm):
    ipsignal_power_lin = getSignalPower(ipsignal)[0]  # Current power in Watts
    target_power_lin = getdBmToLin(target_power_dbm)  # Target power in Watts
    scaling_factor = math.sqrt(target_power_lin/ipsignal_power_lin)  # k = sqrt(P_target / P_current)
    scaled_signal = ipsignal * scaling_factor  # Scale all samples by factor k
    return scaled_signal

# ============================================================================
# FFT AND FREQUENCY ANALYSIS FUNCTIONS
# ============================================================================

# ============================================================================
# FUNCTION: getFFTOfSignal
# Purpose: Compute FFT and extract frequency-domain representations
# Parameters:
#   - signal: Input time-domain signal
#   - samplingFreq: Sampling frequency (Hz)
#   - Nfft: FFT length (number of bins)
# Returns: [frequency, fft_result, amplitude_spectrum, phase_spectrum]
#   - frequency: Frequency bins (-Fs/2 to Fs/2)
#   - fft_result: Complex FFT output
#   - amplitude_spectrum: Magnitude of FFT (linear scale)
#   - phase_spectrum: Phase of FFT (radians, -π to π)
# ============================================================================
def getFFTOfSignal(signal,samplingFreq,Nfft):
        fftsig = np.fft.fft(signal,Nfft)  # Compute FFT
        frequency = np.fft.fftfreq(Nfft,1/samplingFreq)  # Generate frequency bins
        amplitudeSpectrum = np.abs(fftsig)  # Extract magnitude (linear)
        phaseSpectrum = np.angle(fftsig)  # Extract phase (radians)
        return [frequency,fftsig,amplitudeSpectrum,phaseSpectrum]

# ============================================================================
# FUNCTION: getPowerOfFreqFromFFT
# Purpose: Extract power of a specific frequency component from FFT
# Parameters:
#   - fftFreq: Frequency bins from FFT
#   - amplitudeSpectrum: Magnitude spectrum from FFT
#   - freqOfInterest: Frequency to analyze (Hz)
# Returns: powerOfFreqOfInterest - Power in Watts
# Equation: P = 2 * (Amplitude^2) / (N^2 * RefR)
# Notes:
#   - Finds nearest FFT bin to requested frequency
#   - Factor of 2 accounts for one-sided spectrum (positive frequencies only)
#   - N = FFT length (len(amplitudeSpectrum))
#   - Used to extract power of specific tones or harmonics
# ============================================================================
def getPowerOfFreqFromFFT(fftFreq,amplitudeSpectrum,freqOfInterest):
    indexOfFreqOfInterest = np.argmin(np.abs(fftFreq-freqOfInterest))  # Find nearest bin
    powerOfFreqOfInterest = 2*(amplitudeSpectrum[indexOfFreqOfInterest]**2)/((len(amplitudeSpectrum)**2)*RefR)  # Power in Watts
    return powerOfFreqOfInterest