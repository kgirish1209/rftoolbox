#!/usr/local/bin/python3
# Author: Girish Kakalwar
# Date: April 26, 2026
# ============================================================================
# RF Signal Test Script
# Purpose: Simple test and verification script for RF signal generation,
#          time-domain analysis, and power measurement of single-tone signals
# ============================================================================

import math
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from rfWaveform import rfWaveform as rfw
import rfutil as rfutil

# ============================================================================
# SIGNAL PARAMETERS - Single Tone Configuration
# ============================================================================
# Define single-tone signal (for testing/verification purposes)
amplitude = list(np.ones(10))  # Single tone with amplitude = 1 V
phase = list(np.zeros(10))  # Phase = 0 radians
frequency = list(np.linspace(15000,150000,10,endpoint=False))  # Frequencies from 15 kHz to 150 kHz (10 tones)

# ============================================================================
# SAMPLING CONFIGURATION
# ============================================================================
samplingFreq = 30.7e6  # Sampling frequency (30.7 MHz)
timeduration = 1/min(frequency)  # Duration = 1 / frequency (one period at 15 kHz)
timewindow = np.linspace(0,timeduration,int(samplingFreq*timeduration),endpoint=False)  # Time vector
numofSamples = int(timeduration*samplingFreq)  # Total number of samples

# ============================================================================
# STEP 1: Generate single-tone signal
# ============================================================================
ipsignal = rfw(amplitude,frequency,phase,samplingFreq,numofSamples)

# ============================================================================
# STEP 2: Plot time-domain signal
# ============================================================================
#ipsignal.constructSignal(samplingFreq,timewindow)  # Optional: explicit signal construction
ipsignal.plotSignal("Input Signal","time","Amplitude",plt)  # Plot waveform
#plt.show()  # Uncomment to display plot

# ============================================================================
# STEP 3: Measure and print signal power
# ============================================================================
# Calculate total time-domain power and convert to dBm
signal_power_watts = rfutil.getSignalPower(ipsignal.sig)[0]  # Power in Watts
signal_power_dbm = rfutil.getLintodBM(signal_power_watts)  # Convert to dBm
print("Time domain power of the signal is ",signal_power_dbm)

filteredSignal = rfutil.lowPassFilter(ipsignal.sig, 10*frequency[0], samplingFreq, 1000)  # Bandpass filter around 15 kHz with 1 kHz bandwidth
rfutil.plotSignal(timewindow, filteredSignal, "Filtered Signal (15 kHz)", "Time (s)", "Amplitude", plt)
plt.show()  # Uncomment to display plot