#!/usr/local/bin/python3
# Author: Girish Kakalwar
# Date: April 26, 2026
# ============================================================================
# RF DSB-SC Modulation/Demodulation Script
# Purpose: Simulate Double Sideband Suppressed Carrier (DSB-SC) transmission,
#          including upconversion, downconversion, and signal recovery.
# ============================================================================

import math
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from rfWaveform import rfWaveform as rfw
import rfutil as rfutil
from scipy import signal

# ============================================================================
# SIGNAL PARAMETERS - Single Tone Configuration
# ============================================================================
symbolRate = 15e3
pulseDuration = 2/symbolRate
TxLOFreq = 32*symbolRate
passBandsamplingFreq = 64*TxLOFreq
baseBandSamplingFreq = 128*symbolRate
decimationFactor = int(passBandsamplingFreq/baseBandSamplingFreq)

# Define single-tone signal (for testing/verification purposes)
amplitude = list(np.ones(1))  # Single tone with amplitude = 1 V
phase = list(np.zeros(1))  # Phase = 0 radians
frequency = [TxLOFreq]

# ============================================================================
# SAMPLING CONFIGURATION
# ============================================================================

timeduration = pulseDuration # Duration = 1 / frequency (one period at 15 kHz)
timewindow = np.linspace(0,timeduration,int(passBandsamplingFreq*timeduration),endpoint=False)  # Time vector
numofSamples = int(timeduration*passBandsamplingFreq)  # Total number of samples

print("No of Samples are " + str(numofSamples))
# ============================================================================
# STEP 1: Generate Transmit LO Signal (Carrier)
# ============================================================================
txLOSignal = rfw(amplitude,frequency,phase,passBandsamplingFreq,numofSamples)

# ============================================================================
# STEP 2: Plot Carrier Signal in Time Domain
# ============================================================================
# txLOSignal.constructSignal(samplingFreq,timewindow)  # Optional: explicit signal construction
txLOSignal.plotSignal("Input Signal (Carrier)","time","Amplitude",plt)  # Plot waveform
#plt.show()  # Uncomment to display plot

# ============================================================================
# STEP 3: Measure and print signal power
# ============================================================================
# Calculate total time-domain power and convert to dBm
signal_power_watts = rfutil.getSignalPower(txLOSignal.sig)[0]  # Power in Watts
signal_power_dbm = rfutil.getLintodBM(signal_power_watts)  # Convert to dBm
print("Time domain power of the signal is ",signal_power_dbm)

# Frequency Domain Analysis of Transmit LO
[txLOFFTBin,txLOFFTSignal,txLOFFTSignalAmplSpect,txLOFFTSignalPhaseSpect] = rfutil.getFFTOfSignal(txLOSignal.sig,passBandsamplingFreq,len(txLOSignal.sig))
plt.figure("Tx LO Signal")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindow,txLOSignal.sig,"Carrier Signal","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(txLOFFTBin),np.array(txLOFFTSignalAmplSpect),"Carrier Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========TXLO Signal Details")
print("Frequency : " + str(TxLOFreq/1e3) + " KHz\n" + "Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(txLOSignal.sig)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(txLOSignal.sig)/1e3) + " KHz")


################## STEP 4: Generate Baseband Information Signal #############
baseBandPulse = np.array(np.ones(int(timeduration*passBandsamplingFreq)))
numBasebandTones = 3
amplitudeBaseband = list(np.ones(numBasebandTones))  # Single tone with amplitude = 1 V
phaseBaseband = list(np.zeros(numBasebandTones))  # Phase = 0 radians
frequencyBaseband = list(np.linspace(symbolRate,numBasebandTones*symbolRate,numBasebandTones,endpoint=True))
# Construct a multi-tone baseband signal
baseBandSignal = rfw(amplitudeBaseband,frequencyBaseband,phaseBaseband,passBandsamplingFreq,numofSamples)
baseBandPulse = baseBandPulse*baseBandSignal.sig
[baseBandPulseFFTBin,baseBandPulseFFT,baseBandPulseAmplSpect,baseBandPulsePhaseSpect] = rfutil.getFFTOfSignal(baseBandPulse,passBandsamplingFreq,len(baseBandPulse))
plt.figure("Baseband Pulse Signal")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindow,baseBandPulse,"Baseband Signal","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(baseBandPulseFFTBin),np.array(baseBandPulseAmplSpect),"Baseband Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========Baseband Pulse Signal Details")
print("Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(baseBandPulse)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(baseBandPulse)/1e3) + " KHz")


################ STEP 5: Upconversion (Mixing Baseband with Carrier) - DSB-SC ########
mixedTxSignal = txLOSignal.sig*baseBandPulse
[mixedTxSignalFFTBin,mixedTxSignalFFT,mixedTxSignalAmplSpect,mixedTxSignalPhaseSpect] = rfutil.getFFTOfSignal(mixedTxSignal,passBandsamplingFreq,len(mixedTxSignal))
plt.figure("Mixed Signal (DSB-SC)")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindow,mixedTxSignal,"Upconverted Signal","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(mixedTxSignalFFTBin),np.array(mixedTxSignalAmplSpect),"Upconverted Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========Mixed Signal Details")
print("Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(mixedTxSignal)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(mixedTxSignal)/1e3) + " KHz")


######## STEP 6: Receiver LO and Downconversion ########
RxLOFreq = TxLOFreq
rxLOSignal = np.cos(2*np.pi*RxLOFreq*timewindow)
print("==========RXLO Signal Details")
print("Frequency : " + str(RxLOFreq/1e3) + " KHz\n" + "Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(rxLOSignal)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(rxLOSignal)/1e3) + " KHz")

############ Product detection: Multiply received signal with local carrier #########
downConvertedSignal = mixedTxSignal * rxLOSignal
[downConvertedSignalFFTBin,downConvertedSignalFFT,downConvertedSignalAmplSpect,downConvertedSignalPhaseSpect] = rfutil.getFFTOfSignal(downConvertedSignal,passBandsamplingFreq,len(downConvertedSignal))
plt.figure("DownConverted Signal")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindow,downConvertedSignal,"Raw Downconverted Signal","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(downConvertedSignalFFTBin),np.array(downConvertedSignalAmplSpect),"Downconverted Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========DownConverted Signal Details")
print("Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(downConvertedSignal)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(downConvertedSignal)/1e3) + " KHz")


############ STEP 7: Decimation and Filtering for Baseband Recovery #########
timewindowDecimated = np.linspace(0,timeduration,int(baseBandSamplingFreq*timeduration),endpoint=False)
# Decimation effectively filters out high frequency products (at 2*Fc) and reduces sampling rate
decimatedSignal = signal.decimate(downConvertedSignal, int(decimationFactor))
[decimatedSignalFFTBin,decimatedSignalFFT,decimatedSignalAmplSpect,decimatedSignalPhase] = rfutil.getFFTOfSignal(decimatedSignal,baseBandSamplingFreq,len(decimatedSignal))
plt.figure("Decimated Signal (Recovered Baseband)")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindowDecimated,decimatedSignal,"Recovered Baseband Signal","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(decimatedSignalFFTBin),np.array(decimatedSignalAmplSpect),"Recovered Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========Decimated Signal Details")
print("Sampling Frequency : " + str(baseBandSamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(decimatedSignal)))
print("Freq Resolution is : " + str(baseBandSamplingFreq/len(decimatedSignal)/1e3) + " KHz")

plt.show()
