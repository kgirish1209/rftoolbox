#!/usr/local/bin/python3
# Author: Girish Kakalwar
# Date: April 26, 2026
# ============================================================================
# RF SSB Modulation/Demodulation Simulation
# Purpose: Demonstrate Single Sideband (SSB) modulation using the Hartley 
#          method (Hilbert Transform) and its subsequent demodulation.
# ============================================================================

import math
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from rfWaveform import rfWaveform as rfw
import rfutil as rfutil
from scipy import signal

# ============================================================================
# SIGNAL PARAMETERS - System Setup
# ============================================================================
symbolRate = 15e3             # Symbol rate (15 kHz)
pulseDuration = 2/symbolRate  # Total capture duration (2 symbols)
TxLOFreq = 32*symbolRate      # Carrier Frequency (480 kHz)
passBandsamplingFreq = 64*TxLOFreq     # High sampling rate for simulation accuracy
baseBandSamplingFreq = 128*symbolRate  # Reduced sampling rate for output
decimationFactor = int(passBandsamplingFreq/baseBandSamplingFreq) # Factor for sample reduction

# Define carrier signal properties
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
# STEP 1: Generate single-tone signal
# ============================================================================
txLOSignal = rfw(amplitude,frequency,phase,passBandsamplingFreq,numofSamples)
txLOSignalInPhase = txLOSignal.sig
phaseQ = list(-np.pi/2 * np.ones(1))
print(phaseQ)
txLOSignalQ = rfw(amplitude,frequency,phaseQ,passBandsamplingFreq,numofSamples)
txLOSignalQPhase = txLOSignalQ.sig

# ============================================================================
# STEP 2: Plot time-domain signal
# ============================================================================
#ipsignal.constructSignal(samplingFreq,timewindow)  # Optional: explicit signal construction
txLOSignal.plotSignal("Input Signal","time","Amplitude",plt)  # Plot waveform
#plt.show()  # Uncomment to display plot

# ============================================================================
# STEP 3: Measure and print signal power
# ============================================================================
# Calculate total time-domain power and convert to dBm
signal_power_watts = rfutil.getSignalPower(txLOSignal.sig)[0]  # Power in Watts
signal_power_dbm = rfutil.getLintodBM(signal_power_watts)  # Convert to dBm
print("Time domain power of the signal is ",signal_power_dbm)

[txLOFFTBin,txLOFFTSignal,txLOFFTSignalAmplSpect,txLOFFTSignalPhaseSpect] = rfutil.getFFTOfSignal(txLOSignal.sig,passBandsamplingFreq,len(txLOSignal.sig))
[txLOFFTQPhaseBin,txLOFFTQPhaseSignal,txLOFFTQPhaseSignalAmplSpect,txLOFFTQPhaseSignalPhaseSpect] = rfutil.getFFTOfSignal(txLOSignalQ.sig,passBandsamplingFreq,len(txLOSignalQ.sig))
plt.figure("Tx LO Signal")
plt.subplot(2,2,1)
rfutil.plotSignal(timewindow,txLOSignal.sig,"Inphase TXLO Signal","time","Amplitude",plt)
plt.subplot(2,2,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(txLOFFTBin),np.array(txLOFFTSignalAmplSpect),"Inphase TXLO Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
plt.subplot(2,2,3)
rfutil.plotSignal(timewindow,txLOSignalQ.sig,"QPhase Signal","time","Amplitude",plt)
plt.subplot(2,2,4)
rfutil.plotFFTAmplitudeSpectrum(np.array(txLOFFTQPhaseBin),np.array(txLOFFTQPhaseSignalAmplSpect),"Qphase Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========TXLO Signal Details")
print("Frequency : " + str(TxLOFreq/1e3) + " KHz\n" + "Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(txLOSignal.sig)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(txLOSignal.sig)/1e3) + " KHz")



################## Baseband Pulse Inphase Component#############
baseBandPulseInphase = np.array(np.ones(int(timeduration*passBandsamplingFreq)))
numBasebandTones = 3
amplitudeBaseband = list(np.ones(numBasebandTones))  # Single tone with amplitude = 1 V
phaseBaseband = list(np.zeros(numBasebandTones))  # Phase = 0 radians
frequencyBaseband = list(np.linspace(symbolRate,numBasebandTones*symbolRate,numBasebandTones,endpoint=True))
baseBandSignal = rfw(amplitudeBaseband,frequencyBaseband,phaseBaseband,passBandsamplingFreq,numofSamples)
analyticalSignal = signal.hilbert(baseBandSignal.sig)
baseBandPulseInphase = baseBandPulseInphase*baseBandSignal.sig
[baseBandPulseInphaseFFTBin,baseBandPulseInphaseFFT,baseBandPulseInphaseAmplSpect,baseBandPulseInphasePhaseSpect] = rfutil.getFFTOfSignal(baseBandPulseInphase,passBandsamplingFreq,len(baseBandPulseInphase))
plt.figure("Baseband Pulse Signal")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindow,baseBandPulseInphase,"Baseband Pulse Signal","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(baseBandPulseInphaseFFTBin),np.array(baseBandPulseInphaseAmplSpect),"Baseband Pulse Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========Baseband Pulse Signal Details")
print("Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(baseBandPulseInphase)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(baseBandPulseInphase)/1e3) + " KHz")


################## Baseband Pulse QPhase Component#############
baseBandPulseQPhase = np.array(np.ones(int(timeduration*passBandsamplingFreq)))
baseBandPulseQPhase = np.imag(analyticalSignal) * baseBandPulseQPhase
[baseBandPulseQPhaseFFTBin,baseBandPulseQPhaseFFT,baseBandPulseQPhaseAmplSpect,baseBandPulseQPhasePhaseSpect] = rfutil.getFFTOfSignal(baseBandPulseQPhase,passBandsamplingFreq,len(baseBandPulseQPhase))
plt.figure("Baseband Pulse Signal")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindow,baseBandPulseQPhase,"Baseband Pulse Signal","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(baseBandPulseQPhaseFFTBin),np.array(baseBandPulseQPhaseAmplSpect),"Baseband Pulse Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========Baseband Pulse Signal Details")
print("Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(baseBandPulseQPhase)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(baseBandPulseQPhase)/1e3) + " KHz")



################ InPhase MIx both the SIgnals ########
mixedTxInphaseSignal= txLOSignal.sig*baseBandPulseInphase
[mixedTxInphaseSignalFFTBin,mixedTxInphaseSignalFFT,mixedTxInphaseSignalAmplSpect,mixedTxInphaseSignalPhaseSpect] = rfutil.getFFTOfSignal(mixedTxInphaseSignal,passBandsamplingFreq,len(mixedTxInphaseSignal))
plt.figure("Mixed Tx Signal")
plt.subplot(2,2,1)
rfutil.plotSignal(timewindow,mixedTxInphaseSignal,"Mixed InPhase Signal","time","Amplitude",plt)
plt.subplot(2,2,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(mixedTxInphaseSignalFFTBin),np.array(mixedTxInphaseSignalAmplSpect),"Mixed Signal Inphase Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========Mixed Tx Inphase Signal Details")
print("Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(mixedTxInphaseSignal)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(mixedTxInphaseSignal)/1e3) + " KHz")

################ QPhase MIx both the SIgnals ########
mixedTxQphaseSignal= txLOSignalQ.sig*baseBandPulseQPhase
[mixedTxQphaseSignalFFTBin,mixedTxQphaseSignalFFT,mixedTxQphaseSignalAmplSpect,mixedTxQphaseSignalPhaseSpect] = rfutil.getFFTOfSignal(mixedTxQphaseSignal,passBandsamplingFreq,len(mixedTxQphaseSignal))
plt.subplot(2,2,3)
rfutil.plotSignal(timewindow,mixedTxQphaseSignal,"Mixed QPhase Signal","time","Amplitude",plt)
plt.subplot(2,2,4)
rfutil.plotFFTAmplitudeSpectrum(np.array(mixedTxQphaseSignalFFTBin),np.array(mixedTxQphaseSignalAmplSpect),"Mixed Signal QPhase Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)
print("==========Mixed Tx QPhase Signal Details")
print("Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(mixedTxQphaseSignal)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(mixedTxQphaseSignal)/1e3) + " KHz")

mixedTxSignal = mixedTxInphaseSignal - mixedTxQphaseSignal
[mixedTxSignalFFTBin,mixedTxSignalFFT,mixedTxSignalAmplSpect,mixedTxSignalPhaseSpect] = rfutil.getFFTOfSignal(mixedTxSignal,passBandsamplingFreq,len(mixedTxSignal))
plt.figure("Upconverted Signal ")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindow,mixedTxSignal,"Upconverted SSB Signal","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(mixedTxSignalFFTBin),np.array(mixedTxSignalAmplSpect),"Upconverted SSB Signal Amplitude Spectrum ","Frequency (Hz)","Amplitude",plt)


######## Receiver Specs########
RxLOFreq = TxLOFreq
rxLOSignal = np.cos(2*np.pi*RxLOFreq*timewindow)
print("==========RXLO Signal Details")
print("Frequency : " + str(RxLOFreq/1e3) + " KHz\n" + "Sampling Frequency : " + str(passBandsamplingFreq/1e6) + " MHz\n" + "NFFT : " + str(len(rxLOSignal)))
print("Freq Resolution is : " + str(passBandsamplingFreq/len(rxLOSignal)/1e3) + " KHz")

############ Downconvert The MIxed Signal #########
# Multiply received signal with local carrier to shift back to baseband
downConvertedSignal = mixedTxSignal * rxLOSignal
[downConvertedSignalFFTBin,downConvertedSignalFFT,downConvertedSignalAmplSpect,downConvertedSignalPhaseSpect] = rfutil.getFFTOfSignal(downConvertedSignal,passBandsamplingFreq,len(downConvertedSignal))

plt.figure("DownConverted Signal")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindow,downConvertedSignal,"Raw Downconverted Signal (Baseband + High Freq Image)","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(downConvertedSignalFFTBin),np.array(downConvertedSignalAmplSpect),"Downconverted Spectrum","Frequency (Hz)","Amplitude",plt)

############ Decimation and Recovery #########
# Decimation removes the high frequency image and restores the baseband sampling rate.
timewindowDecimated = np.linspace(0,timeduration,int(baseBandSamplingFreq*timeduration),endpoint=False)
decimatedSignal = signal.decimate(downConvertedSignal, int(decimationFactor))
[decimatedSignalFFTBin,decimatedSignalFFT,decimatedSignalAmplSpect,decimatedSignalPhase] = rfutil.getFFTOfSignal(decimatedSignal,baseBandSamplingFreq,len(decimatedSignal))

plt.figure("Recovered Signal")
plt.subplot(2,1,1)
rfutil.plotSignal(timewindowDecimated,decimatedSignal,"Recovered Baseband Message","time","Amplitude",plt)
plt.subplot(2,1,2)
rfutil.plotFFTAmplitudeSpectrum(np.array(decimatedSignalFFTBin),np.array(decimatedSignalAmplSpect),"Recovered Message Spectrum","Frequency (Hz)","Amplitude",plt)
plt.show()