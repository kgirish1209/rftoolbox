"""
QPSK Transreceiver Simulation Script.

This script simulates the end-to-end communication chain for QPSK:
1. Bit generation and symbol mapping.
2. Pulse shaping and upconversion to passband.
3. AWGN channel effects.
4. Demodulation, matched filtering, and hard-decision decoding.
"""
import numpy as np
import matplotlib.pyplot as plt
import modulationUtil as modUtil
import rfutil as rfutil
import math
import channelModeling as channelModeling
import qpskTransreceiver as qpskTransreceiver

# Simulation Parameters
inputBitRate = 15e3             # Input bit rate (15 kbps)
LOFrequency = 2*inputBitRate    # Local oscillator frequency (30 kHz)
samplingFreq = 32*LOFrequency   # Sampling frequency (960 kHz)

inPhaseSymbolRate = inputBitRate / 2  # QPSK: 2 bits per symbol (7.5 ksps)
numOfSymbols = 10000  # Total number of symbols to transmit
symbolDuration = 1 / inPhaseSymbolRate  # Duration of each symbol (seconds)
numOfSamplesPerSymbol = int(samplingFreq * symbolDuration)  # Samples per symbol
timeDuration = numOfSymbols * symbolDuration  # Total duration of the signal
numOfSymbolsPerPlot = 1  # Number of symbols to plot in time domain
timeDurationPlot = numOfSymbolsPerPlot * symbolDuration  # Duration to plot (1 symbol duration)
timeVector = np.linspace(0, timeDuration, numOfSamplesPerSymbol*numOfSymbols, endpoint=False)  # Time vector for the signal
timeVectorPlot = np.linspace(0, timeDurationPlot, numOfSamplesPerSymbol*numOfSymbolsPerPlot, endpoint=False)  # Time vector for plotting
transmitedPowerOfSymbol_dbM = 0 # Power of each symbol (1 Watt)
ipPulseShape = rfutil.getRectangularPulse(numOfSamplesPerSymbol)  # Define the pulse shape

# Output simulation setup for verification
print("symbolDuration is  " + str(symbolDuration*1e6) + " in us")
print("timeDurationPlot is  " + str(timeDurationPlot*1e6) + " in us")
print("numOfSamplesPerSymbol is  " + str(numOfSamplesPerSymbol))

print("===========================================================")
print("LO Frequency is  " + str(LOFrequency/1e6) + " MHz")
print("Sampling Frequency is  " + str(samplingFreq/1e6) + " MHz")
print("Input Bit Rate is  " + str(inputBitRate/1e3) + " kbps")
print("In-phase Symbol Rate is  " + str(inPhaseSymbolRate/1e3) + " ksps")

# Initialize signal buffers
transmittedSignalPassBand =[]
transmittedSignalBaseband = []
rawModulatedIQSamples = np.array([])

# Link Eb/N0 to per-sample SNR for the channel model
eb_n0_db = 0  # Target energy per bit relative to noise power spectral density

# Calculation: SNR_sample = Eb/N0 - 10*log10(fs/Rb)
# This accounts for processing gain of the matched filter due to oversampling.
snr_db = eb_n0_db - 10 * math.log10(samplingFreq / inputBitRate)

DemodulatedIQSamples = np.array([])
rawDemodulatedIQSamples = np.array([])
ipBitstream = np.random.randint(0,2,2*numOfSymbols)  # Generate random bits

# Main simulation loop - processing one symbol at a time
for symbolIndex in range(0,numOfSymbols):
    # 1. Modulation: Map bits to QPSK symbols
    ipBitsym = ipBitstream[2*symbolIndex:2*symbolIndex+2]
    ipSymbol = modUtil.getSymbolMapping(ipBitsym,'QPSK')  # Map bits to QPSK symbol
    rawModulatedIQSamples = np.concatenate((rawModulatedIQSamples, ipSymbol))  # Append to total raw IQ samples
    
    # 2. Transmit: Upconvert to passband at LO Frequency
    [transmittedSignalPassBand,transmittedSignalBaseband] = qpskTransreceiver.getQPSKModulatedSignal(ipSymbol, ipPulseShape, LOFrequency, samplingFreq, transmitedPowerOfSymbol_dbM)
    
    # 3. Channel: Introduce noise (AWGN)
    receivedSignalPassBand = channelModeling.channelModel(transmittedSignalPassBand, snr_db,'AWGN')
    
    # 4. Demodulate: Product detection and filtering
    [receivedInPhaseComponentFiltered, receivedQuadratureComponentFiltered] = qpskTransreceiver.getQPSKDemodulatedSignal(receivedSignalPassBand, inPhaseSymbolRate, LOFrequency, samplingFreq)
    
    # 5. Receive filtering: Matched filter for signal recovery
    receivedInPhaseComponentFiltered = qpskTransreceiver.getmatchedFilterOutput(receivedInPhaseComponentFiltered, ipPulseShape)  # Matched filter output for in-phase
    receivedQuadratureComponentFiltered = qpskTransreceiver.getmatchedFilterOutput(receivedQuadratureComponentFiltered, ipPulseShape)  # Matched filter output for quadrature
    
    # 6. Decision logic: Hard decision on received IQ values
    receivedBasebandSignal = receivedInPhaseComponentFiltered - 1j*receivedQuadratureComponentFiltered
    rawDemodulatedIQSamples = np.concatenate((rawDemodulatedIQSamples, receivedBasebandSignal))  # Append to total raw demodulated IQ samples
    DemodulatedIQSamples = np.concatenate((DemodulatedIQSamples, qpskTransreceiver.getDecidedSymbols(receivedBasebandSignal)))  # Append to total raw IQ samples
    
    # Record signals for visualization of the transmission process
    if(symbolIndex==0):
        transmittedSignalPassBandTotal = transmittedSignalPassBand
        transmittedSignalBasebandTotal = transmittedSignalBaseband
        receivedBasebandSignalTotal = receivedBasebandSignal
    else:
        transmittedSignalPassBandTotal = np.concatenate((transmittedSignalPassBandTotal, transmittedSignalPassBand))  # Append to total passband signal
        transmittedSignalBasebandTotal = np.concatenate((transmittedSignalBasebandTotal, transmittedSignalBaseband))  # Append to total baseband signal
        receivedBasebandSignalTotal = np.concatenate((receivedBasebandSignalTotal, receivedBasebandSignal))  # Append to total received baseband signal
    
# Print data integrity checks
print("len(transmittedSignalPassBand) is  " + str(len(transmittedSignalPassBand)))
print("len(timeVectorPlot) is  " + str(len(timeVectorPlot)))
print("Total Power of transmitted signal in dBm: ", rfutil.getLintodBM(rfutil.getSignalPower(transmittedSignalPassBand)[0]))  # Print power of the signal
print("Length of raw modulated IQ samples: ", len(rawModulatedIQSamples))
print("Length of raw demodulated IQ samples: ", len(rawDemodulatedIQSamples))

# Final performance metrics calculation
outputBitstream = modUtil.getSymbolDemapping(DemodulatedIQSamples, 'QPSK')  # Map decided symbols back to bits
print("Input bitstream: ", ipBitstream)
print("Output bitstream: ", outputBitstream)
numBitErrors = np.sum(np.abs(ipBitstream - outputBitstream))  # Count bit errors
print("Number of bit errors: ", numBitErrors)
print("Bit Error Rate (BER): ", numBitErrors *1e3/ len(ipBitstream), " x 10^-3")

# Visualize results using constellation diagrams
modUtil.plotConstellation(np.real(rawModulatedIQSamples),np.imag(rawModulatedIQSamples), 'QPSK', plt)
modUtil.plotConstellation(np.real(rawDemodulatedIQSamples),np.imag(rawDemodulatedIQSamples), 'QPSK', plt)
