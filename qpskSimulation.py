import numpy as np
import matplotlib.pyplot as plt
import modulationUtil as modUtil
import rfutil as rfutil
import math
import channelModeling as channelModeling
import qpskTransreceiver as qpskTransreceiver

#samplingFreq = 30.72e6/8  # Sampling frequency (30.72 MHz)
inputBitRate = 15e3  # Input bit rate (1 Mbps)
LOFrequency = 2*inputBitRate  # Local oscillator frequency (1 MHz)
samplingFreq = 32*LOFrequency  # Sampling frequency (30.72 MHz)

inPhaseSymbolRate = inputBitRate / 2  # QPSK: 2 bits per symbol
numOfSymbols = 10000  # Total number of symbols to transmit
symbolDuration = 1 / inPhaseSymbolRate  # Duration of each symbol
numOfSamplesPerSymbol = int(samplingFreq * symbolDuration)  # Samples per symbol
timeDuration = numOfSymbols * symbolDuration  # Total duration of the signal
numOfSymbolsPerPlot = 1  # Number of symbols to plot in time domain
timeDurationPlot = numOfSymbolsPerPlot * symbolDuration  # Duration to plot (1 symbol duration)
timeVector = np.linspace(0, timeDuration, numOfSamplesPerSymbol*numOfSymbols, endpoint=False)  # Time vector for the signal
timeVectorPlot = np.linspace(0, timeDurationPlot, numOfSamplesPerSymbol*numOfSymbolsPerPlot, endpoint=False)  # Time vector for plotting
transmitedPowerOfSymbol_dbM = 0 # Power of each symbol (1 Watt)

print("symbolDuration is  " + str(symbolDuration*1e6) + " in us")
print("timeDurationPlot is  " + str(timeDurationPlot*1e6) + " in us")
print("numOfSamplesPerSymbol is  " + str(numOfSamplesPerSymbol))


print("===========================================================")
print("LO Frequency is  " + str(LOFrequency/1e6) + " MHz")
print("Sampling Frequency is  " + str(samplingFreq/1e6) + " MHz")
print("Input Bit Rate is  " + str(inputBitRate/1e3) + " kbps")
print("In-phase Symbol Rate is  " + str(inPhaseSymbolRate/1e3) + " ksps")

transmittedSignalPassBand =[]
transmittedSignalBaseband = []
rawModulatedIQSamples = np.array([])
ipPulseShape = rfutil.getRectangularPulse(numOfSamplesPerSymbol)  # Rectangular pulse shape with samples per symbol

eb_n0_db = 0  # Target Eb/N0 in dB
# Conversion: SNR_sample = Eb/N0 - 10*log10(fs/Rb)
snr_db = eb_n0_db - 10 * math.log10(samplingFreq / inputBitRate)

DemodulatedIQSamples = np.array([])
rawDemodulatedIQSamples = np.array([])
ipBitstream = np.random.randint(0,2,2*numOfSymbols)  # Generate random bits
for symbolIndex in range(0,numOfSymbols):
    ipBitsym = ipBitstream[2*symbolIndex:2*symbolIndex+2]
    ipSymbol = modUtil.getSymbolMapping(ipBitsym,'QPSK')  # Map bits to QPSK symbol
    rawModulatedIQSamples = np.concatenate((rawModulatedIQSamples, ipSymbol))  # Append to total raw IQ samples
    [transmittedSignalPassBand,transmittedSignalBaseband] = qpskTransreceiver.getQPSKModulatedSignal(ipSymbol, ipPulseShape, LOFrequency, samplingFreq, transmitedPowerOfSymbol_dbM)#Gen
    receivedSignalPassBand = channelModeling.channelModel(transmittedSignalPassBand, snr_db,'AWGN')# Add AWGN noise to the transmitted signal
    #print("Received signal power in dBm: ", rfutil.getLintodBM(rfutil.getSignalPower(receivedSignalPassBand)[0]), "dBm")  # Print power of the received signal
    [receivedInPhaseComponentFiltered, receivedQuadratureComponentFiltered] = qpskTransreceiver.getQPSKDemodulatedSignal(receivedSignalPassBand, inPhaseSymbolRate, LOFrequency, samplingFreq)
    #print("Power of received in-phase component in dBm: ", rfutil.getLintodBM(rfutil.getSignalPower(receivedInPhaseComponentFiltered)[0]), "dBm")  # Print power of the received in-phase component
    #print("Power of received quadrature component in dBm: ", rfutil.getLintodBM(rfutil.getSignalPower(receivedQuadratureComponentFiltered)[0]), "dBm")  # Print power of the received quadrature component
    receivedInPhaseComponentFiltered = qpskTransreceiver.getmatchedFilterOutput(receivedInPhaseComponentFiltered, ipPulseShape)  # Matched filter output for in-phase
    receivedQuadratureComponentFiltered = qpskTransreceiver.getmatchedFilterOutput(receivedQuadratureComponentFiltered, ipPulseShape)  # Matched filter output for quadrature
    receivedBasebandSignal = receivedInPhaseComponentFiltered - 1j*receivedQuadratureComponentFiltered
    #print("receivedBasebandSignal: " + str(receivedBasebandSignal) + ", decided sym: " + str(qpskTransreceiver.getDecidedSymbols(receivedBasebandSignal)) + ", original sym: " + str(ipSymbol) + ", ipbitSym: " + str(ipBitsym))  # Print first 10 samples of the received baseband signal for the current symbol
    rawDemodulatedIQSamples = np.concatenate((rawDemodulatedIQSamples, receivedBasebandSignal))  # Append to total raw demodulated IQ samples
    DemodulatedIQSamples = np.concatenate((DemodulatedIQSamples, qpskTransreceiver.getDecidedSymbols(receivedBasebandSignal)))  # Append to total raw IQ samples
    if(symbolIndex==0):
        transmittedSignalPassBandTotal = transmittedSignalPassBand
        transmittedSignalBasebandTotal = transmittedSignalBaseband
        receivedBasebandSignalTotal = receivedBasebandSignal
    else:
        transmittedSignalPassBandTotal = np.concatenate((transmittedSignalPassBandTotal, transmittedSignalPassBand))  # Append to total passband signal
        transmittedSignalBasebandTotal = np.concatenate((transmittedSignalBasebandTotal, transmittedSignalBaseband))  # Append to total baseband signal
        receivedBasebandSignalTotal = np.concatenate((receivedBasebandSignalTotal, receivedBasebandSignal))  # Append to total received baseband signal
    
print("len(transmittedSignalPassBand) is  " + str(len(transmittedSignalPassBand)))
print("len(timeVectorPlot) is  " + str(len(timeVectorPlot)))
print("Total Power of transmitted signal in dBm: ", rfutil.getLintodBM(rfutil.getSignalPower(transmittedSignalPassBand)[0]))  # Print power of the signal
print("Length of raw modulated IQ samples: ", len(rawModulatedIQSamples))
print("Length of raw demodulated IQ samples: ", len(rawDemodulatedIQSamples))

outputBitstream = modUtil.getSymbolDemapping(DemodulatedIQSamples, 'QPSK')  # Map decided symbols back to bits
print("Input bitstream: ", ipBitstream)
print("Output bitstream: ", outputBitstream)
numBitErrors = np.sum(np.abs(ipBitstream - outputBitstream))  # Count bit errors
print("Number of bit errors: ", numBitErrors)
print("Bit Error Rate (BER): ", numBitErrors *1e3/ len(ipBitstream), " x 10^-3")

#plt.figure(figsize=(12,8))
#plt.subplot(3,1,1)
modUtil.plotConstellation(np.real(rawModulatedIQSamples),np.imag(rawModulatedIQSamples), 'QPSK', plt)
modUtil.plotConstellation(np.real(rawDemodulatedIQSamples),np.imag(rawDemodulatedIQSamples), 'QPSK', plt)
#plt.show()
