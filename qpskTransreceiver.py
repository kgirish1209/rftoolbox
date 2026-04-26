import numpy as np
import matplotlib.pyplot as plt
import modulationUtil as modUtil
import rfutil as rfutil
import math
import channelModeling as channelModeling


def getQPSKModulatedSignal(symbol, ipPulseShape, LOFrequency, samplingFreq, transmitedPowerOfSymbol_dbM):
    transmittedSymbolPassBandPerSym = np.array([])  # Initialize transmitted signal array
    transmittedSymbolBasebandPerSym = np.array([])  # Initialize baseband signal array
    inPhaseComponent = np.real(symbol) * ipPulseShape  # Scale pulse by in-phase component
    quadratureComponent = np.imag(symbol) * ipPulseShape  # Scale pulse by quadrature component
    inPhaseUpconverted = inPhaseComponent * np.cos(2 * np.pi * LOFrequency * np.arange(len(inPhaseComponent)) / samplingFreq)  # Upconvert in-phase
    quadratureUpconverted = quadratureComponent * np.sin(2 * np.pi * LOFrequency * np.arange(len(quadratureComponent)) / samplingFreq)  # Upconvert quadrature
    transmittedSymbolBasebandPerSym = inPhaseUpconverted + 1j*quadratureUpconverted  # Combine components
    transmittedSymbolPassBandPerSym = rfutil.powerScale((inPhaseUpconverted - quadratureUpconverted),transmitedPowerOfSymbol_dbM)  # Passband signal (I-Q combination)
    return transmittedSymbolPassBandPerSym, transmittedSymbolBasebandPerSym

#receive QPSK signal and downconvert to baseband
def getQPSKDemodulatedSignal(receivedSignalPassBandNoisy, inPhaseSymbolRate, LOFrequency, samplingFreq):
    receivedInPhaseComponent = 2 * receivedSignalPassBandNoisy * np.cos(2 * np.pi * LOFrequency * np.arange(len(receivedSignalPassBandNoisy)) / samplingFreq)
    receivedQuadratureComponent = 2 * receivedSignalPassBandNoisy * np.sin(2 * np.pi * LOFrequency * np.arange(len(receivedSignalPassBandNoisy)) / samplingFreq)
    
    # Note: Applying a narrow LPF here distorts the rectangular pulse. 
    # The Matched Filter itself acts as the optimal filter.
    receivedInPhaseComponentFiltered = receivedInPhaseComponent
    receivedQuadratureComponentFiltered = receivedQuadratureComponent
    return receivedInPhaseComponentFiltered, receivedQuadratureComponentFiltered


def getmatchedFilterOutput(ipSignal, ipPulseShape):
    matchedFilter = np.conj(ipPulseShape[::-1]) # Matched filter (same as pulse shape)
    matchedFilter = matchedFilter / np.sum(np.abs(matchedFilter)**2)  # Normalize matched filter
    matchedFilterOutput = np.convolve(ipSignal, matchedFilter, mode='valid')  # Matched filter output
    return matchedFilterOutput
#per symbol do matched filtering and symbol decision

def getrawIQSamples(receivedInPhaseComponentFiltered, receivedQuadratureComponentFiltered,ipPulseShape):
    rawIQSamples = []
    receivedSymbolSegmentInPhase = getmatchedFilterOutput(receivedInPhaseComponentFiltered, ipPulseShape) # Extract symbol segment
    receivedSymbolSegmentQuadrature = getmatchedFilterOutput(receivedQuadratureComponentFiltered, ipPulseShape)  # Extract symbol segment
    rawIQSamples = receivedSymbolSegmentInPhase - 1j*receivedSymbolSegmentQuadrature  # Combine to get complex baseband symbol segment
    return rawIQSamples

def getDecidedSymbols(rawIQSamples):
    decidedSymbols = []
    for rawIQ in rawIQSamples:
        decidedSymbol = 1 + 1j if (np.real(rawIQ) > 0 and np.imag(rawIQ) > 0) else \
                        -1 + 1j if (np.real(rawIQ) < 0 and np.imag(rawIQ) > 0) else \
                        1 - 1j if (np.real(rawIQ) > 0 and np.imag(rawIQ) < 0) else \
                        -1 - 1j  # Decision based on quadrant
        decidedSymbols.append(decidedSymbol)
    return np.array(decidedSymbols)
#plt.show()
