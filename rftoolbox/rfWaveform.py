#!/usr/local/bin/python3
# Author: Girish Kakalwar
# Date: April 26, 2026
# ============================================================================
# RF Waveform Generator and Analyzer
# Purpose: Create multi-tone RF signals and analyze their frequency components
# ============================================================================

import math
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
import rfutil as rfutil

# ============================================================================
# CLASS: rfWaveform
# Purpose: Represents and manipulates multi-tone RF signals
# Supports: Time-domain synthesis, frequency analysis, signal operations
# ============================================================================
class rfWaveform:
    # ========================================================================
    # CONSTRUCTOR: __init__
    # Purpose: Initialize multi-tone signal with specified parameters
    # Parameters:
    #   - amplitude: List of amplitudes for each tone
    #   - frequency: List of frequencies for each tone (Hz)
    #   - phase: List of phase values for each tone (radians)
    #   - samplingFreq: Sampling frequency (default: 1 MHz)
    #   - numofSamples: Number of time-domain samples (default: 1024)
    # ========================================================================
    def __init__(self, amplitude, frequency,phase,samplingFreq=1000000,numofSamples=1024):
        # Store input parameters as NumPy arrays
        self.amplitude = np.array(amplitude)  # Amplitude of each tone
        self.frequency = np.array(frequency)  # Frequency of each tone (Hz)
        self.phase = np.array(phase)  # Phase of each tone (radians)
        
        # Signal properties
        self.num_of_tones = len(amplitude)  # Number of tones in signal
        self.dc = False  # Flag: True if DC component exists
        self.single_tone = False  # Flag: True if single tone
        self.multi_tone = False  # Flag: True if multiple tones
        
        # Sampling and FFT parameters
        self.Fs = samplingFreq  # Sampling frequency (Hz)
        self.nFFT = numofSamples  # FFT length (same as number of samples)
        
        # Time-domain signal storage
        self.sig = np.array(np.zeros(numofSamples))  # Time-domain signal samples
        self.timeref = np.array(np.linspace(0,numofSamples/samplingFreq,numofSamples,endpoint=False))  # Time vector
        
        # Frequency-domain signal storage
        self.fftsig = np.array(np.zeros(numofSamples))  # FFT output (complex)
        self.fftFreq = np.array(np.linspace(-1*samplingFreq/2,samplingFreq/2,numofSamples,endpoint=False))  # Frequency axis (-Fs/2 to Fs/2)
        self.amplitudeSpectrum = np.array(np.zeros(numofSamples))  # Magnitude spectrum
        self.phaseSpectrum = np.array(np.zeros(numofSamples))  # Phase spectrum (radians)
        
        # Classify signal type based on frequency content
        if np.any(self.frequency) == 0:
            # DC component detected (frequency = 0)
            self.dc = self.amplitude[np.argwhere(self.frequency == 0)]
        if np.count_nonzero(self.frequency) == 1:
            # Only one non-zero frequency -> single tone
            self.single_tone = True
        else:
            # Multiple non-zero frequencies -> multi-tone
            self.multi_tone = True
        
        # Generate time-domain signal samples
        self.constructSignal(False)
        
    # ========================================================================
    # METHOD: constructSignal
    # Purpose: Generate time-domain signal samples from tone parameters
    # Parameters:
    #   - complex_signal_type: If True, use complex exponential; if False, use cosine
    # Output: Populates self.sig with computed waveform
    # Equation (real): sig(t) = Σ A_k * cos(2πf_k*t + φ_k)
    # Equation (complex): sig(t) = Σ A_k * exp(j(2πf_k*t + φ_k))
    # ========================================================================
    def constructSignal(self,complex_signal_type=False):
        self.sig = np.array(np.zeros(len(self.timeref)))  # Initialize output signal
        # Sum all tones to create multi-tone signal
        for tone in range(0,self.num_of_tones):
            if complex_signal_type:
                # Complex exponential representation (analytic signal)
                self.sig = self.sig + self.amplitude[tone] * np.exp(1j*(2*math.pi*self.frequency[tone]*self.timeref + self.phase[tone]))
            else:
                # Real-valued cosine representation (standard RF signal)
                self.sig = self.sig + self.amplitude[tone] * np.cos(2*math.pi*self.frequency[tone]*self.timeref + self.phase[tone])
    
    # ========================================================================
    # METHOD: mixSignals
    # Purpose: Multiply (mix) this signal with another signal
    # Parameters:
    #   - inputSignal: Signal to mix with (time-domain array)
    # Output: Returns mixed signal (element-wise multiplication)
    # Note: Used for frequency up/down-conversion, modulation/demodulation
    # ========================================================================
    def mixSignals(self,inputSignal):
        mixedSignal = np.array(np.zeros(len(self.timeref)))  # Initialize output
        mixedSignal = self.sig * inputSignal  # Element-wise multiplication
        return mixedSignal
        

    # ========================================================================
    # METHOD: addSignals
    # Purpose: Add or subtract this signal with another signal
    # Parameters:
    #   - inputSignal: Signal to combine with (time-domain array)
    #   - operation: "add" or "subtract" (default: "add")
    # Output: Returns combined signal
    # Note: Used for summing multiple signals or noise injection
    # ========================================================================
    def addSignals(self,inputSignal,operation="add"):
        newSignal = np.array(np.zeros(len(self.timeref)))  # Initialize output
        if operation == "add":
            newSignal = self.sig + inputSignal  # Signal addition
        elif operation == "subtract":
            newSignal = self.sig - inputSignal  # Signal subtraction
        else:
            print("Operation not supported. Adding signals by default")
            newSignal = self.sig + inputSignal  # Default to addition
        return newSignal
    
    
    # ========================================================================
    # METHOD: plotSignal
    # Purpose: Plot time-domain signal waveform
    # Parameters:
    #   - title: Title of the plot
    #   - xLabel: X-axis label
    #   - yLabel: Y-axis label
    #   - plt: Matplotlib pyplot object
    # Output: Displays plot (time axis in microseconds)
    # ========================================================================
    def plotSignal(self,title,xLabel,yLabel,plt):
        plt.title(title)
        plt.xlabel(xLabel + " in us")  # Time axis in microseconds
        plt.ylabel(yLabel)
        time_us = self.timeref*1e6  # Convert time from seconds to microseconds
        plt.plot(time_us,self.sig)  # Plot signal vs time
        #plt.show()
        
    # ========================================================================
    # METHOD: getFFT
    # Purpose: Compute FFT and store frequency-domain representation
    # Output: Updates self.fftFreq, self.fftsig, self.amplitudeSpectrum, self.phaseSpectrum
    # ========================================================================
    def getFFT(self):
        # Compute FFT using utility function
        [self.fftFreq ,self.fftsig,self.amplitudeSpectrum,self.phaseSpectrum] = rfutil.getFFTOfSignal(self.sig,self.Fs,self.nFFT)
    
    # ========================================================================
    # METHOD: plotFFT
    # Purpose: Plot amplitude and phase spectra (frequency-domain representation)
    # Parameters:
    #   - title: Title of the plot
    #   - xLabel: X-axis label (frequency)
    #   - yLabel: Y-axis label (amplitude/phase)
    #   - plt: Matplotlib pyplot object
    # Output: Creates 2-subplot figure with amplitude and phase spectrum
    # ========================================================================
    def plotFFT(self,title,xLabel,yLabel,plt):
        plt.figure("FFT of the signal")  # Create new figure
        plt.subplot(2,1,1)  # Amplitude spectrum (top)
        rfutil.plotFFTAmplitudeSpectrum(self.fftFreq,self.amplitudeSpectrum,title+" Amplitude Spectrum",xLabel,yLabel,plt)
        plt.subplot(2,1,2)  # Phase spectrum (bottom)
        rfutil.plotFFTPhaseSpectrum(self.fftFreq,self.phaseSpectrum,title+" Phase Spectrum",xLabel,yLabel,plt)


