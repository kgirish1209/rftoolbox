import numpy as np
import matplotlib.pyplot as plt
import rfutil as rfutil
import math


# ============================================================================
# CHANNEL MODELING FUNCTIONS
# ============================================================================

# FUNCTION: channelModel
# Purpose: Simulates a communication channel by adding Additive White Gaussian Noise (AWGN)
#          to the transmitted signal. Assumes a fixed path loss of 0 dB (no attenuation).
# Parameters:
#   - transmittedSignalPassBand: The passband signal transmitted through the channel.
#   - snr_db: The desired Signal-to-Noise Ratio in dB for the channel.
#   - channelType: Type of channel model (currently only "AWGN" is supported).
# Returns: noisy_signal - The transmitted signal with AWGN added.
# ============================================================================
def channelModel(transmittedSignalPassBand, snr_db,channelType="AWGN"): 
    # Calculate the power of the transmitted signal
    signal_power = rfutil.getSignalPower(transmittedSignalPassBand)[0]  # Power of the input signal in Watts
    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db/10)  # Convert SNR from dB to linear scale
    # Calculate the required noise power based on the signal power and desired SNR
    noise_power = signal_power / snr_linear  # Calculate noise power in Watts
    #print("Power of noise added in dBm: ", rfutil.getLintodBM(noise_power), "dBm")  # Print noise power in dBm
    # Generate AWGN with the calculated noise power
    noise = getAWGNNoise(transmittedSignalPassBand, noise_power)
    # Add the generated noise to the transmitted signal
    noisy_signal = transmittedSignalPassBand + noise  # Add noise to original signal
    #print("Total power of noisy signal in dBm: ", rfutil.getLintodBM(rfutil.getSignalPower(noisy_signal)[0]), "dBm")  # Print total power of noisy signal
    return noisy_signal

# ============================================================================
# FUNCTION: getAWGNNoise
# Purpose: Generates Additive White Gaussian Noise (AWGN) with a specified power.
#          Handles both real and complex signals.
# Parameters:
#   - signal: The reference signal (used to determine if noise should be complex or real).
#   - noise_power: The desired power of the generated noise in Watts.
# Returns: noise - An array of AWGN samples (complex if signal is complex, real otherwise).
# ============================================================================
def getAWGNNoise(signal, noise_power):
    if(np.iscomplexobj(signal)):
        noise_std_dev = math.sqrt(noise_power*rfutil.RefR/2)  # Standard deviation of noise (assuming Gaussian)
        noise = (np.random.normal(0, noise_std_dev, len(signal)) +1j*np.random.normal(0, noise_std_dev, len(signal)) ) # Generate AWGN noise
    else:
        noise_std_dev = math.sqrt(noise_power*rfutil.RefR)  # Standard deviation of noise (assuming Gaussian)
        noise = np.random.normal(0, noise_std_dev, len(signal))  # Generate AWGN noise for real signals
    return noise
