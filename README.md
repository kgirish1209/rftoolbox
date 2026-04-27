# CASP RF Toolbox

A Python-based Digital Communication and RF Signal Processing toolbox focusing on QPSK (Quadrature Phase Shift Keying) transreceiver simulation. This project demonstrates the full chain of digital communication, from bit generation to baseband modulation, upconversion, channel modeling, downconversion, and optimal detection.

## Features

- **QPSK Modulation/Demodulation:** Complete implementation of passband upconversion and product downconversion.
- **Pulse Shaping:** Support for rectangular pulse shapes (easily extensible to RRC).
- **Matched Filtering:** Optimal receiver design using matched filters to maximize SNR at the sampling instant.
- **Channel Modeling:** Additive White Gaussian Noise (AWGN) channel simulation.
- **Performance Analysis:** Automatic Bit Error Rate (BER) calculation and constellation diagram visualization.
- **Modern Packaging:** Managed via `pyproject.toml` for easy installation.

## Installation

Ensure you have Python 3.8 or higher installed. You can install the toolbox and its dependencies in editable mode using:

```bash
pip install -e .
```

## Project Structure

- `rftoolbox/`: Core package directory.
  - `qpskTransreceiver.py`: Contains logic for modulation, demodulation, and decision making.
  - `qpskSimulation.py`: The main entry point to run a complete end-to-end simulation.
  - `modulationUtil.py`: Utilities for symbol mapping and constellation plotting.
  - `rfutil.py`: General RF utilities like power scaling and pulse generation.
  - `channelModeling.py`: AWGN channel implementation.

## Simulation Parameters

The default simulation is configured with:
- **Bit Rate:** 15 kbps
- **LO Frequency:** 30 kHz
- **Sampling Frequency:** ~960 kHz (32x Oversampling of the LO)
- **Modulation:** QPSK (2 bits per symbol)

## Technical Notes

### SNR vs Eb/N0
In this simulation, the channel noise is added based on the per-sample SNR. Because the sampling frequency ($f_s$) is much higher than the bit rate ($R_b$), there is a significant "Processing Gain." To align the simulation results with theoretical BER curves, the SNR is calculated as:

$$SNR_{sample} (dB) = \frac{E_b}{N_0} (dB) - 10 \log_{10}\left(\frac{f_s}{R_b}\right)$$

### Receiver Design
The receiver utilizes a product demodulator followed by a Matched Filter. Note that while a Low Pass Filter (LPF) is often used after downconversion, this implementation relies on the Matched Filter to act as the optimal filter, preventing the distortion of rectangular pulses that narrow-band LPFs might cause.

## Usage

To run the simulation and see the BER and constellation results:

```bash
python rftoolbox/qpskSimulation.py
```

## Dependencies

- `numpy`: For high-performance numerical operations and vectorization.
- `matplotlib`: For plotting constellations and waveforms.

## Author

Girish Kakalwar
Version: 0.1.0