import numpy as np
import matplotlib.pyplot as plt


# Define the QPSK symbol mapping
symbolMappingQPSK = {
    (0, 0): 1 + 1j,   # Symbol for bits '00'
    (0, 1): 1 - 1j,   # Symbol for bits '01'
    (1, 0): -1 + 1j,  # Symbol for bits '10'
    (1, 1): -1 - 1j   # Symbol for bits '11'
}

symbolMappingQAM4 = symbolMappingQPSK

symbolMappingQAM16 = {
    (0, 0, 0, 0): 1 + 1j,    # Symbol for bits '0000'
    (0, 0, 0, 1): 1 + 3j,    # Symbol for bits '0001'
    (0, 0, 1, 0): 3 + 1j,    # Symbol for bits '0010'
    (0, 0, 1, 1): 3 + 3j,    # Symbol for bits '0011'
    (0, 1, 0, 0): 1 - 1j,    # Symbol for bits '0100'
    (0, 1, 0, 1): 1 - 3j,    # Symbol for bits '0101'
    (0, 1, 1, 0): 3 - 1j,    # Symbol for bits '0110'
    (0, 1, 1, 1): 3 - 3j,    # Symbol for bits '0111'
    (1, 0, 0, 0): -1 + 1j,   # Symbol for bits '1000'
    (1, 0, 0, 1): -1 + 3j,   # Symbol for bits '1001'
    (1, 0, 1, 0): -3 + 1j,   # Symbol for bits '1010'
    (1, 0, 1, 1): -3 + 3j,   # Symbol for bits '1011'
    (1, 1, 0, 0): -1 - 1j,   # Symbol for bits '1100'
    (1, 1, 0, 1): -1 - 3j,   # Symbol for bits '1101'
    (1, 1, 1, 0): -3 - 1j,   # Symbol for bits '1110'
    (1, 1, 1, 1): -3 - 3j    # Symbol for bits '1111'
}

def getSymbolMapping(ipBitstream, modulationType="QPSK"):
    if modulationType in ("QPSK", "4QAM", "QAM4"):
        numBitsPerSymbol = 2
        symbolMapping = symbolMappingQPSK
    elif modulationType in ("16QAM", "QAM16"):
        numBitsPerSymbol = 4
        symbolMapping = symbolMappingQAM16
    else:
        raise ValueError(f"Unsupported modulation type '{modulationType}'. Use 'QPSK', '4QAM', or '16QAM'.")

    if len(ipBitstream) % numBitsPerSymbol != 0:
        raise ValueError(f"Bitstream length ({len(ipBitstream)}) must be a multiple of {numBitsPerSymbol} for {modulationType}.")

    ipSymbolStream = []
    for i in range(0, len(ipBitstream), numBitsPerSymbol):
        bits = tuple(ipBitstream[i:i+numBitsPerSymbol])
        ipSymbolStream.append(symbolMapping[bits])
    return np.array(ipSymbolStream)

def plotConstellation(inPhase, quadraturePhase, title, plt):
    plt.figure(figsize=(6, 6))
    plt.scatter(inPhase, quadraturePhase, color='blue')
    plt.title(title)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid()
    plt.axis('equal')
    
def getSymbolDemapping(receivedSymbols, modulationType="QPSK"):
    if modulationType in ("QPSK", "4QAM", "QAM4"):
        symbolMapping = symbolMappingQPSK
    elif modulationType in ("16QAM", "QAM16"):
        symbolMapping = symbolMappingQAM16
    else:
        raise ValueError(f"Unsupported modulation type '{modulationType}'. Use 'QPSK', '4QAM', or '16QAM'.")
    
    labels = list(symbolMapping.keys())
    points = np.array(list(symbolMapping.values()))

    # Vectorized Minimum Distance Decoding (Hard Decision)
    rx = np.asarray(receivedSymbols)
    distances = np.abs(rx[:, np.newaxis] - points[np.newaxis, :])**2
    closest_indices = np.argmin(distances, axis=1)

    decoded_bits = []
    for idx in closest_indices:
        decoded_bits.extend(labels[idx])

    return np.array(decoded_bits)