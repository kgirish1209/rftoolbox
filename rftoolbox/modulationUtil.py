import numpy as np
import matplotlib.pyplot as plt



# Define the QPSK symbol mapping
symbolMappingQPSK = {
    (0, 0): 1 +1j,   # Symbol for bits '00'
    (0, 1): 1 - 1j,  # Symbol for bits '01'
    (1, 0): -1 + 1j,   # Symbol for bits '10'
    (1, 1): -1 - 1j   # Symbol for bits '11'
}

symbolMappingQAM4 = {
    (0, 0): 1 + 1j,   # Symbol for bits '00'
    (0, 1): 1 - 1j,  # Symbol for bits '01'
    (1, 0): -1 + 1j,   # Symbol for bits '10'
    (1, 1): -1 - 1j   # Symbol for bits '11'
}

symbolMappingQAM16 = {
    (0, 0, 0, 0): 1 + 1j,    # Symbol for bits '0000'
    (0, 0, 0, 1): 1 + 3j,    # Symbol for bits '0001'
    (0, 0, 1, 0): 3 + 1j,    # Symbol for bits '0010'
    (0, 0, 1, 1): 3 + 3j,    # Symbol for bits '0011'
    (0, 1, 0, 0): 1 - 1j,    # Symbol for bits '0100'
    (0, 1, 0, 1): -1 + 3j,    # Symbol for bits '0101'
    (0, 1, 1, 0): -3 + 1j,    # Symbol for bits '0110'
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

def getSymbolMapping(ipBitstream,modulationType="QPSK"):
    # Reshape the bitstream into groups of 2 bits
    if modulationType == "QPSK":
        numBitsPerSymbol = 2
        symbolMapping = symbolMappingQPSK
    elif modulationType == "16QAM":
        numBitsPerSymbol = 4
        symbolMapping = symbolMappingQAM16
    else:
        raise ValueError("Unsupported modulation type. Use 'QPSK' or '16QAM'.")
    # Map each pair of bits to the corresponding symbol
    ipSymbolStream = []
    for i in range(0, len(ipBitstream), numBitsPerSymbol):
        bits = tuple(ipBitstream[i:i+numBitsPerSymbol])
        ipSymbolStream.append(symbolMapping[bits])
    return np.array(ipSymbolStream)

def plotConstellation(inPhase,quadraturePhase,title,plt):
    plt.figure(figsize=(6, 6))
    plt.scatter(inPhase, quadraturePhase, color='blue')
    plt.title(title)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid()
    plt.axis('equal')
    #plt.show()
    
def getSymbolDemapping(receivedSymbols, modulationType="QPSK"):
    if modulationType == "QPSK":
        symbolMapping = symbolMappingQPSK
    elif modulationType == "16QAM":
        symbolMapping = symbolMappingQAM16
    else:
        raise ValueError("Unsupported modulation type. Use 'QPSK' or '16QAM'.")
    
    # 2. Convert dictionary to arrays for vectorized math
    # labels: the bit patterns (e.g., [0, 0], [0, 1])
    # points: the ideal complex coordinates
    labels = list(symbolMapping.keys())
    points = np.array(list(symbolMapping.values()))

    # 3. Minimum Distance Decoding (Hard Decision)
    # We use broadcasting to find the distance from every received symbol 
    # to every ideal constellation point.
    decoded_bits = []
    
    for rx in receivedSymbols:
        # Calculate Euclidean distance to all ideal points
        # |rx - point|^2
        distances = np.abs(rx - points)**2
        
        # Find the index of the closest point
        closest_idx = np.argmin(distances)
        
        # Append the bit label associated with that point
        decoded_bits.extend(labels[closest_idx])

    return np.array(decoded_bits)