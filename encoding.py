import numpy as np
from bitstring import BitArray
from progress.bar import Bar

QAM4_TABLE = [[-1, -1], [-1, 1], [1, -1], [1, 1]] # (I, Q) only to be used with 2 bit symbols
#ASK4_TABLE = [[0.25, 0], [0.5, 0], [0.75, 0], [1, 0]]
ASK4_TABLE = [[0, 0], [0.25, 0], [0.5, 0], [1, 0]]
PSK4_TABLE = [[0, -1], [0, -0.5], [0, 0.5], [0, 1]]

QAM8_TABLE = [[-0.5, -0.5], [-1, -1], [0.5, -0.5], [1, -1], [-0.5, 0.5], [-1, 1], [0.5, 0.5], [1, 1]]

QAM16_TABLE = [[-1, 1], [-1/3, 1/3], [1/3, 1], [1, 1/3], [1/3, -1/3], [1, -1], [-1, -1/3], [-1/3, -1],
               [-1, 1/3], [-1/3, 1], [1/3, 1/3], [1, 1], [1, -1/3], [1/3, -1], [-1/3, -1/3], [-1, -1]]
QAM16_DISTANCE_THRESHOLD = 1/3



def data_bit_to_symbol_array(bit_array, bit_depth):
    print("Splitting Array into bool arrays...")
    print("Bit Array Size: " + str(len(bit_array)))
    split_array = np.array_split(np.array(bit_array), np.array(bit_array).size/bit_depth)
    symbol_array = np.empty(len(split_array), dtype=int)
    print("Symbol Array Size: " + str(symbol_array.size))
    with Bar('Converting bool arrays to symbol arrays...', max=len(split_array)/20000, suffix='%(percent)d%%') as bar:
        for i in range(len(split_array)):
            symbol_array[i] = convert_bool_array_to_uint(split_array[i])
            if i % 20000 == 0: # Used to slow down the progress bar because its slow as hell
                bar.next()
    return symbol_array

def data_symbol_to_qam4_iq_array(symbol_array):
    iq_array = np.empty(len(symbol_array), dtype=object)
    with Bar('Converting symbol array to QAM4 IQ array...', max=len(symbol_array)/10000, suffix='%(percent)d%%') as bar:
        for i in range(len(symbol_array)):
            iq_array[i] = QAM4_TABLE[symbol_array[i]]
            if i % 10000 == 0: # Used to slow down the progress bar because its slow as hell
                bar.next()
    return iq_array

def data_symbol_to_qam8_iq_array(symbol_array):
    iq_array = np.empty(len(symbol_array), dtype=object)
    with Bar('Converting symbol array to QAM8 IQ array...', max=len(symbol_array)/10000, suffix='%(percent)d%%') as bar:
        for i in range(len(symbol_array)):
            iq_array[i] = QAM8_TABLE[symbol_array[i]]
            if i % 10000 == 0: # Used to slow down the progress bar because its slow as hell
                bar.next()
    return iq_array

def data_symbol_to_qam16_iq_array(symbol_array):
    iq_array = np.empty(len(symbol_array), dtype=object)
    print("Symbol Array Size: " + str(symbol_array.size))
    with Bar('Converting symbol array to QAM16 IQ array...', max=len(symbol_array)/10000, suffix='%(percent)d%%') as bar:
        for i in range(len(symbol_array)):
            iq_array[i] = np.array(QAM16_TABLE[symbol_array[i]])
            if i % 10000 == 0: # Used to slow down the progress bar because its slow as hell
                bar.next()
    return iq_array

def data_symbol_to_ask4_iq_array(symbol_array):
    iq_array = np.empty(len(symbol_array), dtype=object)
    print("IQ Array Size: " + str(iq_array.size))
    with Bar('Converting symbol array to ASK4 IQ array...', max=len(symbol_array)/10000, suffix='%(percent)d%%') as bar:
        for i in range(len(symbol_array)):
            iq_array[i] = ASK4_TABLE[symbol_array[i]]
            if i % 10000 == 0: # Used to slow down the progress bar because its slow as hell
                bar.next()
    return iq_array

def seperate_iq_array(iq_array):
    i_array = np.empty( (len(iq_array)) )
    q_array = np.empty( (len(iq_array)) )
    with Bar('Seperating IQ array to I and Q array...', max=len(iq_array)/10000, suffix='%(percent)d%%') as bar:
        for i in range(len(iq_array)):
            iq_pair = iq_array[i]
            i_array[i] = iq_pair[0]
            q_array[i] = iq_pair[1]
            if i % 10000 == 0: # Used to slow down the progress bar because its slow as hell
                bar.next()
    return i_array, q_array

def convert_bool_array_to_uint(bool_array):
    n = 1
    result = 0
    for bit in bool_array:
        result += int(bit * n)
        n = n * 2
    return result

def iq_from_file(filepath, i_depth, q_depth):
    bit_array = BitArray(filename = filepath)
