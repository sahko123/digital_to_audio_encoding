import numpy as np
import signal_processing
from bitstring import BitArray
from progress.bar import Bar
import psutil
import os
import gc
from helpers import print_gate

QAM4_TABLE = [[-1, -1], [-1, 1], [1, -1], [1, 1]] # (I, Q) only to be used with 2 bit symbols
#ASK4_TABLE = [[0.25, 0], [0.5, 0], [0.75, 0], [1, 0]]
ASK4_TABLE = [[0, 0], [0.25, 0], [0.5, 0], [1, 0]]
PSK4_TABLE = [[0, -1], [0, -0.5], [0, 0.5], [0, 1]]

QAM8_TABLE = [[-0.5, -0.5], [-1, -1], [0.5, -0.5], [1, -1], [-0.5, 0.5], [-1, 1], [0.5, 0.5], [1, 1]]

QAM16_TABLE = [[-1, 1], [-1/3, 1/3], [1/3, 1], [1, 1/3], [1/3, -1/3], [1, -1], [-1, -1/3], [-1/3, -1],
               [-1, 1/3], [-1/3, 1], [1/3, 1/3], [1, 1], [1, -1/3], [1/3, -1], [-1/3, -1/3], [-1, -1]]
QAM16_DISTANCE_THRESHOLD = 1/3

def process_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss

def bit_array_to_data_encoded_wave(input_data_bit_array, symbol_bit_depth, samples_per_symbol, sample_rate, carrier_frequency):
    input_data_integer_array = data_bit_to_symbol_array(input_data_bit_array, symbol_bit_depth) # BitArray to Symbol Array
    del input_data_bit_array
    input_data_iq_array = data_symbol_to_qam16_iq_array(input_data_integer_array) # Symbol Array to modulated IQ array
    del input_data_integer_array
    input_data_i_array, input_data_q_array = seperate_iq_array(input_data_iq_array) # Split IQ array to I and Q array
    del input_data_iq_array
    input_vectorised_i_array = signal_processing.vectorised_data(input_data_i_array, samples_per_symbol) # Vectorise I array
    input_vectorised_q_array = signal_processing.vectorised_data(input_data_q_array, samples_per_symbol) # Vectorise Q array
    # NOTE Generation of Data encoded waveform
    length_time = (input_data_i_array.size * samples_per_symbol) / sample_rate
    time_samples = signal_processing.time_step_array(length_time, sample_rate)
    time_samples = time_samples[0:(input_vectorised_i_array.size)] # Hack to prevent this from being larger than symbol vector by 1 which happens sometimes for some fucking reason i dont know how to fix it properly and i cant be assed can probably solved with a simple floor or ceiling operation but ( ͡° ͜ʖ ͡°)
    input_data_encoded_wave = signal_processing.IQ_encoded_wave(carrier_frequency, time_samples, input_vectorised_i_array, input_vectorised_q_array)
    del input_vectorised_i_array, input_vectorised_q_array
    return input_data_encoded_wave

def data_bit_to_symbol_array(bit_array, bit_depth):
    print_gate("Splitting Array into bool arrays...")
    print_gate("Bit Array Size: " + str(len(bit_array)))
    np_bit_array = np.array(bit_array, dtype=bool)
    del bit_array
    split_array: np.array = np.array_split(np_bit_array, np_bit_array.size/bit_depth)
    del np_bit_array
    symbol_array = np.empty(len(split_array), dtype=np.uint8)
    print_gate("Symbol Array Size: " + str(symbol_array.size))
    with Bar('Converting bool arrays to symbol arrays...', max=len(split_array)/20000, suffix='%(percent)d%%') as bar:
        for i in range(len(split_array)):
            symbol_array[i] = np.uint8(convert_bool_array_to_uint(split_array[i]))
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
    print_gate("Symbol Array Size: " + str(symbol_array.size))
    with Bar('Converting symbol array to QAM16 IQ array...', max=len(symbol_array)/10000, suffix='%(percent)d%%') as bar:
        for i in range(len(symbol_array)):
            iq_array[i] = np.array(QAM16_TABLE[symbol_array[i]])
            if i % 10000 == 0: # Used to slow down the progress bar because its slow as hell
                bar.next()
    return iq_array

def data_symbol_to_ask4_iq_array(symbol_array):
    iq_array = np.empty(len(symbol_array), dtype=object)
    print_gate("IQ Array Size: " + str(iq_array.size))
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
    result = BitArray(bool_array)
    
    #n = 1
    #result = 0
    #for bit in bool_array:
    #    result += int(bit * n)
    #    n = n * 2
    return result.uint

def iq_from_file(filepath, i_depth, q_depth):
    bit_array = BitArray(filename = filepath)

def eng_format(x, precision=3):
    """Returns string in engineering format, i.e. 100.1e-3"""
    x = float(x)  # inplace copy
    if x == 0:
        a,b = 0,0
    else: 
        sgn = 1.0 if x > 0 else -1.0
        x = abs(x) 
        a = sgn * x / 10**(np.floor(np.log10(x)))
        b = int(np.floor(np.log10(x)))

    if -3 < b < 3: 
        return ("%." + str(precision) + "g") % x
    else:
        a = a * 10**(b % 3)
        b = b - b % 3
        return ("%." + str(precision) + "gE%s") % (a,b)
