import numpy as np
from bitstring import BitArray
from progress.bar import Bar
import encoding

def decode_raw_IQ_wave(carrier_frequency, time_samples, input_wave):
    sin_cos_coefficient = 2 * np.pi * carrier_frequency * time_samples
    local_osc_sin = np.sin(sin_cos_coefficient)[0:input_wave.size] # Q
    local_osc_cos = np.cos(sin_cos_coefficient)[0:input_wave.size] # I
    i_vector = local_osc_cos * input_wave
    q_vector = local_osc_sin * input_wave
    return i_vector, q_vector

def bin_data(symbol_array, modulation_table, distance_threshold):
    data_array = np.empty(symbol_array.size, dtype=object)
    for symbol_index in range(symbol_array.size):
        for iq_pair in modulation_table:
            symbol = symbol_array[symbol_index]
            iq_distance_to = np.sqrt( ((symbol[0] - iq_pair[0]) ** 2) + ((symbol[0] - iq_pair[0]) ** 2) )
            if iq_distance_to < distance_threshold:
                data_array[symbol_index] = np.array(iq_pair)
                break
    return data_array
    

def merge_i_and_q_arrays(i_array, q_array):
    '''Merges seperated i and q arrays to create array of arrays in the format [i: Float, q: Float]'''
    iq_array = np.empty(i_array.size, dtype=object)
    with Bar('Merging I and Q arrays...', max=len(iq_array)/10000, suffix='%(percent)d%%') as bar:
        for index in range(iq_array.size):
            iq_array[index] = np.array([i_array[index], q_array[index]])
    return iq_array