import numpy as np
import time
from scipy import signal
from progress.bar import Bar
import matplotlib.pyplot as plt
from helpers import print_gate

def time_step_array(length, sample_rate):
    time_samples = np.arange(0, length, 1 / sample_rate)
    return time_samples

def IQ_encoded_wave(carrier_frequency, time_samples, vectorised_i_array, vectorised_q_array):
    print_gate("Generating IQ Data Encoded Wave...")
    tic = time.time()
    sin_cos_coefficient = 2 * np.pi * carrier_frequency * time_samples
    output_wave = ( vectorised_i_array * np.cos(sin_cos_coefficient) ) + ( vectorised_q_array * np.sin(sin_cos_coefficient) )
    toc = time.time()
    print_gate("Time taken for IQ Data Encoded Wave in s: " + str(toc - tic))
    return output_wave

def vectorised_data(data_array, samples_per_data):
    print_gate("Initialising Vectorised Data Array")
    data_vector = np.empty(len(data_array) * samples_per_data, dtype=np.float32)
    print_gate("Generating Vectorised Data")
    tic = time.time()
    print_gate("Data Array Size: " + str(len(data_array)))
    with Bar('Vectorising Data...', max=len(data_vector)/5000, suffix='%(percent)d%%') as bar:
        for n in range(0, len(data_array)):
            for i in range(0, samples_per_data):
                data_vector[(n * samples_per_data) + i] = data_array[n]
                if n % 5000 == 0: # Used to slow down the progress bar because its slow as hell
                    bar.next()
    toc = time.time()
    print_gate("Time taken for data vectorisation in s: " + str(toc - tic))
    return data_vector

def low_pass_filter(signal_to_filter, cuttoff_hz, sample_rate = 192e3, gain = 1):
    print_gate("Applying low pass filter...")
    tic = time.time()
    num_taps = 101 # it helps to use an odd number of taps (god knows why but its probably simple)
    h = signal.firwin(num_taps, cuttoff_hz, fs=sample_rate) # Generate filter taps
    output_signal = np.convolve(h, signal_to_filter, mode='same') # Apply filter
    toc = time.time()
    print_gate("Time taken for data filtering in s: " + str(toc - tic))
    return gain * output_signal

def normalise_signal_range(input_signal, min_bound, max_bound): # NOTE Doesnt really work and may not be needed
    min_amplitude = min(input_signal)
    max_amplitude = max(input_signal)
    gain_coefficient = (max_bound - min_bound) / (max_amplitude - min_amplitude)
    dc_offset = (max_amplitude - min_amplitude) / 2
    output_signal = gain_coefficient * input_signal
    return output_signal

def average_symbols_vector(input_signal, samples_per_symbol): #NOTE Vectorises the derived symbols pretty much only needed for visual testing
    return vectorised_data(average_symbols(input_signal, samples_per_symbol), samples_per_symbol)

def average_symbols(input_signal, samples_per_symbol):
    output_average_symbols = np.empty(int(len(input_signal)/samples_per_symbol)) #TODO this can be removed to output just the output data since we know the samples per symbol but is used for visual testing
    with Bar('Averaging symbols...', max=len(input_signal)/samples_per_symbol/10000, suffix='%(percent)d%%') as bar:
        for sample in range(0, int(len(input_signal)/samples_per_symbol)):
            output_average_symbols[sample] = np.mean(input_signal[sample*samples_per_symbol:(sample*samples_per_symbol) + (samples_per_symbol)-1])
            if sample % 10000 == 0: # Used to slow down the progress bar because its slow as hell
                bar.next()
    return output_average_symbols


