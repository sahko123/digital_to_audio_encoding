import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys
import pydub
from bitstring import Bits, BitArray, BitStream, ConstBitStream
import signal_processing
import encoding
import decoding
from decimal import *
import gc
import os


def read(f, normalized=False, reshape=False):
    a = pydub.AudioSegment.from_file(f, format='wav')
    y = np.array(a.get_array_of_samples())
    if reshape and a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.sample_width, a.frame_rate, np.float32(y) / 2**15
    else:
        return a.sample_width, a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized: # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

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

def iq_plot(i, q, figure_count, dot_size = 5):
    plt.figure(figure_count)
    plt.scatter(i, q, s=dot_size)
    plt.grid()
    figure_count+=1
    #exit()


working_directory = os.getcwd()
print("Working directory: " + os.getcwd())
test_output_directory = working_directory + os.path.join("\\test_output")
if not os.path.isdir(test_output_directory):
    print("Output directory doesnt exist, creating directory")
    os.mkdir(test_output_directory)

sample_rate = 192e3
symbol_bit_depth = 4

cycles_per_symbol = 2
carrier_frequency = 20e3

samples_per_symbol = int(np.ceil( ( (1/carrier_frequency) / (1/sample_rate) ) * cycles_per_symbol))
bits_per_second = ((symbol_bit_depth) / ( cycles_per_symbol/carrier_frequency )) ** 2

# NOTE Conversion of input file to IQ data vectors
print("Begin Encoding...")
input_data_bit_array = BitArray(filename = '128kbps_joint_stereo.mp3')
input_data_bit_array = BitArray(filename = 'test_data') # File to BitArray
#input_data_bit_array = input_data_bit_array[0:10000]
input_data_integer_array = encoding.data_bit_to_symbol_array(input_data_bit_array, symbol_bit_depth) # BitArray to Symbol Array
input_data_iq_array = encoding.data_symbol_to_qam16_iq_array(input_data_integer_array) # Symbol Array to modulated IQ array
input_data_i_array, input_data_q_array = encoding.seperate_iq_array(input_data_iq_array) # Split IQ array to I and Q array
input_vectorised_i_array = signal_processing.vectorised_data(input_data_i_array, samples_per_symbol) # Vectorise I array
input_vectorised_q_array = signal_processing.vectorised_data(input_data_q_array, samples_per_symbol) # Vectorise Q array

# NOTE Generation of Data encoded waveform
length_time = (input_data_i_array.size * samples_per_symbol) / sample_rate
time_samples = signal_processing.time_step_array(length_time, sample_rate)
time_samples = time_samples[0:(input_vectorised_i_array.size)] # Hack to prevent this from being larger than symbol vector by 1 which happens sometimes for some fucking reason i dont know how to fix it properly and i cant be assed can probably solved with a simple floor or ceiling operation but ( ͡° ͜ʖ ͡°)
input_data_encoded_wave = signal_processing.IQ_encoded_wave(carrier_frequency, time_samples, input_vectorised_i_array, input_vectorised_q_array)

#del input_data_bit_array, input_data_symbol_array, input_data_iq_array # NOTE used in an attempt to reduce 24GB ram usage
#gc.collect()
print("bits per second ", eng_format(bits_per_second))
print("Sample length", length_time)
print("Sample length in minutes", length_time/60)

# NOTE Conversion of data encoded wave to output data
filter_frequency = 20e3
local_osc_frequency = 20e3

print("Begin Decoding...")
output_data_decode_i_array, output_data_decode_q_array = decoding.decode_raw_IQ_wave(local_osc_frequency, time_samples, input_data_encoded_wave) # Modulated signal to I and Q array
output_data_i_vector = signal_processing.low_pass_filter(output_data_decode_i_array, filter_frequency, gain=1) # Filter I vector
output_data_q_vector = signal_processing.low_pass_filter(output_data_decode_q_array, filter_frequency, gain=1) # Filter Q vector
output_data_averaged_i_array = signal_processing.average_symbols(output_data_i_vector, samples_per_symbol) # Retrieve averaged I array NOTE average_symbols_vector will return a vectorised version only really needed for visualisation
output_data_averaged_q_array = signal_processing.average_symbols(output_data_q_vector, samples_per_symbol) # Retrieve averaged Q array
output_data_normalised_i_array = signal_processing.normalise_signal_range(output_data_averaged_i_array, -1, 1) # Normalise I array
output_data_normalised_q_array = signal_processing.normalise_signal_range(output_data_averaged_q_array, -1, 1) # Normalise Q array
output_data_averaged_iq_array = decoding.merge_i_and_q_arrays(output_data_normalised_i_array, output_data_normalised_q_array) # Merge I and Q array to IQ array
output_data_iq_array = decoding.decode_to_iq_array(output_data_averaged_iq_array, encoding.QAM16_TABLE, encoding.QAM16_DISTANCE_THRESHOLD) # IQ Array to Symbol Array
output_data_integer_array = decoding.qam16_to_data_array(output_data_iq_array)
output_data_bit_array = decoding.data_symbol_to_bit_array(output_data_integer_array, bit_depth=symbol_bit_depth)# Symbol Array to BitArray
output_file = decoding.bit_array_to_file(output_data_bit_array, test_output_directory + os.path.join("\\test_output"), '.txt')
output_file.close()
del output_data_decode_i_array, output_data_decode_q_array, output_data_averaged_i_array, output_data_averaged_q_array
gc.collect()

output_data_i_symbol_array, output_data_q_symbol_array = encoding.seperate_iq_array(output_data_iq_array) # Seperated version of output_data_symbol_array data array

# NOTE Plotting schtuff
viewing_extent = 200000

iq_plot(output_data_normalised_i_array[0:viewing_extent], output_data_normalised_q_array[0:viewing_extent], 1)
iq_plot(input_vectorised_i_array[0:viewing_extent], input_vectorised_q_array[0:viewing_extent], 1)
iq_plot(output_data_i_symbol_array[0:viewing_extent], output_data_q_symbol_array[0:viewing_extent], 1, dot_size=2)
axs: plt.axes 
fig, axs = plt.subplots(5, sharex=True)
#NOTE time_samples[0:viewing_extent],  add this back for when you want time instead of nsample
axs[0].plot(time_samples[0:viewing_extent], input_data_encoded_wave[0:viewing_extent])
axs[1].plot(time_samples[0:viewing_extent], signal_processing.vectorised_data(output_data_i_symbol_array[0:viewing_extent], samples_per_symbol)[0:viewing_extent])
axs[2].plot(time_samples[0:viewing_extent], input_vectorised_i_array[0:viewing_extent])
axs[3].plot(time_samples[0:viewing_extent], signal_processing.vectorised_data(output_data_q_symbol_array[0:viewing_extent], samples_per_symbol)[0:viewing_extent])
axs[4].plot(time_samples[0:viewing_extent], input_vectorised_q_array[0:viewing_extent])

if (output_data_i_symbol_array == input_data_i_array).all():
    print("SUCCESS I Array", output_data_i_symbol_array.size, " ", input_data_i_array.size)
if (output_data_q_symbol_array == input_data_q_array).all():
    print("SUCCESS Q Array", output_data_q_symbol_array.size, " ", input_data_q_array.size)


#print(input_data_iq_array)
#print(output_data_iq_array)

print(input_data_integer_array)
print(output_data_integer_array)
print(type(input_data_integer_array[0]))
print(type(output_data_integer_array[0]))


print(input_data_bit_array)
print(output_data_bit_array)




plt.show()