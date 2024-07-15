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

sample_rate = 192e3
symbol_depth = 4

print("Creating BitArray from file...")
#test_data = BitArray(filename = '128kbps_joint_stereo.mp3')
test_data = BitArray(filename = 'test_data')
print("bit count ", np.array(test_data).size)

symbol_array = encoding.data_bit_to_symbol_array(test_data, symbol_depth)
print("number of symbols ", symbol_array.size)

iq_array = encoding.data_symbol_to_qam16_iq_array(symbol_array)
print("number of IQ pairs ", iq_array.size)

i_array, q_array = encoding.seperate_iq_array(iq_array)



cycles_per_symbol = 4
carrier_frequency = 10e3
samples_per_symbol = int(np.ceil( ( (1/carrier_frequency) / (1/sample_rate) ) * cycles_per_symbol))
length_time = (i_array.size * samples_per_symbol) / sample_rate
symbols_per_second = (symbol_depth) / ( cycles_per_symbol/carrier_frequency )
print("bits per second ", eng_format(symbols_per_second * symbol_depth))
print("Sample length", length_time)

vectorised_i_array = signal_processing.vectorised_data(i_array, samples_per_symbol)
vectorised_q_array = signal_processing.vectorised_data(q_array, samples_per_symbol)



time_samples = signal_processing.time_step_array(length_time, sample_rate)
time_samples = time_samples[0:(vectorised_i_array.size)] # Hack to prevent this from being larger than symbol vector by 1 which happens sometimes for some fucking reason i dont know how to fix it properly and i cant be assed can probably solved with a simple floor or ceiling operation but ( ͡° ͜ʖ ͡°)


data_encoded_wave = signal_processing.IQ_encoded_wave(carrier_frequency, time_samples, vectorised_i_array, vectorised_q_array)

time_samples = signal_processing.time_step_array(length_time, sample_rate)

decode_i_array, decode_q_array = decoding.decode_raw_IQ_wave(20e3, time_samples, data_encoded_wave)

filter_frequency = 19e3
viewing_extent = time_samples.size
#iq_plot(signal_gen.low_pass_filter(decode_i_array[0:viewing_extent], filter_frequency, gain=5), signal_gen.low_pass_filter(decode_q_array[0:viewing_extent], filter_frequency, gain=5))

processed_i = signal_processing.low_pass_filter(decode_i_array, filter_frequency, gain=1)
averaged_i = signal_processing.average_symbols(processed_i, samples_per_symbol)
processed_q = signal_processing.low_pass_filter(decode_q_array, filter_frequency, gain=1)
averaged_q = signal_processing.average_symbols(processed_q, samples_per_symbol)

averaged_iq_array = decoding.merge_i_and_q_arrays(signal_processing.normalise_signal_range(averaged_i, -1, 1), signal_processing.normalise_signal_range(averaged_q, -1, 1))
print(averaged_iq_array)
decoded_iq_array = decoding.bin_data(averaged_iq_array, encoding.QAM16_TABLE, encoding.QAM16_DISTANCE_THRESHOLD)
print(decoded_iq_array)
test_result_i_array, test_result_q_array = encoding.seperate_iq_array(decoded_iq_array)
iq_plot(signal_processing.normalise_signal_range(averaged_i, -1, 1), signal_processing.normalise_signal_range(averaged_q, -1, 1), 1)
iq_plot(vectorised_i_array, vectorised_q_array, 1)
axs: plt.axes 
fig, axs = plt.subplots(5, sharex=True)
#NOTE time_samples[0:viewing_extent],  add this back for when you want time instead of nsample
axs[0].plot(time_samples[0:viewing_extent],data_encoded_wave[0:viewing_extent])
axs[1].plot(time_samples[0:viewing_extent],processed_i)
axs[2].plot(time_samples[0:viewing_extent],averaged_i)
axs[3].plot(time_samples[0:viewing_extent],test_result_i_array[0:viewing_extent])
axs[4].plot(time_samples[0:viewing_extent],vectorised_i_array[0:viewing_extent])
if all(test_result_i_array == vectorised_i_array):
    print("SUCCESS", test_result_i_array.size)
plt.show()

