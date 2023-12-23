import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from CompareSignal import Compare_Signals


amplitudes_list = []
phases_list = []
frequencies_list = []
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)

def read_signal_from_file_indecies(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        input_signal = []
        for line in lines:
            values = line.split()
            if len(values) >= 2:
                input_signal.append(float(values[0]))
        return input_signal



def read_signal_from_file_two_cloumn(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        input_signal = []
        for line in lines:
            values = line.split()
            if len(values) >= 2:
                input_signal.append(float(values[1]))
        return input_signal


def read_signal_from_file_one_cloumn(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        input_signal = []
        for line in lines:
            input_signal.append(float(line))
        return input_signal



#task 8 done
##########################################33
def dft(signal):
    N = len(signal)
    dft_result = []
    for k in range(N):
        sum_val = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            sum_val += signal[n] * np.exp(-1j * angle)
        dft_result.append(sum_val)
    return np.array(dft_result)

def idft(signal):
    N = len(signal)
    idft_result = []
    for n in range(N):
        sum_val = 0
        for k in range(N):
            angle = 2 * np.pi * k * n / N
            sum_val += signal[k] * np.exp(1j * angle)
        idft_result.append(sum_val / N)
    return np.array(idft_result)

def zero_pad(signal, target_length):
    return np.pad(signal, (0, target_length - len(signal)), 'constant')

def fast_convolution():
    indecies_1 = read_signal_from_file_indecies("Input_conv_Sig1.txt")
    indecies_2 = read_signal_from_file_indecies("Input_conv_Sig2.txt")
    signal1 = read_signal_from_file_two_cloumn("Input_conv_Sig1.txt")
    signal2 = read_signal_from_file_two_cloumn("Input_conv_Sig2.txt")

    min_index = min(indecies_1)
    max_index = max(indecies_2)

    all_indecies = list(range(int(min_index), int(max_index) + 2))
    N1 = len(signal1)
    N2 = len(signal2)

    padded_size = N1 + N2 - 1

    padded_signal1 = zero_pad(signal1, padded_size)
    padded_signal2 = zero_pad(signal2, padded_size)

    freq_domain_sig1 = dft(padded_signal1)
    freq_domain_sig2 = dft(padded_signal2)

    convolved_freq_domain = freq_domain_sig1 * freq_domain_sig2

    convolved_time_domain = idft(convolved_freq_domain)

    result = np.real(convolved_time_domain).round().astype(int)
    print(result)
    return result


def conjugate(signal):
    return np.conj(signal)

def fast_auto_correlation():
    signal = read_signal_from_file_two_cloumn("Corr_input signal1.txt")
    N = len(signal)

    freq_domain_sig = dft(signal)
    conjugate_freq_domain_sig = conjugate(freq_domain_sig)

    correlated_freq_domain = freq_domain_sig * conjugate_freq_domain_sig

    correlated_time_domain = idft(correlated_freq_domain)

    result = np.real(correlated_time_domain) / N
    print(result)

def fast_cross_correlation():
    indecies = read_signal_from_file_indecies("Corr_input signal1.txt")
    signal1 = read_signal_from_file_two_cloumn("Corr_input signal1.txt")
    signal2 = read_signal_from_file_two_cloumn("Corr_input signal2.txt")
    N = len(signal1)

    freq_domain_sig1 = dft(signal1)
    freq_domain_sig2 = dft(signal2)

    conjugate_freq_domain_sig1 = conjugate(freq_domain_sig1)

    correlated_freq_domain = conjugate_freq_domain_sig1 * freq_domain_sig2

    correlated_time_domain = idft(correlated_freq_domain)

    result = np.real(correlated_time_domain) / N
    print(result)
    return result


#################################################


#task 7 done
########################################
def normalized_cross_correlation():
    X1 = read_signal_from_file_two_cloumn("Corr_input signal1.txt")
    X2 = read_signal_from_file_two_cloumn("Corr_input signal2.txt")
    indeces = read_signal_from_file_indecies("Corr_input signal2.txt")
    N = len(X1)
    r12 = cross_correlation(X1, X2)

    sum_X1_squared = sum(x**2 for x in X1)
    sum_X2_squared = sum(x**2 for x in X2)
    normalization_factor =1/N * (sum_X1_squared * sum_X2_squared) ** 0.5

    for j in range(N):
        r12[j] /= normalization_factor
    print(r12)

def cross_correlation(x, y):
    N = len(x)
    result = []

    def circular_shift(arr, shift):
        return arr[-shift:] + arr[:-shift]

    for k in range(N):

        shifted_x = circular_shift(x, k)

        cross_corr = 0
        for i in range(N):
            cross_corr += shifted_x[i] * y[i]

        result.append(cross_corr / N)

    return result


def avg_class1():

    input1=read_signal_from_file_one_cloumn("down1.txt")

    input2=read_signal_from_file_one_cloumn("down2.txt")
    input3=read_signal_from_file_one_cloumn("down3.txt")
    input4=read_signal_from_file_one_cloumn("down4.txt")
    input5=read_signal_from_file_one_cloumn("down5.txt")

    N=len(input5)
    output_list=[]
    for i in range(N):
        output=(input1[i]+input2[i]+input3[i]+input4[i]+input5[i])/5
        output_list.append(output)
    return output_list


def avg_class2():
    input1=read_signal_from_file_one_cloumn("class1.txt")
    input2=read_signal_from_file_one_cloumn("class2.txt")
    input3=read_signal_from_file_one_cloumn("calss3.txt")
    input4=read_signal_from_file_one_cloumn("calss4.txt")
    input5=read_signal_from_file_one_cloumn("calss5.txt")

    N=len(input5)
    output_list=[]
    for i in range(N):
        output=(input1[i]+input2[i]+input3[i]+input4[i]+input5[i])/5
        output_list.append(output)
    return output_list

def match():
    test1=read_signal_from_file_one_cloumn("Test1.txt")
    test2=read_signal_from_file_one_cloumn("Test2.txt")

    class1=avg_class1()
    class2 = avg_class2()

    test1_corr_class1=max(cross_correlation(test1,class1))
    test1_corr_class2=max(cross_correlation(test1,class2))
    test2_corr_class1=max(cross_correlation(test2,class1))
    test2_corr_class2=max(cross_correlation(test2,class2))
    if test1_corr_class1 > test1_corr_class2:
        print("Test 1 Belong To Class 1 ")
    else:
        print("Test 1 Belong To Class 2")


    if test2_corr_class1 > test2_corr_class2:
        print("Test 2 Belong To Class 1 ")
    else:
        print("Test 2 Belong To Class 2")


def time_analysis() :
    X1 = read_signal_from_file_two_cloumn("TD_input signal1.txt")
    X2 = read_signal_from_file_two_cloumn("TD_input signal2.txt")
    cross_corr = cross_correlation(X1, X2)
    Fs = 100
    delay_index = cross_corr.index(max(cross_corr))
    delay = (delay_index) / Fs
    print(delay)
#####################################################################
# task 5 DONE

def compute_dct():
    # Function to compute DCT coefficient for a given k


    input_signal = read_signal_from_file_two_cloumn("DCT_input.txt")
    if input_signal is None:
        return
    N = len(input_signal)

    def dct_coefficient(k):
        sum_val = 0
        for n in range(N):
            sum_val += input_signal[n] * math.cos((math.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
        return sum_val * math.sqrt(2 / N)

    # Computing and printing DCT coefficients
    dct_result = [dct_coefficient(k) for k in range(N)]

    # Plotting the DCT coefficients using a bar plot
    plt.bar(range(N), dct_result)
    plt.xlabel('k')
    plt.ylabel('Coefficient')
    plt.title('Discrete Cosine Transform (DCT) Coefficients')
    plt.show()

def compute_and_subtract_mean(column):
    mean_value = np.mean(column)
    return column - mean_value
def modify_and_plot_signal():


    with open('DC_component_input.txt', 'r') as file:
        # Read the first two lines
        value1 = float(file.readline().strip())
        value2 = float(file.readline().strip())

        # Read the length of the data
        data_length = int(file.readline().strip())

        # Initialize lists for the two columns
        column1 = [value1, value2]
        column2 = []

        # Process the specified number of lines and extract two values from each line
        for _ in range(data_length):
            line = file.readline().strip()
            values = line.split()
            if len(values) == 2:
                column1.append(float(values[0]))
                column2.append(float(values[1]))

    # Compute and subtract the mean from column 2
    column2 = compute_and_subtract_mean(np.array(column2))

    # Plot the modified signal
    plt.plot(column2, label='Modified Signal')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Remove DC')
    plt.legend()
    plt.show()









#####################################################################




# task 6 done
#####################################################################

# input file kan eh ?
def smoothed_signal():
    input_signal = read_signal_from_file_two_cloumn("Signal1.txt")

    num_points = simpledialog.askinteger("Input", "Enter the value of k:")
    if num_points is None:
        return  # If user clicks Cancel
    smoothed_signal = []
    for i in range(len(input_signal) - num_points + 1):
        window = input_signal[i:i + num_points]
        average = sum(window) / num_points
        smoothed_signal.append(average)
    print(smoothed_signal)
   #SignalSamplesAreEqual('OutMovAvgTest2.txt' , smoothed_signal)
    return smoothed_signal

def delay_advance_signal():
    with open('input_fold.txt', 'r') as file:
        # Read the first two lines
        value1 = float(file.readline().strip())
        value2 = float(file.readline().strip())

        # Read the length of the data
        data_length = int(file.readline().strip())

        # Initialize lists for the two columns
        column1 = [value1, value2]
        column2 = []

        # Process the specified number of lines and extract two values from each line
        for _ in range(data_length):
            line = file.readline().strip()
            values = line.split()
            if len(values) == 2:
                column1.append(float(values[0]))
                column2.append(float(values[1]))
    column1.pop(0)
    column1.pop(0)
    k = simpledialog.askinteger("Input", "Enter the value of k:")
    if k is None:
        return  # If user clicks Cancel
    advanced_delayed_signal = [value + k for value in column1]
    print(advanced_delayed_signal)


def delay_advance_folded_signal():
    k = simpledialog.askinteger("Input", "Enter the value of k:")
    if k is None:
        return  # If user clicks Cancel
    with open('input_fold.txt', 'r') as file:
        # Read the first two lines
        value1 = float(file.readline().strip())
        value2 = float(file.readline().strip())

        # Read the length of the data
        data_length = int(file.readline().strip())

        # Initialize lists for the two columns
        column1 = [value1, value2]
        column2 = []

        # Process the specified number of lines and extract two values from each line
        for _ in range(data_length):
            line = file.readline().strip()
            values = line.split()
            if len(values) == 2:
                column1.append(float(values[0]))
                column2.append(float(values[1]))
    column1.pop(0)
    column1.pop(0)
    column2.reverse()

    advanced_delayed_signal = [value + k for value in column1]
   # Shift_Fold_Signal('Output_ShiftFoldedby-500.txt',advanced_delayed_signal, column2)
    print(column1)
    print(column2)
    return advanced_delayed_signal



def fold_signal():
    with open('input_fold.txt', 'r') as file:
        # Read the first two lines
        value1 = float(file.readline().strip())
        value2 = float(file.readline().strip())

        # Read the length of the data
        data_length = int(file.readline().strip())

        # Initialize lists for the two columns
        column1 = [value1, value2]
        column2 = []

        # Process the specified number of lines and extract two values from each line
        for _ in range(data_length):
            line = file.readline().strip()
            values = line.split()
            if len(values) == 2:
                column1.append(float(values[0]))
                column2.append(float(values[1]))
    column1.pop(0)
    column1.pop(0)
    column2.reverse()
    print(column1)
    print(column2)

def DerivativeSignal():
    InputSignal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 , 17 , 18 , 19 , 20 , 21 , 22 ,
                   23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 ,
                   43 , 44 , 45 , 46 , 47 , 48 , 49 , 50 , 51 , 52 , 53 , 54 , 55 , 56 , 57 , 58 , 59 , 60 , 61 , 62 ,
                   63 , 64 , 65 , 66 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 , 76 , 77 , 78 , 79 , 80 , 81 , 82 ,
                   83 , 84 , 85 , 86 , 87 , 88 , 89 , 90 , 91 , 92 , 93 , 94 , 95 , 96 , 97 , 98 , 99 , 100 ]
    expectedOutput_first = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1]
    expectedOutput_second = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0]

    """
    Write your Code here:
    Start
    """

    FirstDrev = [InputSignal[i + 1] - InputSignal[i] for i in range(len(InputSignal) - 1)]
    SecondDrev = [ InputSignal[i + 2] - 2 *  InputSignal[i + 1] +  InputSignal[i] for i in range(len( InputSignal) - 2)]


    """
    End
    """

    """
    Testing your Code
    """
    if ((len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second))):
        print("mismatch in length")
        return
    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first = False
            print("1st derivative wrong")
            return
    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second = False
            print("2nd derivative wrong")
            return
    if (first and second):
        print("Derivative Test case passed successfully")
    else:
        print("Derivative Test case failed")
    return

def convolution():
    with open('Input_conv_Sig1.txt', 'r') as file:
        # Read the first two lines
        value1 = float(file.readline().strip())
        value2 = float(file.readline().strip())

        # Read the length of the data
        data_length = int(file.readline().strip())

        # Initialize lists for the two columns
        column1 = [value1, value2]
        column2 = []

        # Process the specified number of lines and extract two values from each line
        for _ in range(data_length):
            line = file.readline().strip()
            values = line.split()
            if len(values) == 2:
                column1.append(float(values[0]))
                column2.append(float(values[1]))
    with open('Input_conv_Sig2.txt', 'r') as file:
        # Read the first two lines
        value1 = float(file.readline().strip())
        value2 = float(file.readline().strip())

        # Read the length of the data
        data_length = int(file.readline().strip())

        # Initialize lists for the two columns
        data1 = [value1, value2]
        data2 = []

        # Process the specified number of lines and extract two values from each line
        for _ in range(data_length):
            line = file.readline().strip()
            values = line.split()
            if len(values) == 2:
                data1.append(float(values[0]))
                data2.append(float(values[1]))
    len_signal1 = len(column2)
    len_signal2 = len(data2)

    # Length of the resulting convolved signal
    len_result = len_signal1 + len_signal2 - 1

    # Initialize the result with zeros
    result = [0] * len_result

    # Perform the convolution
    for i in range(len_signal1):
        for j in range(len_signal2):
            result[i + j] += column2[i] * data2[j]

    print(result)
    return result



def compute_and_subtract_mean(column):
    mean_value = np.mean(column)
    return column - mean_value



def modify_and_plot_signal():


    with open('DC_component_input.txt', 'r') as file:

        value1 = float(file.readline().strip())
        value2 = float(file.readline().strip())


        data_length = int(file.readline().strip())


        column1 = [value1, value2]
        column2 = []


        for _ in range(data_length):
            line = file.readline().strip()
            values = line.split()
            if len(values) == 2:
                column1.append(float(values[0]))
                column2.append(float(values[1]))


    column2 = compute_and_subtract_mean(np.array(column2))

    plt.plot(column2, label='Modified Signal')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Remove DC')
    plt.legend()
    plt.show()

#####################################################################



# task 4


#######################################33
def compute_dft():


    input_signal = read_signal_from_file_two_cloumn("input_Signal_DFT.txt")


    sampling_frequency = simpledialog.askfloat("Sampling Frequency", "Enter the sampling frequency (Hz):")

    X = [0] * len(input_signal)
    N = len(input_signal)

    for k in range(N):
            X[k] = sum(input_signal[n] * cmath.exp(-2j * math.pi * k * n / N) for n in range(N))
            real_part = X[k].real
            imaginary_part = X[k].imag
            amplitudes = math.sqrt(real_part **2 + imaginary_part **2)
            amplitudes_list.append(amplitudes)
            phases=math.atan2(imaginary_part,real_part)
            phases_list.append(phases)
            frequencies = (2 * math.pi * k * sampling_frequency) / N
            frequencies_list.append(frequencies)
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.bar(frequencies_list, amplitudes_list, width=1.0, color='b')
    plt.title("Frequency vs Amplitude")
    plt.xlabel("Frequency ")
    plt.ylabel("Amplitude")

    plt.grid(True)


    plt.subplot(212)
    plt.bar(frequencies_list, phases_list, width=1.0, color='r')
    plt.title("Frequency vs. Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")

    plt.grid(True)


    plt.tight_layout()
    plt.show()



def save_to_file():
    file_name = filedialog.asksaveasfilename(defaultextension=".txt")
    if file_name:
        with open(file_name, 'w') as file:
            file.write(f"{0}\n")
            file.write(f"{1}\n")
            file.write(f"{len(frequencies_list)}\n")
            for i in range(len(frequencies_list)):
                file.write(f"{amplitudes_list[i]}, {phases_list[i]}\n")


#modify msh s7 ?

def modify_component():
    index = simpledialog.askinteger("Modify Component", "Enter the index of the component to modify ")

    new_amplitude = simpledialog.askfloat("Modify Component", "Enter the new amplitude:")
    new_phase = simpledialog.askfloat("Modify Component", "Enter the new phase (radians):")

    amplitudes_list[index] = new_amplitude
    phases_list[index] = new_phase


def idft_signal():
    signal = read_signal_from_file_two_cloumn("Input_Signal_IDFT_A,Phase.txt")
    print(signal)
    N = len(signal)
    idft_result = []
    for n in range(N):
        sum_val = 0
        for k in range(N):
            angle = 2 * np.pi * k * n / N
            sum_val += signal[k] * np.exp(1j * angle)
        idft_result.append(sum_val / N)
        print(idft_result)
    return np.array(idft_result)



# task 3
#msh s7
#######################################33

def quantize_signal_final(signal, num_bits):
    max_value = max(signal)
    min_value = min(signal)
    step_size = (max_value - min_value) / (2 ** num_bits)
    quantized_signal = []

    for value in signal:
        quantized_value = round((value - min_value) / step_size) * step_size + min_value + step_size/2
        quantized_signal.append(quantized_value)

    return quantized_signal





def quantize_signal():
    levels_or_bits = 2



    levels = 2 ** levels_or_bits
    input_signal= read_signal_from_file_two_cloumn("Quan1_input.txt")




    quantized_signal = quantize_signal_final(input_signal, levels_or_bits)

    min_value = min(quantized_signal)
    step_size = (max(quantized_signal) - min_value) / (levels - 1)
    encoded_signal = [bin(int((sample - min_value) / step_size))[2:].zfill(levels_or_bits) for sample in quantized_signal]

    quantization_error = sum([(input_signal[i] - quantized_signal[i]) ** 2 for i in range(len(input_signal))]) / len(
        input_signal)

    messagebox.showinfo("Quantized Signal", "Quantized Signal:\n" + "\n".join(
        [f"{encoded_signal[i]} {quantized_signal[i]:.2f}" for i in range(len(encoded_signal))]))
    messagebox.showinfo("Quantization Error", "Quantization Error: " + str(quantization_error))

    plt.plot(input_signal, label="Input Signal")
    plt.plot(quantized_signal, label="Quantized Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Quantized Signal")
    plt.legend()
    plt.show()

#################################




window = tk.Tk()



label_task8 = tk.Label(window, text="TASK 8:")
label_task8.pack()
fast_corr = tk.Button(window, text="Fast Correlation", command=fast_cross_correlation)
fast_corr.pack()
fast_conv = tk.Button(window, text="Fast Convolution", command=fast_convolution)
fast_conv.pack()
fast_auto_corr = tk.Button(window, text="Fast Auto Correlation", command=fast_auto_correlation)
fast_auto_corr.pack()

label_task7 = tk.Label(window, text="TASK 7:")
label_task7.pack()
match_btn = tk.Button(window, text="Match", command=match)
match_btn.pack()
time_analysis_btn = tk.Button(window, text="Time Analysis", command=time_analysis)
time_analysis_btn.pack()

normalized_cross_correlation_btn = tk.Button(window, text="Normalized Cross Correlation", command=normalized_cross_correlation)
normalized_cross_correlation_btn.pack()

# Create label for TASK 6
label_task6 = tk.Label(window, text="TASK 6:")
label_task6.pack()

# Button for smoothed_signal()
smoothed_btn = tk.Button(window, text="Smoothed Signal", command=smoothed_signal)
smoothed_btn.pack()

# Button for delay_advance_signal()
delay_advance_btn = tk.Button(window, text="Delay Advance Signal", command=delay_advance_signal)
delay_advance_btn.pack()

# Button for delay_advance_folded_signal()
delay_advance_folded_btn = tk.Button(window, text="Delay Advance Folded Signal", command=delay_advance_folded_signal)
delay_advance_folded_btn.pack()

# Button for fold_signal()
fold_btn = tk.Button(window, text="Fold Signal", command=fold_signal)
fold_btn.pack()

# Button for DerivativeSignal()
derivative_btn = tk.Button(window, text="Derivative Signal", command=DerivativeSignal)
derivative_btn.pack()

# Button for convolution()
convolution_btn = tk.Button(window, text="Convolution", command=convolution)
convolution_btn.pack()

# Button for modify_and_plot_signal()
modify_plot_btn = tk.Button(window, text="Modify and Plot Signal", command=modify_and_plot_signal)
modify_plot_btn.pack()




label_task5 = tk.Label(window, text="TASK 5:")
label_task5.pack()

# Button for compute_dft()
compute_dct_btn = tk.Button(window, text="Compute Dct", command=compute_dct)
compute_dct_btn.pack()

button_modify_and_plot = tk.Button(window, text="Remove DC", command=modify_and_plot_signal)
button_modify_and_plot.pack()







label_task4 = tk.Label(window, text="TASK 4:")
label_task4.pack()

# Button for compute_dft()
compute_dft_btn = tk.Button(window, text="Compute DFT", command=compute_dft)
compute_dft_btn.pack()

# Button for save_to_file()
save_to_file_btn = tk.Button(window, text="Save to File", command=save_to_file)
save_to_file_btn.pack()

# Button for modify_component()
modify_component_btn = tk.Button(window, text="Modify Component", command=modify_component)
modify_component_btn.pack()

idft_signal_btn = tk.Button(window, text="IDFT", command=idft_signal)
idft_signal_btn.pack()



window.mainloop()