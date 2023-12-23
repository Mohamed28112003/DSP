import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import math
from CompareSignal import Compare_Signals





def conjugate(signal):
    return np.conj(signal)




def dft(signal):
    lenth = len(signal)
    dft_result = []
    for k in range(lenth):
        sum_val = 0
        for n in range(lenth):
            angle = 2 * np.pi * k * n / lenth
            sum_val += signal[n] * np.exp(-1j * angle)
        dft_result.append(sum_val)
    return np.array(dft_result)


def idft(signal):
    lenth = len(signal)
    idft_result = []
    for n in range(lenth):
        sum_val = 0
        for k in range(lenth):
            angle = 2 * np.pi * k * n / lenth
            sum_val += signal[k] * np.exp(1j * angle)
        idft_result.append(sum_val / lenth)
    return np.array(idft_result)


def zero_pad(signal, target_length):
    return np.pad(signal, (0, target_length - len(signal)), 'constant')

def read_signal_from_file_two_cloumn(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        input_signal = []
        for line in lines:
            values = line.split()
            if len(values) >= 2:
                input_signal.append(float(values[1]))
        return input_signal

ecg_list = read_signal_from_file_two_cloumn('ecg400.txt')

def type_of_window_function(stop_band, transtion_witdth, sampling_freq):
    flag = 0
    if stop_band > 0 and stop_band <= 21:
        # rectangle
        delta_f = transtion_witdth / sampling_freq
        N = 0.9 / delta_f
        N = math.ceil(N)
        if N % 2 == 0:
            N = N + 1
        flag = 1

    elif stop_band > 21 and stop_band <= 44:
        # hanning
        delta_f = transtion_witdth / sampling_freq
        N = 3.1 / delta_f
        N = math.ceil(N)
        if N % 2 == 0:
            N = N + 1
        flag = 2

    elif stop_band > 44 and stop_band <= 53:
        # hamming
        delta_f = transtion_witdth / sampling_freq
        N = 3.3 / delta_f
        N = math.ceil(N)
        if N % 2 == 0:
            N = N + 1
        flag = 3

    elif stop_band > 53 and stop_band <= 74:
        # blackman
        delta_f = transtion_witdth / sampling_freq
        N = 5.5 / delta_f
        N = math.ceil(N)
        if N % 2 == 0:
            N = N + 1
        flag = 4
        # print("iam in blackman")

    return N, flag


def low_pass(n, cut_freq_, sampling_freq, transtion_witdth):
    # delta_f = transtion_witdth/sampling_freq
    cut_freq_ = (cut_freq_ + (transtion_witdth / 2)) / sampling_freq
    if n == 0:
        h = 2 * cut_freq_
        # print("ANa aho ya rgala")
        # print(h)
    else:
        deg_rad = (n * 2 * math.pi * cut_freq_)
        h = 2 * cut_freq_ * math.sin(deg_rad) / (n * 2 * math.pi * cut_freq_)

    return h


def high_pass(n, cut_freq, sampling_freq, transtion_witdth):
    cut_freq = (cut_freq - (transtion_witdth / 2)) / sampling_freq
    if n == 0:
        h = 1 - 2 * cut_freq
    else:
        deg_rad = (n * 2 * math.pi * cut_freq)
        h = -2 * cut_freq * math.sin(deg_rad) / (n * 2 * math.pi * cut_freq)

    return h





def band_pass_freq(n, freq1, freq2, sampling_freq, transtion_witdth):
    freq2 = (freq2 + (transtion_witdth / 2)) / sampling_freq
    freq1 = (freq1 - (transtion_witdth / 2)) / sampling_freq
    if n == 0:
        h = 2 * (freq2 - freq1)
    else:
        deg_rad_1 = (n * 2 * math.pi * freq1)
        deg_rad_2 = (n * 2 * math.pi * freq2)
        h2 = (2 * freq2) * math.sin(deg_rad_2) / (n * 2 * math.pi * freq2)
        h1 = (2 * freq1) * math.sin(deg_rad_1) / (n * 2 * math.pi * freq1)
        h = h2 - h1

    return h


def band_stop(n, freq1, freq2, sampling_freq, transtion_witdth):
    freq2 = (freq2 - (transtion_witdth / 2)) / sampling_freq
    freq1 = (freq1 + (transtion_witdth / 2)) / sampling_freq
    if n == 0:
        h = 1 - 2 * (freq2 - freq1)
    else:
        deg_rad_1 = (n * 2 * math.pi * freq1)
        deg_rad_2 = (n * 2 * math.pi * freq2)
        h2 = (2 * freq2) * math.sin(deg_rad_2) / (n * 2 * math.pi * freq2)
        h1 = (2 * freq1) * math.sin(deg_rad_1) / (n * 2 * math.pi * freq1)
        h = h1 - h2
    return h



def window_function(n, stop_band, transtion_witdth, sampling_freq):
    N, flag = type_of_window_function(stop_band, transtion_witdth, sampling_freq)
    if flag == 1:
        w = 1
    elif flag == 2:
        rad = (2 * math.pi * n / N)
        w = 0.5 + 0.5 * math.cos(rad)
    elif flag == 3:
        rad = (2 * math.pi * n / N)
        w = 0.54 + 0.46 * math.cos(rad)
    elif flag == 4:
        rad1 = (2 * math.pi * n / (N - 1))
        rad2 = (4 * math.pi * n / (N - 1))
        w = 0.42 + (0.5 * math.cos(rad1)) + (0.08 * math.cos(rad2))

    return w


def final_result(filter, stop_band, transtion_witdth, sampling_freq, cut_freq):
    N, flag = type_of_window_function(stop_band, transtion_witdth, sampling_freq)

    result_list = []
    indcis_list =[]
    if filter == 'low':
        for i in range(int(-N / 2), int((N / 2) + 1)):
            h = low_pass(i, cut_freq, sampling_freq, transtion_witdth)
            w = window_function(i, stop_band, transtion_witdth, sampling_freq)
            res = h * w
            result_list.append(res)
            indcis_list.append(i)
    elif filter == 'high':
        for i in range(int(-N / 2), int((N / 2) + 1)):
            h = high_pass(i, cut_freq, sampling_freq, transtion_witdth)
            w = window_function(i, stop_band, transtion_witdth, sampling_freq)
            res = h * w
            result_list.append(res)
            indcis_list.append(i)

            # print(i)
    return result_list,indcis_list


def final_2(filter, stop_band, transtion_witdth, sampling_freq, freq1, freq2):
    N, flag = type_of_window_function(stop_band, transtion_witdth, sampling_freq)
    result_list = []
    indcis_list =[]
    if filter == 'band_pass':
        for i in range(int(-N / 2), int((N / 2) + 1)):
            h = band_pass_freq(i, freq1, freq2, sampling_freq, transtion_witdth)
            w = window_function(i, stop_band, transtion_witdth, sampling_freq)
            res = h * w
            result_list.append(res)
            indcis_list.append(i)

    elif filter == 'band_stop':
        for i in range(int(-N / 2), int((N / 2) + 1)):
            h = band_stop(i, freq1, freq2, sampling_freq, transtion_witdth)
            w = window_function(i, stop_band, transtion_witdth, sampling_freq)
            res = h * w
            result_list.append(res)
            indcis_list.append(i)

    return result_list,indcis_list

def fast_convolution(filter, stop_band, transtion_witdth, sampling_freq, cut_freq, list_ecg):
    signal1 = list_ecg
    signal2 ,indcis = final_result(filter, stop_band, transtion_witdth, sampling_freq, cut_freq)

    N1 = len(signal1)
    N2 = len(signal2)

    padded_size = N1 + N2 - 1

    padded_signal1 = zero_pad(signal1, padded_size)
    padded_signal2 = zero_pad(signal2, padded_size)

    freq_domain_sig1 = dft(padded_signal1)
    freq_domain_sig2 = dft(padded_signal2)

    convolved_freq_domain = freq_domain_sig1 * freq_domain_sig2

    convolved_time_domain = idft(convolved_freq_domain)

    result = np.real(convolved_time_domain)

    return result



def fast_convolution_2(filter, stop_band, transtion_witdth, sampling_freq, frec1, frec2, list_1):
    signal1 = list_1
    signal2,indcis = final_2(filter, stop_band, transtion_witdth, sampling_freq, frec1, frec2)

    N1 = len(signal1)
    N2 = len(signal2)

    padded_size = N1 + N2 - 1

    padded_signal1 = zero_pad(signal1, padded_size)
    padded_signal2 = zero_pad(signal2, padded_size)

    freq_domain_sig1 = dft(padded_signal1)
    freq_domain_sig2 = dft(padded_signal2)

    convolved_freq_domain = freq_domain_sig1 * freq_domain_sig2

    convolved_time_domain = idft(convolved_freq_domain)

    result = np.real(convolved_time_domain)

    return result

def resampling(list, m, l, filter, stop_band, transtion_witdth, sampling_freq, cut_freq):
    if m == 0 and l != 0:
        upsampled_signal = np.zeros(len(list) * l)
        upsampled_signal[::l] = list
        list_sampling = fast_convolution(filter, stop_band, transtion_witdth, sampling_freq, cut_freq, upsampled_signal)
        list_sampling = list_sampling[:-2]
        return list_sampling
    elif m != 0 and l == 0:
        list_sampling = fast_convolution(filter, stop_band, transtion_witdth, sampling_freq, cut_freq, list)
        downsampled_signal = list_sampling[::m]
        return downsampled_signal
    elif m!=0 and l!=0:
        upsampled_signal = np.zeros(len(list) * l)
        upsampled_signal[::l] = list
        list_sampling = fast_convolution(filter, stop_band, transtion_witdth, sampling_freq, cut_freq, upsampled_signal)
        list_sampling = list_sampling[:-2]
        downsampled_signal = list_sampling[::m]
        return downsampled_signal
    elif m==0 and l==0:
        print("Error can't resample")



def resample_signal():
    try:
        # Retrieve values from GUI input fields
        fs_val = float(fs_entry.get())
        stop_attenuation_val = float(stop_attenuation_entry.get())
        transition_band_val = float(transition_band_entry.get())
        filter_type_val = filter_type_var.get()
        m = int(m_entry.get())
        l = int(l_entry.get())


        cutoff_val = cutoff_entry.get()
        if cutoff_val == "":
            cutoff_val = 0  # Set a default value if cutoff is not provided
        else:
            cutoff_val = float(cutoff_val)

        output = "Output:\n"

        if filter_type_val == "low":
            result_list= resampling(ecg_list, m, l, "low", stop_attenuation_val, transition_band_val, fs_val, cutoff_val)
            output += f"Filter Type: Low-pass\n"
            output += "Coefficients: " + str(result_list) + "\n"
            #Compare_Signals("output_task1/LPFCoefficients.txt", indcies_list, result_list)
        index=[]
        for i, value in enumerate(result_list[::-1], start=-26):  # Reversed and adjusted index
            index.append(i)


        if m == 0 and l != 0:
         Compare_Signals("output_task1/Sampling_Up.txt", index, result_list)

        elif m != 0 and l == 0:
            Compare_Signals("output_task1/Sampling_Down.txt", index, result_list)
        elif m != 0 and l != 0:
            Compare_Signals("output_task1/Sampling_Up_Down.txt", index, result_list)
        elif m==0 and l==0 :
            messagebox.showerror("Error can't resample")









        messagebox.showinfo("Success", "Filter calculation completed! Check console for output.")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")



def perform_fast_convolution():
    try:
        # Retrieve values from GUI input fields
        fs_val = float(fs_entry.get())
        stop_attenuation_val = float(stop_attenuation_entry.get())
        transition_band_val = float(transition_band_entry.get())
        filter_type_val = filter_type_var.get()

        # Check if cutoff frequency is provided
        cutoff_val = cutoff_entry.get()
        if cutoff_val == "":
            cutoff_val = 0  # Set a default value if cutoff is not provided
        else:
            cutoff_val = float(cutoff_val)

        # Perform calculations using your functions based on the input values
        output = "Output:\n"

        if filter_type_val == "low":

            result_list = fast_convolution("low", stop_attenuation_val, transition_band_val, fs_val, cutoff_val,ecg_list)
            output += f"Filter Type: Low-pass\n"

            output += "Coefficients: " + str(result_list) + "\n"

        elif filter_type_val == "high":
            result_list = fast_convolution("high", stop_attenuation_val, transition_band_val, fs_val, cutoff_val,ecg_list)
            output += f"Filter Type: High-pass\n"
            output += "Coefficients: " + str(result_list) + "\n"

        elif filter_type_val == "band_pass":
            f1_val = float(f1_entry.get())
            f2_val = float(f2_entry.get())
            result_list = fast_convolution_2("band_pass", stop_attenuation_val, transition_band_val, fs_val, f1_val, f2_val,ecg_list)
            output += f"Filter Type: Band-pass\n"
            output += "Coefficients: " + str(result_list) + "\n"


        elif filter_type_val == "band_stop":
            f1_val = float(f1_entry.get())
            f2_val = float(f2_entry.get())
            result_list = fast_convolution_2("band_stop", stop_attenuation_val, transition_band_val, fs_val, f1_val, f2_val,ecg_list)
            output += f"Filter Type: Band-stop\n"
            output += "Coefficients: " + str(result_list) + "\n"

        messagebox.showinfo("Success", "Filter calculation completed! Check console for output.")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")


def calculate_filter():
    try:
        # Retrieve values from GUI input fields
        fs_val = float(fs_entry.get())
        stop_attenuation_val = float(stop_attenuation_entry.get())
        transition_band_val = float(transition_band_entry.get())
        filter_type_val = filter_type_var.get()

        # Check if cutoff frequency is provided
        cutoff_val = cutoff_entry.get()
        if cutoff_val == "":
            cutoff_val = 0  # Set a default value if cutoff is not provided
        else:
            cutoff_val = float(cutoff_val)

        output = "Output:\n"

        if filter_type_val == "low":
            result_list ,indcies_list= final_result("low", stop_attenuation_val, transition_band_val, fs_val, cutoff_val)
            output += f"Filter Type: Low-pass\n"
            output += "Coefficients: " + str(result_list) + "\n"
            Compare_Signals("output_task1/LPFCoefficients.txt",indcies_list,result_list)

        elif filter_type_val == "high":
            #N, flag = type_of_window_function(stop_attenuation_val, transition_band_val, fs_val)
            result_list,indcies_list = final_result("high", stop_attenuation_val, transition_band_val, fs_val, cutoff_val)
            output += f"Filter Type: High-pass\n"
            output += "Coefficients: " + str(result_list) + "\n"
            Compare_Signals("output_task1/HPFCoefficients.txt",indcies_list,result_list)


        elif filter_type_val == "band_pass":
            f1_val = float(f1_entry.get())
            f2_val = float(f2_entry.get())
            result_list,indcies_list = final_2("band_pass", stop_attenuation_val, transition_band_val, fs_val, f1_val, f2_val)
            output += f"Filter Type: Band-pass\n"
            output += "Coefficients: " + str(result_list) + "\n"
            Compare_Signals("output_task1/BPFCoefficients.txt",indcies_list,result_list)


        elif filter_type_val == "band_stop":
            f1_val = float(f1_entry.get())
            f2_val = float(f2_entry.get())
            result_list ,indcies_list= final_2("band_stop", stop_attenuation_val, transition_band_val, fs_val, f1_val, f2_val)
            output += f"Filter Type: Band-stop\n"
            output += "Coefficients: " + str(result_list) + "\n"
            Compare_Signals("output_task1/BSFCoefficients.txt",indcies_list,result_list)


        #print(output)
        messagebox.showinfo("Success", "Filter calculation completed! Check console for output.")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# GUI setup with added input fields for f1 and f2 frequencies
root = tk.Tk()
root.title("Filter Parameter Calculator")

# Create labels and input fields for user input
fs_label = tk.Label(root, text="Sampling Frequency:")
fs_label.pack()
fs_entry = tk.Entry(root)
fs_entry.pack()

cutoff_label = tk.Label(root, text="Cutoff Frequency:")
cutoff_label.pack()
cutoff_entry = tk.Entry(root)
cutoff_entry.pack()

stop_attenuation_label = tk.Label(root, text="Stop Attenuation (δs):")
stop_attenuation_label.pack()
stop_attenuation_entry = tk.Entry(root)
stop_attenuation_entry.pack()

transition_band_label = tk.Label(root, text="Transition Band:")
transition_band_label.pack()
transition_band_entry = tk.Entry(root)
transition_band_entry.pack()

filter_type_label = tk.Label(root, text="Filter Type:")
filter_type_label.pack()

# Radio buttons for selecting filter type
filter_type_var = tk.StringVar()
filter_type_var.set("low")  # Default value
filter_type_low = tk.Radiobutton(root, text="Low-pass", variable=filter_type_var, value="low")
filter_type_low.pack()
filter_type_high = tk.Radiobutton(root, text="High-pass", variable=filter_type_var, value="high")
filter_type_high.pack()
filter_type_bandpass = tk.Radiobutton(root, text="Band-pass", variable=filter_type_var, value="band_pass")
filter_type_bandpass.pack()
filter_type_bandstop = tk.Radiobutton(root, text="Band-stop", variable=filter_type_var, value="band_stop")
filter_type_bandstop.pack()

# Additional input fields for Band-pass and Band-stop filters
f1_label = tk.Label(root, text="f1 Frequency:")
f1_label.pack()
f1_entry = tk.Entry(root)
f1_entry.pack()

f2_label = tk.Label(root, text="f2 Frequency:")
f2_label.pack()
f2_entry = tk.Entry(root)
f2_entry.pack()

calculate_button = tk.Button(root, text="Calculate Filter", command=calculate_filter)
calculate_button.pack()
# Create buttons to trigger functions
button_convolution = tk.Button(root, text="Perform Fast Convolution", command=perform_fast_convolution)
button_convolution.pack()



m_label = tk.Label(root, text="Value of m:")
m_label.pack()
m_entry = tk.Entry(root)
m_entry.pack()

l_label = tk.Label(root, text="Value of l:")
l_label.pack()
l_entry = tk.Entry(root)
l_entry.pack()

# Create a button to trigger the resampling process
resample_button = tk.Button(root, text="Resample Signal", command=resample_signal)
resample_button.pack()


root.mainloop()