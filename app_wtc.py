from WaveTransform_Comparative import bciform
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import welch
import pandas as pd
import os
from datetime import datetime
import glob
import keyboard

base_path = 'C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/Datapoints'
output_path='C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/WaveTransforms-0'

output = []  # List to store patient folder names
done=[]
# Traverse through each patient folder
for patient_folder in os.listdir(base_path):
    patient_path = os.path.join(base_path, patient_folder)

    if os.path.isdir(patient_path):
        output.append(patient_folder)  # Add patient folder name to the list

for patient_folder1 in os.listdir(output_path):
    patient_path1 = os.path.join(output_path, patient_folder1)

    if os.path.isdir(patient_path1):
        done.append(patient_folder1)  # Add patient folder name to the list

# Join all file paths into a single string if needed
start=[]
for y in output:
    if y not in done:
        start.append(y)

if len(start)==0:
    print("All patients data have been processed")
else:
    for x in start:
        mf ="C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/Datapoints/"+x
        obf = r"C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/output/"+x

        bciform(mf,obf)
