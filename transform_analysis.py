import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
import os
from datetime import datetime
import glob

# Global constants
SAMPLING_FREQ = 500  # Sampling frequency (Hz)
TIME_UNIT = 0.01    # Time unit in seconds

# Define input and output directories
INPUT_FOLDER = "C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/nr"  # Folder containing patient folders with filtered data
OUTPUT_BASE_FOLDER = "C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/ft_wt_transform"  # Base folder for all transformed data

print("Libraries imported successfully")

# Create output base folder if it doesn't exist
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

# Get all patient folders in the input directory
patient_folders = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
print(f"\nFound {len(patient_folders)} patient folders to process")

# Process each patient folder
for patient_folder in patient_folders:
    print(f"\n{'='*50}")
    print(f"Processing patient: {patient_folder}")
    print(f"{'='*50}")
    
    # Set up paths for this patient
    patient_input_path = os.path.join(INPUT_FOLDER, patient_folder)
    patient_output_path = os.path.join(OUTPUT_BASE_FOLDER, f"Transformed_{patient_folder}")
    os.makedirs(patient_output_path, exist_ok=True)
    
    # Get all Excel files in the patient folder
    excel_files = glob.glob(os.path.join(patient_input_path, "*.xlsx"))
    print(f"Found {len(excel_files)} Excel files to process")
    
    # Process each Excel file
    for excel_file in excel_files:
        try:
            print(f"\nProcessing file: {os.path.basename(excel_file)}")
            
            # Create output folder for this joint
            joint_name = os.path.splitext(os.path.basename(excel_file))[0]
            joint_output_path = os.path.join(patient_output_path, f"Transformed_{joint_name}")
            plots_path = os.path.join(joint_output_path, "plots")
            os.makedirs(joint_output_path, exist_ok=True)
            os.makedirs(plots_path, exist_ok=True)
            
            # Read all sheets from the Excel file
            xls = pd.ExcelFile(excel_file)
            fourier_results = {}
            wavelet_results = {}
            
            # Process each sheet
            for sheet_name in xls.sheet_names:
                print(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Process each electrode
                for electrode in df.columns:
                    print(f"Processing electrode: {electrode}")
                    data = df[electrode].values
                    
                    # Remove NaN values
                    data = data[~np.isnan(data)]
                    
                    # Create time array with 0.01s units
                    t = np.arange(0, len(data)) * TIME_UNIT
                    
                    # Perform Fourier transform
                    n = len(data)
                    yf = fft(data)
                    xf = fftfreq(n, TIME_UNIT)  # Using TIME_UNIT for frequency calculation
                    positive_freqs = xf[:n//2]
                    positive_fft = np.abs(yf[:n//2])
                    
                    # Store Fourier results
                    fourier_results[f"{sheet_name}_{electrode}"] = {
                        'frequency': positive_freqs,
                        'magnitude': positive_fft
                    }
                    
                    # Perform Wavelet transform
                    wavelet = 'db4'
                    coeffs = pywt.wavedec(data, wavelet, level=4)
                    reconstructed_signal = pywt.waverec(coeffs, wavelet)
                    
                    # Create time arrays for each wavelet coefficient level
                    coeff_times = []
                    for i, coeff in enumerate(coeffs):
                        # Create time array for this coefficient level
                        level_time = np.linspace(0, t[-1], len(coeff))
                        coeff_times.append(level_time)
                    
                    # Store Wavelet results
                    wavelet_results[f"{sheet_name}_{electrode}"] = {
                        'time': t,
                        'reconstructed': reconstructed_signal,
                        'coefficients': coeffs,
                        'coeff_times': coeff_times
                    }
                    
                    # Create plots
                    plt.figure(figsize=(20, 15))
                    
                    # Original EEG Signal Plot
                    plt.subplot(3, 1, 1)
                    # Plot only 50 points
                    points_to_plot = min(50, len(data))
                    plt.plot(t[:points_to_plot], data[:points_to_plot])
                    plt.title(f'EEG Signal - {sheet_name} - {electrode}')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                    plt.grid(True)
                    # Set x-axis ticks to show time values every 0.2 seconds
                    max_time = t[points_to_plot - 1]
                    tick_interval = 0.2  # Show ticks every 0.2 seconds
                    plt.xticks(np.arange(0, max_time + tick_interval, tick_interval))
                    
                    # Fourier Transform Plot
                    plt.subplot(3, 1, 2)
                    # Plot only 50 points
                    points_to_plot_fft = min(50, len(positive_freqs))
                    plt.plot(positive_freqs[:points_to_plot_fft], positive_fft[:points_to_plot_fft])
                    plt.title('Fourier Transform')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude')
                    plt.grid(True)
                    
                    # Wavelet Transform Plot
                    plt.subplot(3, 1, 3)
                    for i, (coeff, coeff_time) in enumerate(zip(coeffs, coeff_times)):
                        # Plot only 50 points for each coefficient level
                        points_to_plot_wavelet = min(50, len(coeff))
                        plt.plot(coeff_time[:points_to_plot_wavelet], coeff[:points_to_plot_wavelet], label=f'Level {i}')
                    plt.title('Wavelet Transform Coefficients')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Coefficient Value')
                    plt.legend()
                    plt.grid(True)
                    # Set x-axis ticks to show time values every 0.2 seconds
                    plt.xticks(np.arange(0, max_time + tick_interval, tick_interval))
                    
                    plt.tight_layout()
                    
                    # Save plot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = f"{sheet_name}_{electrode}_{timestamp}.png"
                    plt.savefig(os.path.join(plots_path, plot_filename), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved plot: {plot_filename}")
            
            # Save Fourier results to Excel with common frequency column
            fourier_df = pd.DataFrame()
            # Add common frequency column
            first_key = next(iter(fourier_results))
            fourier_df['Frequency (Hz)'] = fourier_results[first_key]['frequency']
            
            # Add magnitude columns for each sheet and electrode
            for key, value in fourier_results.items():
                fourier_df[f"{key}_magnitude"] = value['magnitude']
            
            fourier_df.to_excel(os.path.join(joint_output_path, "fourier_transforms.xlsx"), index=False)
            print("Saved Fourier transform data")
            
            # Save Wavelet results to Excel
            wavelet_df = pd.DataFrame()
            for key, value in wavelet_results.items():
                wavelet_df[f"{key}_time"] = value['time']
                wavelet_df[f"{key}_reconstructed"] = value['reconstructed']
                for i, (coeff, coeff_time) in enumerate(zip(value['coefficients'], value['coeff_times'])):
                    wavelet_df[f"{key}_level{i}_time"] = coeff_time
                    wavelet_df[f"{key}_level{i}_coeff"] = coeff
            wavelet_df.to_excel(os.path.join(joint_output_path, "wavelet_transforms.xlsx"), index=False)
            print("Saved Wavelet transform data")
            
        except Exception as e:
            print(f"Error processing file {excel_file}: {str(e)}")
            continue
    
    print(f"\nCompleted processing patient: {patient_folder}")
    print(f"Results saved in: {patient_output_path}")

print("\nAll processing complete!")
print(f"All transformed data is saved in: {OUTPUT_BASE_FOLDER}")