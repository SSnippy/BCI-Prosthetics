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

# Global constants
SAMPLING_FREQ = 1024  # Sampling frequency (Hz)

print("Libraries imported successfully")

# Define the folder containing all movement files
MOVEMENTS_FOLDER ="C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/Datapoints/007-Archith"  # Change this to your folder path
OUTPUT_BASE_FOLDER = r"C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/output/archith"

def bciform(MOVEMENTS_FOLDER,OUTPUT_BASE_FOLDER):
    # Get all Excel files in the movements folder
    movement_files = glob.glob(os.path.join(MOVEMENTS_FOLDER, "*.xlsx"))
    print(f"\nFound {len(movement_files)} movement files to process")

    # Process each movement file
    for eeg_data_path in movement_files:
        try:
            print(f"\n{'='*50}")
            print(f"Processing file: {os.path.basename(eeg_data_path)}")
            print(f"{'='*50}")
            
            # Create output folder with same name as input file (without extension)
            input_filename = os.path.splitext(os.path.basename(eeg_data_path))[0]
            OUTPUT_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, input_filename)
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)

            # Create subfolders for plots and data
            PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")
            DATA_FOLDER = os.path.join(OUTPUT_FOLDER, "data")
            os.makedirs(PLOTS_FOLDER, exist_ok=True)
            os.makedirs(DATA_FOLDER, exist_ok=True)

            # Create subfolders for different types of data
            EEG_SIGNAL_FOLDER = os.path.join(DATA_FOLDER, "eeg_signal")
            FFT_FOLDER = os.path.join(DATA_FOLDER, "fourier_transform")
            PSD_FOLDER = os.path.join(DATA_FOLDER, "power_spectral_density")
            WAVELET_FOLDER = os.path.join(DATA_FOLDER, "wavelet_transform")

            # Create all data subfolders
            for folder in [EEG_SIGNAL_FOLDER, FFT_FOLDER, PSD_FOLDER, WAVELET_FOLDER]:
                os.makedirs(folder, exist_ok=True)

            print(f"Output will be saved in: {OUTPUT_FOLDER}")

            def save_to_excel(data_dict, electrode_name, timestamp):
                try:
                    # Save EEG Signal data
                    signal_data = {
                        'Time (s)': data_dict['Time (s)'],
                        'EEG Signal (μV)': data_dict['EEG Signal (μV)']
                    }
                    signal_filepath = os.path.join(EEG_SIGNAL_FOLDER, f"{electrode_name}_{timestamp}_eeg_signal.xlsx")
                    pd.DataFrame(signal_data).to_excel(signal_filepath, index=False)
                    print(f"Saved EEG signal data to: {signal_filepath}")
                    
                    # Save Fourier Transform data
                    fft_data = {
                        'Frequency (Hz)': data_dict['Frequency (Hz)'],
                        'FFT Magnitude': data_dict['FFT Magnitude']
                    }
                    fft_filepath = os.path.join(FFT_FOLDER, f"{electrode_name}_{timestamp}_fft.xlsx")
                    pd.DataFrame(fft_data).to_excel(fft_filepath, index=False)
                    print(f"Saved FFT data to: {fft_filepath}")
                    
                    # Save Power Spectral Density data
                    psd_data = {
                        'Welch Frequency (Hz)': data_dict['Welch Frequency (Hz)'],
                        'Power Spectral Density': data_dict['Power Spectral Density']
                    }
                    psd_filepath = os.path.join(PSD_FOLDER, f"{electrode_name}_{timestamp}_psd.xlsx")
                    pd.DataFrame(psd_data).to_excel(psd_filepath, index=False)
                    print(f"Saved PSD data to: {psd_filepath}")
                    
                    # Save Wavelet Transform data
                    wavelet_data = {
                        'Time (s)': data_dict['Time (s)'],
                        'Reconstructed Signal': data_dict['Reconstructed Signal']
                    }
                    wavelet_filepath = os.path.join(WAVELET_FOLDER, f"{electrode_name}_{timestamp}_wavelet.xlsx")
                    pd.DataFrame(wavelet_data).to_excel(wavelet_filepath, index=False)
                    print(f"Saved wavelet data to: {wavelet_filepath}")
                    
                    return True
                except Exception as e:
                    print(f"Error saving Excel files: {str(e)}")
                    return False

            def process_electrode(eeg_signal, electrode_name):
                try:
                    # Remove NaN values
                    eeg_signal = eeg_signal[~np.isnan(eeg_signal)]
                    
                    # Create time array for full duration
                    total_duration = len(eeg_signal) / SAMPLING_FREQ  # Total duration in seconds
                    t = np.linspace(0, total_duration, len(eeg_signal))
                    
                    # Wavelet Transform
                    wavelet = 'db4'
                    coeffs = pywt.wavedec(eeg_signal, wavelet, level=4)
                    reconstructed_signal = pywt.waverec(coeffs, wavelet)
                    
                    # Fourier Transform
                    fft_vals = fft(eeg_signal)
                    fft_freqs = np.fft.fftfreq(len(fft_vals), 1/SAMPLING_FREQ)
                    
                    # Power Spectral Density
                    f_welch, psd = welch(eeg_signal, SAMPLING_FREQ)
                    
                    # Create plots for this electrode
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
                    fig.suptitle(f'EEG Analysis for {electrode_name}', fontsize=16)
                    
                    # Original Signal Plot
                    ax1.plot(t, eeg_signal, label='EEG Signal')
                    ax1.set_title('Original EEG Signal')
                    ax1.set_xlabel('Time [s]')
                    ax1.set_ylabel('Amplitude [μV]')
                    ax1.grid(True)
                    ax1.set_xlim(0, total_duration)  # Set x-axis to show full duration
                    
                    # Fourier Transform Plot
                    positive_freqs = fft_freqs[:len(fft_freqs)//2]
                    positive_fft = np.abs(fft_vals[:len(fft_vals)//2])
                    ax2.plot(positive_freqs, positive_fft)
                    ax2.set_title('Fourier Transform (FFT)')
                    ax2.set_xlabel('Frequency [Hz]')
                    ax2.set_ylabel('Magnitude')
                    ax2.grid(True)
                    
                    # Power Spectral Density Plot
                    ax3.semilogy(f_welch, psd)
                    ax3.set_title('Power Spectral Density (Welch Method)')
                    ax3.set_xlabel('Frequency [Hz]')
                    ax3.set_ylabel('PSD')
                    ax3.grid(True)
                    
                    plt.tight_layout()
                    
                    # Save the plots
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = f"{electrode_name.replace(' ', '_')}_{timestamp}.png"
                    plot_filepath = os.path.join(PLOTS_FOLDER, plot_filename)
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                    print(f"Saved plot for {electrode_name} to: {plot_filepath}")
                    
                    # Save the analysis data to separate Excel files
                    data_dict = {
                        'Time (s)': t,
                        'EEG Signal (μV)': eeg_signal,
                        'Frequency (Hz)': positive_freqs,
                        'FFT Magnitude': positive_fft,
                        'Welch Frequency (Hz)': f_welch,
                        'Power Spectral Density': psd,
                        'Reconstructed Signal': reconstructed_signal
                    }
                    
                    if save_to_excel(data_dict, electrode_name.replace(' ', '_'), timestamp):
                        print(f"Saved all analysis data for {electrode_name}")
                    else:
                        print(f"Failed to save some Excel files for {electrode_name}")
                    
                    return fig
                except Exception as e:
                    print(f"Error processing electrode {electrode_name}: {str(e)}")
                    return None

            # Get list of all sheets in the Excel file
            excel_file = pd.ExcelFile(eeg_data_path)
            sheet_names = excel_file.sheet_names
            print(f"\nFound {len(sheet_names)} sheets in the Excel file:")
            for sheet in sheet_names:
                print(f"- {sheet}")
            
            # Read and combine all sheets
            all_data = []
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(eeg_data_path, sheet_name=sheet)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error reading sheet {sheet}: {str(e)}")
            
            # Concatenate all sheets vertically
            eeg_df = pd.concat(all_data, ignore_index=True)
            print("\nExcel file loaded successfully")
            print(f"Data shape after combining all sheets: {eeg_df.shape}")
            print(f"Total duration: {len(eeg_df)/SAMPLING_FREQ:.2f} seconds")
            
            # Save the combined raw data
            raw_data_path = os.path.join(DATA_FOLDER, "combined_raw_data.xlsx")
            eeg_df.to_excel(raw_data_path, index=False)
            print(f"\nSaved combined raw data to: {raw_data_path}")
            
            # Print available electrodes
            print("\nProcessing the following electrodes:")
            for col in eeg_df.columns:
                print(f"- {col}")
            
            # Process each electrode
            for electrode in eeg_df.columns:
                if keyboard.is_pressed('q'):
                        print("Stopping because 'q' was pressed.")
                        break
                if electrode != 'EKG-REF':  # Skip the EKG reference channel
                    print(f"\nProcessing electrode: {electrode}")
                    eeg_signal = eeg_df[electrode].values
                    fig = process_electrode(eeg_signal, electrode)
                    if fig is not None:
                        plt.pause(0.1)  # Small pause to allow plots to be displayed
                        plt.close(fig)  # Close the figure to free memory
            
            print("\nAnalysis complete. All files have been saved.")
            print(f"Results are saved in: {OUTPUT_FOLDER}")
            print(f"- Plots are in: {PLOTS_FOLDER}")
            print(f"- Data files are organized in subfolders under: {DATA_FOLDER}")
            print(f"  - EEG Signals: {EEG_SIGNAL_FOLDER}")
            print(f"  - Fourier Transforms: {FFT_FOLDER}")
            print(f"  - Power Spectral Density: {PSD_FOLDER}")
            print(f"  - Wavelet Transforms: {WAVELET_FOLDER}")
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            continue
