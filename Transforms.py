import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from datetime import datetime
from scipy.fft import fft, fftfreq
import pywt
import warnings
warnings.filterwarnings('ignore')  # Suppress some PyWavelets warnings

# Global constants
SAMPLING_FREQ = 1024  # Sampling frequency (Hz)
TIME_WINDOW = 0.1  # Time window in seconds
TIME_TICK_INTERVAL = 0.01  # Time tick interval in seconds

print("Libraries imported successfully")

# Define the folder containing all movement files
MOVEMENTS_FOLDER = "C:/Users/aksha/OneDrive/Documents/nr1"  
OUTPUT_BASE_FOLDER = r"C:/Users/aksha/OneDrive/Documents/output"

# Get all Excel files in the movements folder
movement_files = glob.glob(os.path.join(MOVEMENTS_FOLDER, "*.xlsx"))
print(f"\nFound {len(movement_files)} movement files to process")

# Process each movement file
for eeg_data_path in movement_files:
    try:
        print(f"\n{'='*100}")
        print(f"Processing file: {os.path.basename(eeg_data_path)}")
        print(f"{'='*100}")
        
        # Create parent folder with same name as input file (without extension)
        input_filename = os.path.splitext(os.path.basename(eeg_data_path))[0]
        PARENT_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, input_filename)
        os.makedirs(PARENT_FOLDER, exist_ok=True)

        # Create Fourier Transform subfolder
        FT_FOLDER = os.path.join(PARENT_FOLDER, "fourier_transform")
        os.makedirs(FT_FOLDER, exist_ok=True)

        # Create Wavelet Transform subfolder
        WAVELET_FOLDER = os.path.join(PARENT_FOLDER, "wavelet_transform")
        os.makedirs(WAVELET_FOLDER, exist_ok=True)

        print(f"Output will be saved in: {PARENT_FOLDER}")
        print(f"Fourier Transform results will be saved in: {FT_FOLDER}")
        print(f"Wavelet Transform results will be saved in: {WAVELET_FOLDER}")

        def extract_window(eeg_signal, start_time):
            """
            Extract a 0.2-second window starting from the specified time.
            
            Args:
                eeg_signal: The complete EEG signal
                start_time: Start time in seconds
                
            Returns:
                window_signal: The extracted signal window
                window_time: Time array for the window
            """
            # Calculate start and end indices
            start_idx = int(start_time * SAMPLING_FREQ)
            end_idx = int((start_time + TIME_WINDOW) * SAMPLING_FREQ)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, len(eeg_signal)))
            end_idx = max(0, min(end_idx, len(eeg_signal)))
            
            # Extract the window
            window_signal = eeg_signal[start_idx:end_idx]
            window_time = np.linspace(start_time, start_time + TIME_WINDOW, len(window_signal))
            
            return window_signal, window_time

        def perform_fourier_transform(eeg_signal, electrode_name):
            """
            Perform Fourier Transform on the EEG signal and save results.
            
            Args:
                eeg_signal: The EEG signal array
                electrode_name: Name of the electrode
                
            Returns:
                frequencies: Array of frequencies
                magnitudes: Array of FFT magnitudes
            """
            # Remove NaN values
            eeg_signal = eeg_signal[~np.isnan(eeg_signal)]
            
            # Perform FFT
            n = len(eeg_signal)
            yf = fft(eeg_signal)
            xf = fftfreq(n, 1/SAMPLING_FREQ)
            
            # Calculate magnitude spectrum
            magnitude = np.abs(yf)
            
            # Filter frequencies to match noise reduction limits (0.5 Hz to 40 Hz)
            freq_mask = (xf >= 0.5) & (xf <= 40)
            xf = xf[freq_mask]
            magnitude = magnitude[freq_mask]
            
            # Create electrode-specific folder for FT plots
            electrode_ft_folder = os.path.join(FT_FOLDER, electrode_name.replace(' ', '_'))
            os.makedirs(electrode_ft_folder, exist_ok=True)
            
            # Plot and save FT results
            plt.figure(figsize=(10, 6))
            plt.plot(xf, magnitude)
            plt.title(f'Fourier Transform - {electrode_name}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"ft_{electrode_name.replace(' ', '_')}_{timestamp}.png"
            plot_filepath = os.path.join(electrode_ft_folder, plot_filename)
            try:
                plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error saving FT plot: {str(e)}")
                try:
                    plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
                    print(f"Saved FT plot with reduced DPI (150)")
                except Exception as e2:
                    print(f"Failed to save FT plot even with reduced DPI: {str(e2)}")
            plt.close()
            plt.clf()  # Clear the figure to free memory
            
            return xf, magnitude

        def perform_wavelet_transform(eeg_signal, electrode_name):
            """
            Perform Continuous Wavelet Transform on the EEG signal and save results.
            
            Args:
                eeg_signal: The EEG signal array
                electrode_name: Name of the electrode
                
            Returns:
                time: Time array
                frequencies: Array of frequencies
                coefficients: Wavelet coefficients
            """
            # Remove NaN values
            eeg_signal = eeg_signal[~np.isnan(eeg_signal)]
            
            # No downsampling
            fs = SAMPLING_FREQ
            num_scales = 500  # Increase for better frequency resolution
            min_freq = 0.5
            max_freq = 40
            target_freqs = np.linspace(min_freq, max_freq, num_scales)
            scales = pywt.central_frequency('morl') * fs / target_freqs
            
            # Perform Continuous Wavelet Transform
            print(f"\nCalculating wavelet transform for {electrode_name}...")
            coefficients, frequencies = pywt.cwt(eeg_signal, scales, 'morl', sampling_period=1/fs)
            
            # Print detailed frequency information
            print(f"\nDetailed frequency information for {electrode_name}:")
            print(f"Total number of frequencies: {len(frequencies)}")
            print(f"Frequency range: {frequencies.min():.2f} Hz to {frequencies.max():.2f} Hz")
            print("\nFrequency distribution:")
            for freq in np.linspace(0, 40, 9):
                count = np.sum((frequencies >= freq) & (frequencies < freq + 5))
                print(f"{freq:5.1f}-{freq+5:5.1f} Hz: {count} frequencies")
            
            # Filter frequencies to match noise reduction limits
            freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
            frequencies = frequencies[freq_mask]
            coefficients = coefficients[freq_mask]
            
            # Create time array (no downsampling)
            time = np.arange(len(eeg_signal)) * (1/fs)
            
            # Create electrode-specific folder for wavelet plots
            electrode_wavelet_folder = os.path.join(WAVELET_FOLDER, electrode_name.replace(' ', '_'))
            os.makedirs(electrode_wavelet_folder, exist_ok=True)
            
            # Plot and save wavelet results
            frequency_ranges = [
                (0.5, 4, 'Delta'),
                (4, 8, 'Theta'),
                (8, 13, 'Alpha'),
                (13, 30, 'Beta'),
                (30, 40, 'Gamma')
            ]
            
            print("\nFrequency bands analysis:")
            for freq_min, freq_max, band_name in frequency_ranges:
                mask = (frequencies >= freq_min) & (frequencies <= freq_max)
                if any(mask):
                    band_freqs = frequencies[mask]
                    band_coeffs = coefficients[mask]
                    print(f"- {band_name} ({freq_min}-{freq_max} Hz):")
                    print(f"  Number of frequencies: {len(band_freqs)}")
                    print(f"  Range: {band_freqs.min():.2f} Hz to {band_freqs.max():.2f} Hz")
                    print(f"  Average magnitude: {np.mean(np.abs(band_coeffs)):.2f}")
                    print(f"  Max magnitude: {np.max(np.abs(band_coeffs)):.2f}")
                else:
                    print(f"- {band_name} ({freq_min}-{freq_max} Hz): No frequencies found")
            
            # Create a combined plot showing all bands
            plt.figure(figsize=(15, 8))
            for freq_min, freq_max, band_name in frequency_ranges:
                mask = (frequencies >= freq_min) & (frequencies <= freq_max)
                if not any(mask):
                    print(f"Warning: No frequencies found in {band_name} band ({freq_min}-{freq_max} Hz)")
                    continue
                band_coefficients = np.abs(coefficients[mask])
                avg_magnitude = np.mean(band_coefficients, axis=0)
                plt.plot(time, avg_magnitude, label=f'{band_name} Band ({freq_min}-{freq_max} Hz)')
            plt.title(f'Wavelet Transform - {electrode_name} - All Bands')
            plt.xlabel('Time (s)')
            plt.ylabel('Average Magnitude')
            plt.grid(True)
            plt.legend()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"wavelet_{electrode_name.replace(' ', '_')}_all_bands_{timestamp}.png"
            plot_filepath = os.path.join(electrode_wavelet_folder, plot_filename)
            try:
                plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error saving wavelet plot: {str(e)}")
                try:
                    plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
                    print(f"Saved wavelet plot with reduced DPI (150)")
                except Exception as e2:
                    print(f"Failed to save wavelet plot even with reduced DPI: {str(e2)}")
            plt.close()
            plt.clf()
            for freq_min, freq_max, band_name in frequency_ranges:
                mask = (frequencies >= freq_min) & (frequencies <= freq_max)
                if not any(mask):
                    continue
                band_coefficients = np.abs(coefficients[mask])
                avg_magnitude = np.mean(band_coefficients, axis=0)
                plt.figure(figsize=(12, 6))
                plt.plot(time, avg_magnitude, label=f'{band_name} Band ({freq_min}-{freq_max} Hz)')
                plt.title(f'Wavelet Transform - {electrode_name} - {band_name} Band')
                plt.xlabel('Time (s)')
                plt.ylabel('Average Magnitude')
                plt.grid(True)
                plt.legend()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"wavelet_{electrode_name.replace(' ', '_')}_{band_name}_{timestamp}.png"
                plot_filepath = os.path.join(electrode_wavelet_folder, plot_filename)
                try:
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                except Exception as e:
                    print(f"Error saving {band_name} band plot: {str(e)}")
                    try:
                        plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
                        print(f"Saved {band_name} band plot with reduced DPI (150)")
                    except Exception as e2:
                        print(f"Failed to save {band_name} band plot even with reduced DPI: {str(e2)}")
                plt.close()
                plt.clf()
            return time, frequencies, np.abs(coefficients)

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
        
        # Print available electrodes
        print("\nProcessing the following electrodes:")
        for col in eeg_df.columns:
            print(f"- {col}")

        # Dictionaries to store results
        ft_results = {}
        ft_dataframes = []
        wavelet_results = {}
        wavelet_dataframes = []

        # First, perform Fourier Transform for all electrodes
        print("\nPerforming Fourier Transform analysis...")
        for electrode in eeg_df.columns:
            if electrode != 'EKG-REF':  # Skip the EKG reference channel
                print(f"Processing FT for electrode: {electrode}")
                eeg_signal = eeg_df[electrode].values
                frequencies, magnitudes = perform_fourier_transform(eeg_signal, electrode)
                
                # Store FT results
                ft_results[electrode] = {
                    'frequencies': frequencies,
                    'magnitudes': magnitudes
                }
                
                # Create DataFrame for this electrode's FT
                df = pd.DataFrame({
                    'Frequency (Hz)': frequencies,
                    f'{electrode} Magnitude': magnitudes
                })
                ft_dataframes.append(df)

        # Combine all FT results into a single DataFrame
        print("\nCombining FT results...")
        if ft_dataframes:
            # Start with the first DataFrame
            combined_ft_df = ft_dataframes[0]
            
            # Merge remaining DataFrames on frequency
            for df in ft_dataframes[1:]:
                combined_ft_df = pd.merge(combined_ft_df, df, on='Frequency (Hz)', how='outer')
            
            # Sort by frequency
            combined_ft_df = combined_ft_df.sort_values('Frequency (Hz)')
            
            # Save to Excel
            ft_excel_path = os.path.join(FT_FOLDER, "fourier_transform_results.xlsx")
            combined_ft_df.to_excel(ft_excel_path, index=False)
            print(f"Fourier Transform results saved to: {ft_excel_path}")
        else:
            print("No FT results to save.")

        # Perform Wavelet Transform for all electrodes
        print("\nPerforming Wavelet Transform analysis...")
        for electrode in eeg_df.columns:
            if electrode != 'EKG-REF':  # Skip the EKG reference channel
                print(f"Processing Wavelet Transform for electrode: {electrode}")
                eeg_signal = eeg_df[electrode].values
                time, frequencies, coefficients = perform_wavelet_transform(eeg_signal, electrode)
                
                # Store wavelet results
                wavelet_results[electrode] = {
                    'time': time,
                    'frequencies': frequencies,
                    'coefficients': coefficients
                }
                
                # Create DataFrame for this electrode's wavelet transform
                # Downsample the data to reduce size
                time_step = max(1, len(time) // 1000)  # Target about 1000 time points
                
                # For frequencies, we want to keep more points in the lower frequencies
                # where EEG activity is most relevant
                freq_mask = (frequencies >= 0.5) & (frequencies <= 30)  # Keep frequencies from 0.5 to 30 Hz
                downsampled_freq = frequencies[freq_mask]
                downsampled_coeffs = coefficients[freq_mask]
                
                # Create downsampled time array
                downsampled_time = time[::time_step]
                
                # Create meshgrid for downsampled data
                time_grid, freq_grid = np.meshgrid(downsampled_time, downsampled_freq)
                
                # Downsample coefficients
                downsampled_coeffs = downsampled_coeffs[:, ::time_step]
                
                # Create DataFrame with downsampled data
                df = pd.DataFrame({
                    'Time (s)': time_grid.flatten(),
                    'Frequency (Hz)': freq_grid.flatten(),
                    f'{electrode} Magnitude': downsampled_coeffs.flatten()
                })
                wavelet_dataframes.append(df)

        # Combine all wavelet results into a single DataFrame
        print("\nCombining Wavelet Transform results...")
        if wavelet_dataframes:
            # Start with the first DataFrame
            combined_wavelet_df = wavelet_dataframes[0]
            
            # Merge remaining DataFrames on time and frequency
            for df in wavelet_dataframes[1:]:
                combined_wavelet_df = pd.merge(combined_wavelet_df, df, 
                                            on=['Time (s)', 'Frequency (Hz)'], 
                                            how='outer')
            
            # Sort by time and frequency
            combined_wavelet_df = combined_wavelet_df.sort_values(['Time (s)', 'Frequency (Hz)'])
            
            # Save to Excel
            wavelet_excel_path = os.path.join(WAVELET_FOLDER, "wavelet_transform_results.xlsx")
            combined_wavelet_df.to_excel(wavelet_excel_path, index=False)
            print(f"Wavelet Transform results saved to: {wavelet_excel_path}")
        else:
            print("No Wavelet Transform results to save.")
        
        print("\nAnalysis complete. All transform results have been saved.")
        print(f"Results are saved in: {PARENT_FOLDER}")
        print(f"- Frequency domain plots and results are in: {FT_FOLDER}")
        print(f"- Time-frequency domain plots are in: {WAVELET_FOLDER}")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        continue
