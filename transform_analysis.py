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
MOVEMENTS_FOLDER = "C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/nr/002- Ayush"  
OUTPUT_BASE_FOLDER = r"C:/Users/pande/OneDrive/Desktop/eeg sem projet/eeg code/ft_wt_transform/007"

def bcitransform(MOVEMENTS_FOLDER, OUTPUT_BASE_FOLDER):
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

            # Create plots subfolder within the parent folder
            #PLOTS_FOLDER = os.path.join(PARENT_FOLDER, "plots23")
            #os.makedirs(PLOTS_FOLDER, exist_ok=True)

            # Create Fourier Transform subfolder
            FT_FOLDER = os.path.join(PARENT_FOLDER, "fourier_transform")
            os.makedirs(FT_FOLDER, exist_ok=True)

            # Create Wavelet Transform subfolder
            WAVELET_FOLDER = os.path.join(PARENT_FOLDER, "wavelet_transform")
            os.makedirs(WAVELET_FOLDER, exist_ok=True)

            print(f"Output will be saved in: {PARENT_FOLDER}")
            #print(f"Plots will be saved in: {PLOTS_FOLDER}")
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

            def process_electrode(eeg_signal, electrode_name):
                try:
                    # Remove NaN values
                    eeg_signal = eeg_signal[~np.isnan(eeg_signal)]
                    
                    # Calculate total duration
                    total_duration = len(eeg_signal) / SAMPLING_FREQ
                    num_windows = int(np.ceil(total_duration / TIME_WINDOW))
                    
                    print(f"\nProcessing {electrode_name}:")
                    print(f"Total duration: {total_duration:.2f} seconds")
                    print(f"Number of samples: {len(eeg_signal)}")
                    print(f"Number of windows to create: {num_windows}")
                    
                    # Create electrode-specific folder
                    #electrode_folder = os.path.join(PLOTS_FOLDER, electrode_name.replace(' ', '_'))
                    #os.makedirs(electrode_folder, exist_ok=True)
                    
                    # Process each time window
                    for window_idx in range(num_windows):
                        # Calculate start and end time for this window
                        start_time = window_idx * TIME_WINDOW
                        end_time = min(start_time + TIME_WINDOW, total_duration)
                        
                        print(f"Creating window {window_idx + 1}/{num_windows}: {start_time:.1f}s - {end_time:.1f}s")
                        
                        # Calculate start and end indices
                        start_idx = int(start_time * SAMPLING_FREQ)
                        end_idx = int(end_time * SAMPLING_FREQ)
                        
                        # Get the signal segment for this window
                        window_signal = eeg_signal[start_idx:end_idx]
                        
                        # Create time array for this window
                        window_time = np.linspace(start_time, end_time, len(window_signal))
                        
                        # Create plot for this window
                        plt.figure(figsize=(8, 4))  # Slightly smaller width for 0.1s window
                        plt.plot(window_time, window_signal, label='EEG Signal', color='purple', linewidth=1)
                        
                        # Create title with time duration information
                        title = f'EEG Signal for {electrode_name}\n'
                        title += f'Time: {start_time:.2f}s - {end_time:.2f}s (Window {window_idx + 1}/{num_windows})'
                        plt.title(title, fontsize=12)
                        
                        plt.xlabel('Time (seconds)', fontsize=10)
                        plt.ylabel('Amplitude (µV)', fontsize=10)
                        
                        # Set x-axis ticks every 0.01 seconds
                        plt.xticks(np.arange(start_time, end_time + TIME_TICK_INTERVAL, TIME_TICK_INTERVAL))
                        plt.yticks(fontsize=8)
                        
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        plt.legend(fontsize=10)
                        
                        # Set y-axis limits with some padding
                        window_y_min = np.min(window_signal) - 0.1 * np.abs(np.min(window_signal))
                        window_y_max = np.max(window_signal) + 0.1 * np.abs(np.max(window_signal))
                        plt.ylim(window_y_min, window_y_max)
                        
                        plt.tight_layout()
                        
                        # Save the plot in electrode-specific folder
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        plot_filename = f"window_{window_idx + 1}_{timestamp}.png"
                        #plot_filepath = os.path.join(electrode_folder, plot_filename)
                        #plt.savefig(plot_filepath, dpi=600, bbox_inches='tight')
                        #print(f"Saved plot for {electrode_name} window {window_idx + 1} to: {plot_filepath}")
                        
                        plt.close()
                    
                    print(f"\nCompleted processing {electrode_name}")
                    print(f"Total windows created: {num_windows}")
                    return True
                except Exception as e:
                    print(f"Error processing electrode {electrode_name}: {str(e)}")
                    return False

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
                plt.savefig(plot_filepath, dpi=600, bbox_inches='tight')
                plt.close()
                
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
                
                # Downsample the signal to reduce computation time
                # Keep every 4th sample (reducing sampling rate to 256 Hz)
                downsampling_factor = 4
                eeg_signal = eeg_signal[::downsampling_factor]
                
                # Define scales for wavelet transform
                # Match the noise reduction filter limits (0.5 Hz to 40 Hz)
                min_freq = 0.5  # Minimum frequency in Hz
                max_freq = 40   # Maximum frequency in Hz
                
                # Calculate the number of scales needed for good frequency resolution
                num_scales = 200  # Number of scales for frequency resolution
                
                # Calculate scales that will give us our desired frequency range
                # Using a logarithmic scale to get better resolution in lower frequencies
                scales = np.logspace(np.log10((SAMPLING_FREQ/downsampling_factor * 6) / max_freq),
                                np.log10((SAMPLING_FREQ/downsampling_factor * 6) / min_freq),
                                num_scales)
                
                # Perform Continuous Wavelet Transform
                print(f"\nCalculating wavelet transform for {electrode_name}...")
                coefficients, frequencies = pywt.cwt(eeg_signal, scales, 'morl', sampling_period=downsampling_factor/SAMPLING_FREQ)
                
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
                
                # Create time array (adjusted for downsampling)
                time = np.arange(len(eeg_signal)) * (downsampling_factor/SAMPLING_FREQ)
                
                # Create electrode-specific folder for wavelet plots
                electrode_wavelet_folder = os.path.join(WAVELET_FOLDER, electrode_name.replace(' ', '_'))
                os.makedirs(electrode_wavelet_folder, exist_ok=True)
                
                # Plot and save wavelet results
                # Create separate plots for different frequency ranges
                frequency_ranges = [
                    (0.5, 4, 'Delta'),
                    (4, 8, 'Theta'),
                    (8, 13, 'Alpha'),
                    (13, 30, 'Beta'),
                    (30, 40, 'Gamma')
                ]
                
                # Print frequency band information
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
                    # Find indices for this frequency range
                    mask = (frequencies >= freq_min) & (frequencies <= freq_max)
                    if not any(mask):
                        print(f"Warning: No frequencies found in {band_name} band ({freq_min}-{freq_max} Hz)")
                        continue
                    
                    # Get coefficients for this frequency range
                    band_coefficients = np.abs(coefficients[mask])
                    band_frequencies = frequencies[mask]
                    
                    # Calculate average magnitude for this band
                    avg_magnitude = np.mean(band_coefficients, axis=0)
                    
                    # Plot this band
                    plt.plot(time, avg_magnitude, label=f'{band_name} Band ({freq_min}-{freq_max} Hz)')
                
                plt.title(f'Wavelet Transform - {electrode_name} - All Bands')
                plt.xlabel('Time (s)')
                plt.ylabel('Average Magnitude')
                plt.grid(True)
                plt.legend()
                
                # Save the combined plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"wavelet_{electrode_name.replace(' ', '_')}_all_bands_{timestamp}.png"
                plot_filepath = os.path.join(electrode_wavelet_folder, plot_filename)
                plt.savefig(plot_filepath, dpi=600, bbox_inches='tight')
                plt.close()
                
                # Create individual plots for each band
                for freq_min, freq_max, band_name in frequency_ranges:
                    # Find indices for this frequency range
                    mask = (frequencies >= freq_min) & (frequencies <= freq_max)
                    if not any(mask):
                        continue
                    
                    # Get coefficients for this frequency range
                    band_coefficients = np.abs(coefficients[mask])
                    band_frequencies = frequencies[mask]
                    
                    # Calculate average magnitude for this band
                    avg_magnitude = np.mean(band_coefficients, axis=0)
                    
                    # Create plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(time, avg_magnitude, label=f'{band_name} Band ({freq_min}-{freq_max} Hz)')
                    
                    plt.title(f'Wavelet Transform - {electrode_name} - {band_name} Band')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Average Magnitude')
                    plt.grid(True)
                    plt.legend()
                    
                    # Save the plot
                    '''timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = f"wavelet_{electrode_name.replace(' ', '_')}_{band_name}_{timestamp}.png"
                    plot_filepath = os.path.join(electrode_wavelet_folder, plot_filename)
                    plt.savefig(plot_filepath, dpi=600, bbox_inches='tight')
                    plt.close()'''
                
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

            # Now process time domain plots
            '''print("\nProcessing time domain plots...")
            for electrode in eeg_df.columns:
                if electrode != 'EKG-REF':  # Skip the EKG reference channel
                    print(f"Processing time domain plots for electrode: {electrode}")
                    eeg_signal = eeg_df[electrode].values
                    process_electrode(eeg_signal, electrode)
            
            print("\nAnalysis complete. All plots and transform results have been saved.")
            print(f"Results are saved in: {PARENT_FOLDER}")
            print(f"- Time domain plots are in: {PLOTS_FOLDER}")
            print(f"- Frequency domain plots and results are in: {FT_FOLDER}")
            print(f"- Time-frequency domain plots are in: {WAVELET_FOLDER}")'''
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            continue


